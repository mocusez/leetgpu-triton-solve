### 1. 核心状态机：`seg_scan_combine`

这个函数是告诉 GPU：**“当你把两个小块的数据合并在一起时，应该遵循什么规则？”**

```python
@triton.jit
def seg_scan_combine(v1, f1, v2, f2):
    # v1, f1 是左边的数据和标记；v2, f2 是右边的数据和标记。
    # 规则1：如果右侧块 (f2) 包含一个新段的开头 (f2 == 1)，
    # 那么跨越这两个块的前缀和应该被“切断”，右侧的总和就是它自己 (v2)。
    # 否则，总和等于左边加右边 (v1 + v2)。
    v_out = tl.where(f2, v2, v1 + v2)
    
    # 规则2：只要左边或右边任意一块里出现过新段标记(1)，
    # 合并后的大块在对外报告时，就要宣告“我这里面包含新段”。(使用按位或 |)
    f_out = f1 | f2
    
    return v_out, f_out

```

---

### 2. 第一阶段：局部规约 `pass1_reduce`

这部分的任务是：**把一个巨大的 Tile（65536长度）浓缩成 1 个数字（这个 Tile 的总和）**。

```python
@triton.jit
def pass1_reduce(values_ptr, flags_ptr, tile_sums_ptr, tile_flags_ptr, N, TILE_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # tl.program_id(0) 获取当前 GPU Block 的编号 (也就是 Tile 的编号)
    pid = tl.program_id(0)
    # 计算当前 Tile 在整个原数组中的起始索引
    tile_start = pid * TILE_SIZE
    
    # 在寄存器中初始化当前 Tile 的累加器
    acc_sum = 0.0
    acc_flag = 0
    
    # 为什么要有这个 for 循环？
    # 因为 GPU SRAM 存不下 65536 这么大的 Tile。我们每次只处理 BLOCK_SIZE (4096) 个元素。
    for i in range(0, TILE_SIZE, BLOCK_SIZE):
        # 计算这 4096 个元素的绝对索引
        offsets = tile_start + i + tl.arange(0, BLOCK_SIZE)
        # 防止最后一个 Tile 越界读取
        mask = offsets < N
        
        # 从全局显存加载数据到片上 SRAM
        vals = tl.load(values_ptr + offsets, mask=mask, other=0.0)
        flgs = tl.load(flags_ptr + offsets, mask=mask, other=0)
        flgs_bool = flgs == 1
        
        # 调用 Triton 底层的高效扫描算子，计算这 4096 个元素内部的前缀和
        loc_sums, loc_flags = tl.associative_scan((vals, flgs_bool), axis=0, combine_fn=seg_scan_combine)
        
        # 下面三行是核心技巧：提取这 4096 个元素里的【最后一位】的状态！
        # 建立一个掩码，只有最后一位是 True
        mask_last = tl.arange(0, BLOCK_SIZE) == BLOCK_SIZE - 1
        # 把最后一位的值取出来，其他位变 0，然后求和（相当于安全地提取出那个标量值）
        chunk_sum = tl.sum(tl.where(mask_last, loc_sums, 0.0))
        chunk_flag = tl.max(tl.where(mask_last, tl.cast(loc_flags, tl.int32), 0))
        
        # 把这 4096 个元素的结果，累加到整个 Tile 的累加器中
        # 如果新块里有截断信号(chunk_flag == 1)，那之前的累加作废，直接用新的 chunk_sum
        acc_sum = chunk_sum if chunk_flag == 1 else acc_sum + chunk_sum
        acc_flag = acc_flag | chunk_flag
        
    # for循环结束，整个 Tile 的 65536 个元素被浓缩成了 acc_sum 和 acc_flag
    # 把它们写到显存里一个非常小的中间数组中，供 Pass 2 使用
    tl.store(tile_sums_ptr + pid, acc_sum)
    tl.store(tile_flags_ptr + pid, acc_flag)

```

---

### 3. 第二阶段：全局桥接 `pass2_scan`

这部分的任务是：**只用 1 个 Block，把上一步得到的浓缩数组（只有大约 1500 个元素）从头到尾扫一遍**。

```python
@triton.jit
def pass2_scan(tile_sums_ptr, tile_flags_ptr, NUM_TILES, BLOCK_SIZE: tl.constexpr):
    # 因为只有极少数元素，直接生成 0 ~ NUM_TILES 的索引
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < NUM_TILES
    
    # 加载中间数组
    sums = tl.load(tile_sums_ptr + offsets, mask=mask, other=0.0)
    flags = tl.load(tile_flags_ptr + offsets, mask=mask, other=0)
    flags_bool = flags == 1
    
    # 全局算一次前缀和
    # 这样第 i 个元素存的就是前 i 个 Tile 的总和（考虑了分段截断）
    inc_sums, inc_flags = tl.associative_scan((sums, flags_bool), axis=0, combine_fn=seg_scan_combine)
    
    # 覆盖写回原位置。现在中间数组里存放的是每个 Tile 对外的 "全局 Base (基底值)" 了
    tl.store(tile_sums_ptr + offsets, inc_sums, mask=mask)
    tl.store(tile_flags_ptr + offsets, tl.cast(inc_flags, tl.int32), mask=mask)

```

---

### 4. 第三阶段：向下传递与最终计算 `pass3_downsweep`

任务：**拿着 Pass 2 算出的全局 Base 回到原来的数组，计算最终结果并写出**。

```python
@triton.jit
def pass3_downsweep(values_ptr, flags_ptr, output_ptr, tile_sums_ptr, tile_flags_ptr, N, TILE_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    tile_start = pid * TILE_SIZE
    
    # 核心：获取当前 Tile 在全局视角下的起始基底值 (Base)
    if pid == 0:
        # 第0个Tile前面没有东西，基底是0
        acc_sum = 0.0
        acc_flag = 0
    else:
        # 其他Tile，读取它上一个 Tile (pid - 1) 在 Pass 2 里算出来的总和！
        acc_sum = tl.load(tile_sums_ptr + pid - 1)
        acc_flag = tl.load(tile_flags_ptr + pid - 1)
        
    # 再次遍历当前 Tile 内部的元素
    for i in range(0, TILE_SIZE, BLOCK_SIZE):
        offsets = tile_start + i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        
        # 重新加载原始数据
        vals = tl.load(values_ptr + offsets, mask=mask, other=0.0)
        flgs = tl.load(flags_ptr + offsets, mask=mask, other=0)
        flgs_bool = flgs == 1
        
        # 再次计算这 4096 个元素的局部前缀和
        loc_sums, loc_flags = tl.associative_scan((vals, flgs_bool), axis=0, combine_fn=seg_scan_combine)
        
        # 【关键步骤】：把全局基底加上去！
        # 如果当前小段本身自带新的分段标记 (loc_flags)，就直接用小段的(截断基底)
        # 否则，全局值 = 全局基底 (acc_sum) + 局部值 (loc_sums)
        global_inc_sums = tl.where(loc_flags, loc_sums, acc_sum + loc_sums)
        
        # 【最终转换】：题意要求Exclusive (互斥前缀和)。
        # 用我们刚算出来的 Inclusive Sum 减去当前位置原来的值，完美！
        exc_sums = global_inc_sums - vals
        
        # 写回到最终的输出数组 output 中
        tl.store(output_ptr + offsets, exc_sums, mask=mask)
        
        # 下面几行与 Pass 1 一样，更新累加器，传递给这个 Tile 内的下一个 4096 循环块
        mask_last = tl.arange(0, BLOCK_SIZE) == BLOCK_SIZE - 1
        chunk_sum = tl.sum(tl.where(mask_last, loc_sums, 0.0))
        chunk_flag = tl.max(tl.where(mask_last, tl.cast(loc_flags, tl.int32), 0))
        acc_sum = chunk_sum if chunk_flag == 1 else acc_sum + chunk_sum
        acc_flag = acc_flag | chunk_flag

```

---

### 5. 调度层：`solve` 函数

这部分运行在 CPU 端（PyTorch 中），负责分配显存并发射 GPU 内核。

```python
def solve(values: torch.Tensor, flags: torch.Tensor, output: torch.Tensor, N: int):
    # 定义切割策略：每个 GPU Block 负责一块 65536 长度的 Tile。
    # 每个 Tile 内部按 4096 的大小进行循环。
    TILE_SIZE = 65536
    BLOCK_SIZE = 4096
    
    # math.ceil(N / TILE_SIZE) 计算一共需要切分成多少个 Tile。
    # 假设 N 是一亿，这里 NUM_TILES 约等于 1526。
    NUM_TILES = math.ceil(N / TILE_SIZE)
    
    # 在 GPU 上开辟微小的中间过渡数组，大小仅为 1526。
    tile_sums = torch.empty(NUM_TILES, device=values.device, dtype=torch.float32)
    tile_flags = torch.empty(NUM_TILES, device=flags.device, dtype=torch.int32)
    
    # 启动 Pass 1：派发 NUM_TILES 个 Block（例如1526个）同时工作。
    # num_warps=8 意思是每个 Block 分配 256 个 GPU 线程。
    pass1_reduce[(NUM_TILES,)](...)
    
    # 启动 Pass 2：计算要处理的元素量，并向上取 2 的次幂（Triton scan 的要求）。
    # (1,) 意味着只派发 1 个单独的 Block 去把那 1526 个元素扫一遍。
    BLOCK_SIZE_2 = max(16, triton.next_power_of_2(NUM_TILES))
    pass2_scan[(1,)](...)
    
    # 启动 Pass 3：再次派发 NUM_TILES 个 Block，计算并写入最终结果。
    pass3_downsweep[(NUM_TILES,)](...)

```