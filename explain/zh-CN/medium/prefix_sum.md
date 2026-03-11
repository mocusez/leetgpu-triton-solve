# Prefix Sum

这段代码是经典的**Blelloch 扫描算法（Reduce-then-Scan）**在 Triton 中的实现。我们将它分为四个部分逐行拆解：三个 GPU 内核（Kernel）和一个 CPU 宿主函数（Host Function）。

---

### 1. `single_block_scan_kernel` (基础情况：单块扫描)

当数组非常小（长度 $\le$ `BLOCK_SIZE`）时，我们不需要拆分，直接用一个线程块搞定。

```python
@triton.jit
def single_block_scan_kernel(data_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    # 生成从 0 到 BLOCK_SIZE-1 的索引数组：[0, 1, 2, ..., 1023]
    offsets = tl.arange(0, BLOCK_SIZE)
    # 创建掩码 (mask)，防止越界访问。例如 n=500 时，索引 >= 500 的位置 mask 为 False
    mask = offsets < n
    
    # 根据 offsets 从显存中加载数据。如果越界 (mask=False)，则用 0.0 填充，不影响求和
    x = tl.load(data_ptr + offsets, mask=mask, other=0.0)
    
    # 调用 Triton 内置函数，在 SRAM (共享内存/寄存器) 中极速计算这 1024 个元素的前缀和
    res = tl.cumsum(x, axis=0)
    
    # 将计算好的前缀和写回全局显存的 output 数组中
    tl.store(output_ptr + offsets, res, mask=mask)

```

---

### 2. `local_scan_kernel` (第一阶段：局部扫描与提取块总和)

当数组很大时，我们将它切分成多个块。每个线程块负责处理自己那一部分数据。

```python
@triton.jit
def local_scan_kernel(data_ptr, output_ptr, block_sums_ptr, n, BLOCK_SIZE: tl.constexpr):
    # 获取当前线程块的 ID (Program ID)。例如第 0 块、第 1 块...
    pid = tl.program_id(axis=0)
    
    # 计算当前块在整个大数组中的起始物理位置。例如 pid=1, BLOCK_SIZE=1024，起始点就是 1024
    block_start = pid * BLOCK_SIZE
    
    # 计算当前块需要处理的绝对索引
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    # 加载当前块负责的这段数据
    x = tl.load(data_ptr + offsets, mask=mask, other=0.0)
    
    # 【核心操作 1】：计算这段数据的内部前缀和 (局部前缀和)，并写回 output
    # 注意：此时 output 里存的还不是全局正确结果。比如第 1 块算出来的没有加上第 0 块的总和
    local_scan = tl.cumsum(x, axis=0)
    tl.store(output_ptr + offsets, local_scan, mask=mask)
    
    # 【核心操作 2】：计算这段数据所有元素的总和
    block_sum = tl.sum(x, axis=0)
    
    # 将这个总和单独存入 block_sums_ptr 数组的第 pid 个位置
    # 这样 CPU 就能知道每个小块的数据总量是多少了
    tl.store(block_sums_ptr + pid, block_sum)

```

---

### 3. `add_base_kernel` (第三阶段：累加基础偏移量)

在第二阶段（CPU 中通过递归完成）算出各个块总和的前缀和之后，我们需要把这些“前面块的总和”加到当前块的局部前缀和上。

```python
@triton.jit
def add_base_kernel(output_ptr, scanned_block_sums_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    # 第 0 块的数据前面没有任何块，它的局部前缀和就是全局前缀和，所以直接退出
    if pid == 0:
        return
        
    # 计算当前块的物理索引和掩码 (与阶段 1 完全相同)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # 【关键】：读取当前块的 Base 值。
    # 为什么是 pid - 1？因为第 pid 块需要的是 "它之前所有块" 的总和。
    # scanned_block_sums 已经是前缀和数组了，所以索引 pid - 1 刚好存着从块 0 到块 pid-1 的总和。
    base = tl.load(scanned_block_sums_ptr + pid - 1)
    
    # 从 output 中读出第一阶段算好的、残缺的 "局部前缀和"
    local_scan = tl.load(output_ptr + offsets, mask=mask)
    
    # 加上之前所有块的总和 (base)，得到真正的 "全局前缀和"
    global_scan = local_scan + base
    
    # 覆盖写回 output 数组的对应位置
    tl.store(output_ptr + offsets, global_scan, mask=mask)

```

---

### 4. `solve` 宿主函数 (控制流与魔法递归)

这是在 CPU 上运行的 Python 代码，负责显存分配和调度 GPU Kernel。

```python
def solve(data: torch.Tensor, output: torch.Tensor, n: int):
    # 边界保护
    if n <= 0: return
        
    BLOCK_SIZE = 1024
    
    # ==========================================
    # 递归的终点 (Base Case)
    # 如果数据量 <= 1024，直接启动 1 个线程块算完就结束。
    # ==========================================
    if n <= BLOCK_SIZE:
        single_block_scan_kernel[(1,)](data, output, n, BLOCK_SIZE=BLOCK_SIZE)
        return

    # ==========================================
    # 数据量 > 1024 时的多阶段处理
    # ==========================================
    
    # 计算需要多少个线程块。triton.cdiv 是向上取整除法，例如 1025 / 1024 = 2 个块
    num_blocks = triton.cdiv(n, BLOCK_SIZE)
    
    # 动态分配两块显存，大小等于线程块的数量 (num_blocks)
    # block_sums: 存每个块的原始总和
    # scanned_block_sums: 存块总和的前缀和结果
    block_sums = torch.empty((num_blocks,), dtype=data.dtype, device=data.device)
    scanned_block_sums = torch.empty((num_blocks,), dtype=data.dtype, device=data.device)
    
    # 【阶段 1】：启动 num_blocks 个线程块，并行计算局部前缀和与块总和
    local_scan_kernel[(num_blocks,)](data, output, block_sums, n, BLOCK_SIZE=BLOCK_SIZE)
    
    # 【阶段 2：魔法递归】：
    # 此时我们有了一个由各个块总和组成的数组 `block_sums`。我们需要求它的前缀和。
    # 怎么求？直接把这个任务再次交给 solve() 函数自己！
    # 无论 num_blocks 是 10（直接被 base case 处理），还是 10 万（再次被切分），递归都能完美处理。
    solve(block_sums, scanned_block_sums, num_blocks)
    
    # 【阶段 3】：启动 num_blocks 个线程块，把阶段 2 算好的基准值 (Base) 累加回去
    add_base_kernel[(num_blocks,)](output, scanned_block_sums, n, BLOCK_SIZE=BLOCK_SIZE)

```

通过这种**局部计算 -> 提取宏观特征(块总和) -> 递归处理宏观特征 -> 补全局部计算**的思想，我们将一个无法跨块同步的串行问题，完美转化成了可以打满 GPU 算力的并行分治问题。

需要我提供一段包含断言测试的代码，让你直接运行验证这段逻辑处理长数组（比如 100 万长度）的准确性吗？