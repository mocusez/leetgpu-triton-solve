### 一、 Triton GPU 内核：`binoticsort_des_kernel`

这个 Kernel 负责在 GPU 上执行单次的“比较并交换（Compare-and-Swap）”操作。由于每个线程同时处理 2 个元素，因此 `BLOCK_SIZE` 个线程可以处理 `BLOCK_SIZE * 2` 个元素。

```python
@triton.jit
def binoticsort_des_kernel(
    input_ptr, N,          # input_ptr: 数据指针; N: 数据的实际长度
    stage, stride,         # stage: 当前构建的双调序列的长度; stride: 比较元素的跨度
    BLOCK_SIZE: tl.constexpr # BLOCK_SIZE: 编译期常量，每个 block 的线程数
):
    # 1. 计算当前线程的全局唯一偏移量 (offset)
    pid = tl.program_id(axis = 0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

```

* **`pid`**: 获取当前 block 的 ID。
* **`offset`**: 计算当前 block 中每个线程处理的逻辑索引（从 0 到 `BLOCK_SIZE-1` 加上 block 偏移）。这里的 `offset` 不是直接的数组下标，而是“**第几个比较对**”。

```python
    # 2. 计算需要比较的两个元素的实际内存索引
    slice_1_offset = (offset // stride) * 2 * stride + (offset % stride)
    slice_2_offset = slice_1_offset + stride

```

* 这是整个程序最精妙的数学映射。由于步长为 `stride` 时，比较操作会在数组中跳跃进行（例如 `stride=2` 时，比较 0和2，1和3，然后跳过 2和3，接着比较 4和6，5和7）。
* **`slice_1_offset`**: 通过上述公式，将连续的线程 `offset` 完美映射到了需要比较的**第一个元素**的索引上，避免了线程间的冲突。
* **`slice_2_offset`**: 需要比较的**第二个元素**索引，恰好在第一个元素的基础上加上步长 `stride`。

```python
    # 3. 从显存中加载数据
    slice_1_t = tl.load(input_ptr + slice_1_offset, mask = slice_1_offset < N, other = -float('inf'))
    slice_2_t = tl.load(input_ptr + slice_2_offset, mask = slice_2_offset < N, other = -float('inf'))

```

* 使用计算出的索引加载数据。
* **`mask`** 和 **`other`**: 防止越界读取。如果索引超出了数组长度 `N`，则用负无穷 `-float('inf')` 填充。由于我们要做降序排序，用负无穷填充可以保证这些越界数据永远排在最后。

```python
    # 4. 判断当前的排序方向 (升序还是降序)
    descend = (slice_1_offset // stage) % 2 == 1
    greater = slice_1_t > slice_2_t

```

* **`descend`**: 双调排序需要交替构建升序和降序块。通过 `(slice_1_offset // stage) % 2` 可以判断当前元素所处的块是偶数块还是奇数块。
* 在这里，`descend == False` 代表目标是**降序**（把较大的元素放前面）。
* `descend == True` 代表目标是**升序**（把较小的元素放前面）。


* **`greater`**: 记录第一个元素是否大于第二个元素。

```python
    # 5. 执行条件交换 (核心排序逻辑)
    new_slice_1_t = tl.where(descend == greater, slice_2_t, slice_1_t)
    new_slice_2_t = tl.where(descend == greater, slice_1_t, slice_2_t)

```

* 利用 `tl.where(condition, x, y)` 进行无分支的条件赋值。如果 `condition` 为真，返回 `x`，否则返回 `y`。
* **逻辑推演**：
* 如果目标是**降序 (`descend = False`)**：当 `slice_1 > slice_2` (`greater = True`) 时，`False == True` 为假，`new_slice_1` 保持 `slice_1`，较大的留在前面（不交换）。当 `slice_1 < slice_2` 时，触发交换，把较大的换到前面。
* 如果目标是**升序 (`descend = True`)**：当 `slice_1 > slice_2` (`greater = True`) 时，`True == True` 为真，`new_slice_1` 变成 `slice_2`，触发交换，把较小的换到前面。



```python
    # 6. 将排序后的结果写回显存
    tl.store(input_ptr + slice_1_offset, new_slice_1_t, mask = slice_1_offset < N)
    tl.store(input_ptr + slice_2_offset, new_slice_2_t, mask = slice_2_offset < N)

```

* 使用原位（In-place）更新的方式，将比较并交换后的结果写回原数组。

---

### 二、 Python 主控函数：`solve`

这个函数是宿主机（CPU）调用 GPU kernel 的入口，负责准备数据、计算网格并控制双调排序的循环阶段。

```python
def solve(input: torch.Tensor, output: torch.Tensor, N: int, k: int):
    # 1. 补齐到 2 的幂次方
    paddingLen = triton.next_power_of_2(N)
    inputPadding = torch.zeros((paddingLen), device = input.device, dtype = input.dtype)
    inputPadding[:N] = input
    inputPadding[N:] = -float('inf')

```

* 双调排序算法要求数组长度必须是 **2 的幂次方**。
* `triton.next_power_of_2(N)` 找到大于等于 `N` 的最小 2 的幂。
* 申请一个新张量 `inputPadding`，将原数据拷贝进去，超出 `N` 的部分全部填充为 `-inf`。因为我们要的是降序排序，`-inf` 自然会被挤到数组的最末尾，不会干扰 Top-K 的结果。

```python
    # 2. 配置 Triton Grid (线程块分发)
    BLOCK_SIZE = 1024
    grid = lambda metadata: (triton.cdiv(paddingLen, metadata['BLOCK_SIZE'] * 2),)

```

* 设定每个 Block 有 1024 个线程。
* `grid` 函数计算启动多少个 block。由于每个线程处理 2 个元素（slice_1 和 slice_2），所以每个 block 能处理 `BLOCK_SIZE * 2` 个元素。总共需要的 block 数量为向上取整的 `paddingLen / (BLOCK_SIZE * 2)`。

```python
    # 3. 双调排序的多重循环 (调度 Kernel)
    stage = 2
    while stage <= paddingLen:
        stride = (stage >> 1)       # 等价于 stage // 2
        while stride:
            binoticsort_des_kernel[grid](inputPadding, paddingLen, stage, stride, BLOCK_SIZE)
            stride >>= 1            # 等价于 stride //= 2
        stage <<= 1                 # 等价于 stage *= 2

```

* 这是标准的双调排序外部调度逻辑，包含两层循环：
* **外层循环 (`stage`)**：决定当前构建的双调序列大小，从 2 开始，按 2 的倍数递增（4, 8, 16...），直到达到数组全长 `paddingLen`。
* **内层循环 (`stride`)**：在当前的 `stage` 下，比较步长从 `stage / 2` 开始，每次减半，直到 1。


* 每一次内层循环都会启动一次 `binoticsort_des_kernel`，在 GPU 上对所有数据进行一次完全并行的“比较-交换”网络路由。
* 当 `stage = paddingLen` 且执行完毕后，整个数组 `inputPadding` 就完成了**全局降序排序**。

```python
    # 4. 提取 Top-K
    output.copy_(inputPadding[:k])

```

* 因为数组已经降序排列（且补齐的 `-inf` 都在最后面），所以直接切片截取前 `k` 个元素并拷贝给 `output` 即可完成 Top-K 操作。