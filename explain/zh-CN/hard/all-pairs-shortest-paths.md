## Kernel 部分

**函数签名**

```python
@triton.jit
def floyd_warshall(
    dist, stride_dm, stride_dn,
    output, stride_om, stride_on,
    N, k, BLOCK_N: tl.constexpr,
):
```

- `@triton.jit`：声明为 Triton kernel，首次调用时按参数类型即时编译。
- `dist`：距离矩阵指针；`stride_dm/stride_dn` 是它的行/列步长（元素为单位），用来支持非连续内存的 tensor。
- `output` 及其步长：**实际没被使用**——kernel 全程在 `dist` 上原地读写，`output` 是死参数，最后靠 host 端 `output.copy_(dist)` 收尾。
- `N`：顶点数；`k`：本轮的中间顶点（每次 launch 传入一个标量）。
- `BLOCK_N: tl.constexpr`：tile 边长，编译期常量，决定每个 program 处理 `BLOCK_N × BLOCK_N` 的子块。

**program 编号与全局索引**

```python
pid_i = tl.program_id(0)
pid_j = tl.program_id(1)
i_offs = tl.arange(0, BLOCK_N) + pid_i * BLOCK_N
j_offs = tl.arange(0, BLOCK_N) + pid_j * BLOCK_N
```

- 2D grid 中每个 program 负责一块 tile；`pid_i`、`pid_j` 是 tile 的行/列编号。
- `i_offs`：该 tile 覆盖的全局行号 `[pid_i*B, pid_i*B+B)`，形状 `(B,)`；`j_offs` 同理为列号。

**k 向量**

```python
k_offs = tl.full((BLOCK_N,), k, dtype=tl.int32)
```

构造一个长度 B、值全为 k 的向量。作用是让下面取"第 k 列/第 k 行"时也保持 2D 广播形状（`(B,1)`、`(1,B)`），和 `dist_ij` 的 `(B,B)` 对齐。

**加载 dist[i, j] 块**

```python
dist_ptrs = dist + i_offs[:, None]*stride_dm + j_offs[None, :]*stride_dn
dist_mask = (i_offs[:, None] < N) & (j_offs[None, :] < N)
dist_ij = tl.load(dist_ptrs, mask=dist_mask, other=float('inf'))
```

- `i_offs[:, None]` 是 `(B,1)`、`j_offs[None, :]` 是 `(1,B)`，广播成 `(B,B)` 的指针网格，即 `&dist[i][j]`。
- mask 处理 N 不能被 64 整除时的越界 tile；越界位置填 `+inf`，对 min 运算无害。

**加载 dist[i, k]（第 k 列，本 tile 的行）**

```python
dist_ik = tl.load(
    dist + i_offs[:, None]*stride_dm + k_offs[None, :]*stride_dn,
    mask=(i_offs[:, None] < N),
    other=float('inf'))
```

- 指针网格 `(B,B)`：第 r 行的地址全是 `&dist[i_r][k]`，所以 `dist_ik[r, :]` 整行都是同一个值 `dist[i_r, k]`。相当于把列向量"复制"成方阵，以便下一步直接逐元素相加（同一元素被重复加载 B 次，靠缓存吸收）。
- mask 只需管行越界，因为 k 本身一定是合法顶点。

**加载 dist[k, j]（第 k 行）**

```python
dist_kj = tl.load(
    dist + k_offs[:, None]*stride_dm + j_offs[None, :]*stride_dn,
    mask=(j_offs[None, :] < N),
    other=float('inf'))
```

对称地，`dist_kj[:, c]` 整列都是 `dist[k, j_c]`，即行向量扩成方阵。

**松弛与写回**

```python
out = tl.minimum(dist_ij, dist_ik + dist_kj)
tl.store(dist_ptrs, mask=dist_mask, value=out)
```

- `dist_ik + dist_kj`：行常量 + 列常量广播，`[i,j]` 处恰好是 `dist[i,k] + dist[k,j]`。
- 逐元素取 min，即题目公式 `min(dist[i,j], dist[i,k]+dist[k,j])`。
- **原地写回 `dist`**（不是 output），带越界 mask。

## Host 部分

```python
def solve(dist, output, N):
    BLOCK_N = 64
    dist = dist.view(N, N)
    output = output.view(N, N)
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(N, BLOCK_N))
    for k in range(N):
        floyd_warshall[grid](
            dist, dist.stride(0), dist.stride(1),
            output, output.stride(0), output.stride(1),
            N, k, BLOCK_N,
        )
    output.copy_(dist)
```

- 把一维输入 reshape 成 N×N，拿到行/列步长。
- grid 是 `⌈N/64⌉²` 的 2D 网格。
- **串行循环 N 次**（N=2048 就是 2048 次 launch），每次全矩阵做一次关于顶点 k 的松弛；k 的顺序即 Floyd-Warshall 要求的顺序，launch 间靠同一 stream 的顺序执行保证先后。
- 最后把结果从 `dist` 拷进 `output`。

## 几个关键点

1. **为什么单次 launch 内原地读写没有竞争问题**：第 k 轮里第 k 行和第 k 列的值不变（`dist[k,k]=0` 且无负环 ⇒ `min(dist[i,k], dist[i,k]+dist[k,k]) = dist[i,k]`）。别的 program 读到的 k 行/k 列，无论读到"旧值还是新值"都一样，是良性竞争。
2. **`inf` 语义安全**：全程只有加法和 min，`inf + 有限值 = inf`，不可达路径自然保持 inf。
3. **冗余加载**：`dist_ik`、`dist_kj` 各自把同一行/列重复读了 B 次，属于是用带宽换写法简洁。

**性能点评**：这个版本正确但偏慢——每轮 k 都要把 16MB 矩阵从显存读一遍再写一遍，2048 轮累计约 100GB 显存流量，再加 2048 次 launch 开销，N=2048 时大致在 60~100ms 量级。相比之下，我上一份给你的分块（blocked）版本把每 32 个 k 合并在一轮里、tile 全程驻留寄存器做 min-plus，只需 192 次 launch，通常能快一个数量级左右。这份代码适合作为正确性基线，性能场景建议用分块版。