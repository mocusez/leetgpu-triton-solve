可以。我会按“整体目标 → 装饰器/配置 → kernel 参数 → program 映射 → patch 展开 → GEMM → cls/pos → store → Python launch”这个顺序逐行解释。

这段 Triton kernel 本质上是在 GPU 上融合完成 Vision Transformer 的 Patch Embedding：

$$
\text{patch tokens} = \text{flatten(patches)}W^T+b
$$

然后在最前面插入 `cls_token`，再加上位置编码：

$$
\text{output}=
[\text{CLS};\text{PatchEmbed}(x)] + \text{PosEmbed}
$$

它避免了显式 `unfold / im2col`，而是直接从 NCHW 图像中按照 patch 的位置计算地址并加载数据。

---

## 1. import

```python
import torch
import triton
import triton.language as tl
```

分别是：

* `torch`：PyTorch tensor 和 CUDA tensor。
* `triton`：Triton runtime、autotune、kernel launch 等。
* `triton.language as tl`：Triton kernel 里的 SIMD / tensor 操作语言。

例如：

```python
tl.arange
tl.load
tl.store
tl.dot
```

都来自 `triton.language`。

---

# 2. dot 的精度

```python
_DOT_PRECISION = "tf32x3"
```

这里指定：

```python
tl.dot(..., input_precision="tf32x3")
```

的计算模式。

你的 accumulator 是：

```python
float32
```

但是如果输入也是 FP32，在 NVIDIA Ampere（A100）上，矩阵乘可以利用 Tensor Core。

`tf32x3` 可以理解成用多个 TF32 运算组合，提高 FP32 矩阵乘的数值精度。

所以它是在：

> 吞吐量和 FP32 精度之间做折中。

A100 是 Ampere 架构，支持 TF32 Tensor Core，因此这是这个 kernel 很适合 A100 的原因之一。

---

# 3. autotune

```python
@triton.autotune(
    configs = [
```

告诉 Triton：

> 同一个 kernel 准备几个不同的 tile 配置，第一次遇到某种输入尺寸时 benchmark，然后选最快的配置。

第一组：

```python
triton.Config(
    {"BM": 32, "BD": 64, "BK": 32},
    num_warps = 4,
    num_stages = 3,
),
```

表示一个 Triton program 计算输出矩阵的：

```text
32 × 64
```

区域。

也就是：

```text
BM = 32
BD = 64
```

同时 K 维一次处理：

```text
BK = 32
```

可以类比普通 GEMM：

```text
A: [M, K]
B: [K, D]
C: [M, D]
```

一个 program 负责：

```text
C[m:m+BM, d:d+BD]
```

---

下一组：

```python
{"BM": 64, "BD": 64, "BK": 32}
```

表示：

```text
64 × 64 output tile
K tile = 32
```

---

```python
{"BM": 64, "BD": 128, "BK": 32}
```

输出 tile 更宽：

```text
64 × 128
```

---

```python
{"BM": 64, "BD": 128, "BK": 64}
```

同时 K tile 也变为：

```text
64
```

这可能提高 Tensor Core 利用率，但也会：

* 增加寄存器压力
* 增加 shared memory / pipeline 压力
* 降低 occupancy

因此不一定永远更快，交给 autotune。

---

```python
num_warps = 8,
```

一个 Triton program block 使用 8 个 warp。

NVIDIA 一个 warp 是 32 threads，因此大致对应：

```text
8 × 32 = 256 threads
```

不过 Triton 的抽象比 CUDA thread block 更高级，不应简单理解成你手写的 256-thread CUDA block。

---

```python
num_stages = 3,
```

控制 software pipelining。

大致可以理解成：

> 在当前矩阵块计算的同时，提前准备后面的数据。

对于 A100 这种 GPU，增加 pipeline stage 经常能更好隐藏 global memory latency。

---

## autotune key

```python
key = ["B", "C", "H", "W", "P", "D", "PRECISION"]
```

表示这些值变化时重新 autotune。

例如：

```text
B=32,C=3,H=224,W=224,P=16,D=768
```

和：

```text
B=64,C=3,H=224,W=224,P=16,D=768
```

会被认为是不同 workload。

因此 Triton 会分别寻找最佳：

```text
BM, BD, BK
```

配置。

---

# 4. Triton kernel

```python
@triton.jit
def _patch_embed_kernel(
```

`@triton.jit` 表示这是一个 Triton GPU kernel。

运行时 Triton 会编译为 GPU code。

---

参数：

```python
Images, Weight, Bias, Cls, Pos, Out,
```

全部是 pointer。

对应：

```text
Images → 输入图片
Weight → patch projection 权重
Bias   → patch projection bias
Cls    → cls token
Pos    → position embedding
Out    → 最终输出
```

---

# 5. compile-time constants

```python
B: tl.constexpr,
C: tl.constexpr,
H: tl.constexpr,
W: tl.constexpr,
P: tl.constexpr,
D: tl.constexpr,
```

这些在 Triton 编译期作为常量。

例如：

```text
B = batch
C = input channels
H = image height
W = image width
P = patch size
D = embedding dimension
```

为什么使用 `tl.constexpr` 很重要？

因为 Triton 编译器可以把：

```python
W // P
C * P * P
```

这些计算直接常量折叠。

---

```python
PRECISION: tl.constexpr,
```

也是编译期确定，例如：

```text
"tf32x3"
```

---

```python
BM: tl.constexpr,
BD: tl.constexpr,
BK: tl.constexpr,
```

分别是 GEMM tile：

```text
BM → token 维度 tile
BD → embedding 维度 tile
BK → reduction 维度 tile
```

---

```python
GROUP_M: tl.constexpr
```

用于 program ID 的 grouped ordering。

这是 Triton GEMM 很常见的优化。

它的主要目的是：

> 改善 cache locality。

---

# 6. 计算 patch 网格

```python
GW: tl.constexpr = W // P
```

`GW`：

```text
Grid Width
```

即每一行有多少 patch。

例如：

```text
W = 224
P = 16
```

那么：

```text
GW = 14
```

---

```python
N: tl.constexpr = (H // P) * GW
```

patch 总数。

如果：

```text
H = W = 224
P = 16
```

则：

```text
N = 14 × 14 = 196
```

---

```python
T: tl.constexpr = N + 1
```

每张图片最终 token 数：

```text
1 CLS token
+
N patch tokens
```

因此：

```text
T = 197
```

---

```python
M: tl.constexpr = B * T
```

把：

```text
[B, T, D]
```

逻辑上 flatten 成：

```text
[M, D]
```

其中：

```text
M = B*T
```

因此整个 kernel 实际上把任务看成一个矩阵：

```text
Out[M, D]
```

---

```python
K: tl.constexpr = C * P * P
```

一个 patch 展平后的维度。

例如 RGB 16×16 patch：

```text
K = 3 × 16 × 16
  = 768
```

于是 patch embedding 实质上就是：

```text
[M_patch, 768] @ [768, D]
```

---

# 7. 当前 program id

```python
pid = tl.program_id(0)
```

launch grid 是一维的。

所以这里获得：

```text
0, 1, 2, ...
```

的 program ID。

每一个 `pid` 最终映射到一个：

```text
BM × BD
```

输出 tile。

---

# 8. output tile 数

```python
num_m = tl.cdiv(M, BM)
```

M 方向一共有多少 tile。

`tl.cdiv` 是 ceiling division：

$$
\lceil M/BM \rceil
$$

---

```python
num_d = tl.cdiv(D, BD)
```

D 方向 tile 数：

$$
\lceil D/BD \rceil
$$

所以总 program 数：

```text
num_m × num_d
```

这和 Python 下面的 grid 完全对应。

---

# 9. grouped ordering

```python
group_size = GROUP_M * num_d
```

一个 group 最多处理：

```text
GROUP_M
```

个 M tile，并覆盖所有 D tile。

如果：

```text
GROUP_M = 8
```

那么大致形成：

```text
8 个 M tiles × num_d 个 D tiles
```

---

```python
group_id = pid // group_size
```

判断当前 program 属于哪个 group。

---

```python
first_m = group_id * GROUP_M
```

该 group 从哪个 M tile 开始。

---

```python
actual_m = tl.minimum(num_m - first_m, GROUP_M)
```

最后一个 group 可能不到 8 个 M tile。

所以：

```text
actual_m = min(剩余 M tiles, GROUP_M)
```

---

```python
in_group = pid % group_size
```

得到当前 program 在 group 内的位置。

---

```python
pid_m = first_m + in_group % actual_m
```

确定当前 program 对应哪个 M tile。

---

```python
pid_d = in_group // actual_m
```

确定对应哪个 D tile。

这种排序不是简单：

```text
m0,d0
m0,d1
m0,d2
...
```

而会让附近 program 更倾向复用数据。

这是经典 Triton GEMM grouped ordering 思路。

---

# 10. 当前 tile 的 M / D index

```python
m = pid_m * BM + tl.arange(0, BM)
```

得到 BM 个 token index。

比如：

```text
BM = 32
pid_m = 2
```

那么：

```text
m = [64, 65, ..., 95]
```

shape：

```text
[BM]
```

---

```python
d = pid_d * BD + tl.arange(0, BD)
```

得到当前 embedding channel。

比如：

```text
BD = 64
pid_d = 3
```

则：

```text
d = [192,...,255]
```

shape：

```text
[BD]
```

所以这个 program 计算：

```text
Out[m, d]
```

即：

```text
[BM, BD]
```

的小矩阵。

---

# 11. 把 m 恢复成 batch/token

```python
batch = m // T
```

因为之前：

```text
m = batch * T + token
```

所以：

```text
batch = m // T
```

---

```python
row = m % T
```

获得图片内部 token index。

例如：

```text
row = 0   → CLS
row = 1   → patch 0
row = 2   → patch 1
...
row = N   → patch N-1
```

---

# 12. token → patch index

```python
patch = tl.maximum(row - 1, 0)
```

因为：

```text
row 0 = CLS
row 1 = patch 0
row 2 = patch 1
```

所以理论上：

```python
patch = row - 1
```

但 CLS 会产生：

```text
-1
```

因此 clamp 到：

```text
0
```

注意：

CLS 最后其实不会真正使用 image 数据，因为下面 `is_patch=False`。

所以这里让它指向 patch 0 只是为了保证地址计算合法、简单。

---

```python
is_patch = (m < M) & (row != 0)
```

判断：

1. M 没越界
2. 不是 CLS

所以只有真正 patch token 才读取 image。

---

# 13. 计算 patch 左上角地址

```python
image_base= (
    batch * (C * H * W)
    + (patch // GW) * (P * W)
    + (patch % GW) * P
)
```

这是整个 kernel 最值得理解的地方之一。

假设：

```text
Images shape = [B,C,H,W]
```

按 contiguous NCHW 存储。

---

先看：

```python
batch * (C * H * W)
```

跳到第 `batch` 张图片。

因为每张图片有：

```text
C*H*W
```

个元素。

---

```python
patch // GW
```

得到 patch row。

例如 14×14 patches：

```text
patch = 17
GW = 14

patch_row = 17 // 14 = 1
```

---

```python
patch % GW
```

得到 patch column：

```text
patch_col = 17 % 14 = 3
```

---

图片中的左上角坐标：

```text
y = patch_row * P
x = patch_col * P
```

NCHW 中 spatial offset：

```text
y*W+x
```

所以：

```python
(patch // GW) * (P * W)
+
(patch % GW) * P
```

恰好是：

```text
patch_y * W + patch_x
```

因此：

```python
image_base
```

就是这个 patch 在每张图片中的左上角。

不过现在还没包含 channel offset。

channel offset 会在 `image_delta` 中加入。

---

# 14. K tile

```python
rk = tl.arange(0, BK)
```

例如：

```text
BK=32
```

得到：

```text
[0,1,...,31]
```

---

```python
acc = tl.zeros((BM, BD), dtype = tl.float32)
```

建立 GEMM accumulator：

```text
[BM, BD]
```

并使用 FP32 accumulation。

这对数值稳定性很重要。

---

# 15. K reduction loop

```python
for block_k in range(0, tl.cdiv(K, BK)):
```

把：

```text
K = C*P*P
```

分成若干 BK tile。

例如：

```text
K = 768
BK = 32
```

循环：

```text
768 / 32 = 24
```

次。

---

```python
k = block_k * BK + rk
```

当前 reduction indices。

例如第二块：

```text
k = 32..63
```

---

# 16. 将 flatten 后的 k 映射回 c,y,x

```python
image_delta = (
    (k //(P * P)) * (H * W)
    + ((k // P) % P) * W
    + k % P
)
```

这里是在实现：

```text
flattened patch index k
↓
channel, patch_y, patch_x
```

一个 patch flatten 顺序是：

```text
[C,P,P]
```

所以：

```python
channel = k // (P*P)
```

---

patch 内部行：

```python
(k // P) % P
```

相当于：

```text
patch_y = (k % (P*P)) // P
```

---

patch 内部列：

```python
k % P
```

于是 NCHW offset：

```text
channel * H*W
+
patch_y * W
+
patch_x
```

正好就是：

```python
image_delta
```

---

所以：

```python
image_base + image_delta
```

最终得到：

```text
Images[b, c, patch_global_y, patch_global_x]
```

的地址。

这就是为什么不需要：

```python
torch.nn.functional.unfold
```

。

kernel 一边 GEMM，一边动态计算 im2col 地址。

---

# 17. load A

```python
a = tl.load(
    Images + image_base[:, None] + image_delta[None, :],
```

这里 broadcasting 非常关键。

```python
image_base[:, None]
```

shape：

```text
[BM,1]
```

而：

```python
image_delta[None,:]
```

shape：

```text
[1,BK]
```

相加得到：

```text
[BM,BK]
```

所以 `a` 是：

```text
BM 个 token
×
BK 个 patch feature
```

即 GEMM 左矩阵：

```text
A_tile[BM,BK]
```

---

mask：

```python
mask = is_patch[:, None] & (k[None, :] < K)
```

两层保护。

第一层：

```python
is_patch
```

CLS 不读图片。

第二层：

```python
k < K
```

最后一个 K tile 防止越界。

---

```python
other = 0.0
```

越界值当 0。

因此不会影响矩阵乘结果。

---

# 18. load Weight

```python
w = tl.load(
    Weight + d[None, :] * K + k[:, None],
```

假设：

```text
Weight.shape = [D,K]
```

也就是常见 Conv2d weight：

```text
[D,C,P,P]
```

在内存里 flatten 成：

```text
[D,K]
```

对于：

```text
Weight[d,k]
```

线性地址：

```text
d*K+k
```

这里构造 shape：

```text
[BK,BD]
```

因为：

```python
d[None,:] * K
```

是：

```text
[1,BD]
```

而：

```python
k[:,None]
```

是：

```text
[BK,1]
```

broadcast 后：

```text
[BK,BD]
```

所以：

```text
w.shape = [BK,BD]
```

---

mask：

```python
(k[:, None] < K) & (d[None, :] < D)
```

同时防止：

```text
K 尾部越界
D 尾部越界
```

---

# 19. GEMM

```python
acc = tl.dot(a, w, acc, input_precision = PRECISION)
```

这里：

```text
a   : [BM,BK]
w   : [BK,BD]
acc : [BM,BD]
```

因此：

$$
acc = A W + acc
$$

循环结束后：

```text
acc ≈ patch_flat @ Weight.T
```

因为 `Weight` 内存逻辑是：

```text
[D,K]
```

但这里读取成：

```text
w[k,d] = Weight[d,k]
```

所以数学上相当于：

```python
patch @ Weight.T
```

这和 PyTorch：

```python
F.linear(patch, weight, bias)
```

的行为一致。

---

# 20. bias

```python
bias = tl.load(Bias + d, mask = d < D, other = 0.0)
```

读取：

```text
Bias[d]
```

shape：

```text
[BD]
```

---

# 21. CLS token

```python
cls = tl.load(Cls + d, mask = d < D, other = 0.0)
```

读取：

```text
Cls[d]
```

所以这里假定 cls token 的有效内存布局能看作：

```text
[D]
```

例如这些 shape 都可能在 contiguous 情况下工作：

```text
[D]
[1,D]
[1,1,D]
```

因为最终都是连续 D 个元素。

---

# 22. output valid mask

```python
valid = (m[:, None] < M) & (d[None, :] < D)
```

输出 tensor 是：

```text
[M,D]
```

所以同时防：

```text
M 越界
D 越界
```

shape：

```text
[BM,BD]
```

---

# 23. 加载 position embedding

```python
pos = tl.load(
    Pos + row[:, None].to(tl.int64) * D + d[None, :],
```

Position embedding 被视为：

```text
[T,D]
```

因此：

```text
Pos[row,d]
```

地址是：

```text
row*D+d
```

这里 `row` 不包含 batch。

这是正确的，因为所有 batch 共享同一组 position embedding：

```text
batch 0: Pos[0:T]
batch 1: Pos[0:T]
batch 2: Pos[0:T]
...
```

---

```python
.to(tl.int64)
```

把索引转为 64 位。

主要用于 pointer arithmetic，避免较大的 tensor offset 下整数范围问题。

---

# 24. CLS 与 patch 二选一

```python
result = tl.where(
    row[:, None] == 0,
    cls[None, :],
    acc + bias[None, :],
)
```

如果：

```text
row == 0
```

表示 CLS。

所以：

```text
result = cls
```

否则：

```text
result = patch_projection + bias
```

即：

$$
xW^T+b
$$

因此每张图片最终是：

```text
row 0 → cls_token
row 1 → patch0 @ W.T + b
row 2 → patch1 @ W.T + b
...
```

---

# 25. position embedding

```python
result = result + pos
```

所以：

CLS：

$$
CLS + Pos_0
$$

Patch：

$$
PatchProjection_i + Pos_i
$$

最终就是标准 ViT embedding：

```python
x = torch.cat([cls_token, patch_tokens], dim=1)
x = x + pos_embed
```

---

# 26. store

```python
tl.store(
    Out + m[:, None].to(tl.int64) * D + d[None, :],
```

`Out` 被看成：

```text
[M,D]
```

地址：

```text
m*D+d
```

---

```python
result,
mask = valid,
```

只写合法部分。

所以逻辑输出实际上是：

```text
[B,T,D]
```

只是 kernel 内将前两维 flatten 成：

```text
[B*T,D]
```

。

---

# 27. Python wrapper

```python
def solve(
    images: torch.Tensor,
    patch_weight: torch.Tensor,
    patch_bias: torch.Tensor,
    cls_token: torch.Tensor,
    pos_embed: torch.Tensor,
    output: torch.Tensor,
```

这个函数负责 launch Triton kernel。

典型 tensor shape 应该是：

```text
images       [B,C,H,W]
patch_weight [D,C,P,P]
patch_bias   [D]
cls_token    [1,1,D] / [D]
pos_embed    [1,N+1,D] / [N+1,D]
output       [B,N+1,D]
```

只要实际 storage contiguous 且布局和上述地址计算一致即可。

---

# 28. patch 数

```python
N = (H // P) * (W // P)
```

例如：

```text
224×224
patch=16
```

得到：

```text
14×14=196
```

patch。

---

# 29. Triton launch grid

```python
grid = lambda meta: (
```

这里 `meta` 是 Triton autotune 选出的 configuration。

例如其中可能有：

```python
meta["BM"] = 64
meta["BD"] = 128
```

---

```python
triton.cdiv(B * (N + 1), meta["BM"])
```

M 维 program 数：

$$
\left\lceil
\frac{B(N+1)}{BM}
\right\rceil
$$

---

```python
* triton.cdiv(D, meta["BD"]),
```

再乘 D 方向 tile 数。

所以总 programs：

$$
\left\lceil\frac{M}{BM}\right\rceil
\times
\left\lceil\frac{D}{BD}\right\rceil
$$

。

注意这里 launch 的是一维 grid：

```python
(grid_size,)
```

而 kernel 里面再把 `pid` 解码成：

```text
pid_m
pid_d
```

---

# 30. launch kernel

```python
_patch_embed_kernel[grid](
```

Triton 特有 kernel launch 语法。

类似 CUDA：

```cpp
kernel<<<grid, block>>>(...)
```

但：

* block/thread mapping 由 Triton compiler 处理。
* `num_warps`
* `num_stages`

等由 config 控制。

---

```python
images,
patch_weight,
patch_bias,
cls_token,
pos_embed,
output,
```

传入各 GPU pointer。

---

```python
B,
C,
H,
W,
P,
D,
```

这些对应 kernel 中的：

```python
tl.constexpr
```

因此会被 specialization。

---

```python
PRECISION = _DOT_PRECISION,
```

即：

```text
tf32x3
```

---

```python
GROUP_M = 8
```

固定 M grouping 为 8。

用于前面：

```python
group_size = GROUP_M * num_d
```

的 program ordering。

---

# 把整个 kernel 翻译成 PyTorch

这段 Triton 大体等价于下面的逻辑：

```python
import torch
import torch.nn.functional as F

def reference(
    images,
    patch_weight,
    patch_bias,
    cls_token,
    pos_embed,
):
    B, C, H, W = images.shape
    D, _, P, _ = patch_weight.shape

    # [B, D, H/P, W/P]
    x = F.conv2d(
        images,
        patch_weight,
        patch_bias,
        stride=P,
    )

    # [B, D, GH, GW]
    # ->
    # [B, N, D]
    x = x.flatten(2).transpose(1, 2)

    cls = cls_token.expand(B, -1, -1)

    # [B, N+1, D]
    x = torch.cat([cls, x], dim=1)

    x = x + pos_embed

    return x
```

区别是你的 Triton kernel 把几件事情融合成了**一次 kernel**：

```text
读取 image patch
      ↓
隐式 flatten
      ↓
patch × Weight
      ↓
加 bias
      ↓
插入 CLS
      ↓
加 position embedding
      ↓
写 output
```

而不是 PyTorch 常见的：

```text
conv/unfold kernel
↓
reshape kernel
↓
concat kernel
↓
add kernel
```

---

# 用矩阵角度理解它最简单

整个 Triton kernel 可以看成计算：

```text
                   D
             ┌──────────────┐
             │   output     │
             │              │
M = B*(N+1)  │              │
             │              │
             └──────────────┘
```

一个 program 负责：

```text
            BD
        ┌────────┐
        │        │
     BM │ tile   │
        │        │
        └────────┘
```

对于 patch row：

```text
A [BM × K]
       ×
W [K × BD]
       =
C [BM × BD]
```

其中：

```text
K = C × P × P
```

最大的技巧就是：

> `A` 并不存在于内存中。

通常 `A` 会是 `im2col/unfold` 后的 patch matrix：

```text
[B*N, C*P*P]
```

但这里不创建它。

它通过：

```python
image_base + image_delta
```

直接从原始：

```text
[B,C,H,W]
```

里读取对应元素。

所以这属于一种：

**implicit im2col + GEMM + epilogue fusion**。

---

## 以 ViT-B/16 为例

如果输入：

```text
B = 32
C = 3
H = W = 224
P = 16
D = 768
```

那么：

```text
GW = 224 / 16 = 14
N  = 14 × 14 = 196
T  = 197
M  = 32 × 197 = 6304
K  = 3 × 16 × 16 = 768
```

kernel 最终要得到：

```text
Out = [6304,768]
```

逻辑上 reshape 为：

```text
[32,197,768]
```

其中每张图片：

```text
token 0:
    CLS + Pos[0]

token 1:
    Patch0 @ W.T + bias + Pos[1]

token 2:
    Patch1 @ W.T + bias + Pos[2]

...

token 196:
    Patch195 @ W.T + bias + Pos[196]
```

所以从模型意义来说，这就是标准 ViT 的 embedding 前端。

---

## 这份代码里几个非常值得注意的细节

第一，代码隐含要求：

```text
H % P == 0
W % P == 0
```

否则：

```python
H // P
W // P
```

会直接忽略右侧或底部不能组成完整 patch 的区域。

第二，它假设 tensor 的实际存储布局与下面一致：

```text
images       → contiguous NCHW
weight       → contiguous [D,C,P,P]
bias         → contiguous [D]
cls          → 最后 D 个元素连续
pos          → [T,D] 连续
output       → [B,T,D] 连续
```

这里**完全没有读取 PyTorch stride**。

因此像：

```python
images = images.transpose(...)
```

这样的 non-contiguous tensor 直接传进去可能算错。

第三：

```python
Weight + d * K + k
```

说明权重按照：

```text
[D,K]
```

读取。

也就是原始：

```text
[D,C,P,P]
```

连续布局。

第四，CLS 行依然会执行整个 K-loop：

```python
for block_k ...
    a = 0
    w = load(...)
    acc = tl.dot(...)
```

虽然：

```python
is_patch=False
```

会令 `a=0`，但对应 program 仍然进行了 dot 工作。

所以从优化角度说，CLS row 存在一些无效计算。

不过 CLS 只有：

```text
B 行
```

而 patch 有：

```text
B*N 行
```

当 N=196 时占比很小，因此通常未必值得单独拆一个 kernel。

第五，`tf32x3` 是针对 FP32 dot 的重要性能/精度选择；如果你的 `images` 和 `patch_weight` 是 `fp16/bf16`，Tensor Core 的执行路径和性能考虑会有所不同。

---

最后可以把整个地址计算浓缩成一句非常重要的话：

```python
m
↓
(batch, token)
↓
patch index
↓
patch top-left position

k
↓
(channel, patch_y, patch_x)

image_base + image_delta
↓
Images[batch, channel,
       patch_global_y,
       patch_global_x]
```

然后：

```python
[BM, BK] @ [BK, BD]
```

得到一块：

```python
[BM, BD]
```

的 ViT patch embedding。

这也是理解这份 Triton 代码的核心。
