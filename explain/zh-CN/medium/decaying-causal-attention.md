把 `seq_len` 切成若干块（tile），每个 GPU program 负责一块 query 行；由于因果性（`n` 只能看 `m ≤ n`），每块只需累加自己**对角线左侧**的 K/V 块；衰减项 `γ^(n-m)` 用 `exp2((n-m)·log2γ)` 计算，避免溢出。

---

## 第一部分：`solve`（主机端包装函数）

```python
def solve(Q, K, V, output, seq_len, d_model, gamma):
```

按题目模板签名，张量已在 GPU 上，结果写入 `output`。

```python
BLOCK_D = max(16, triton.next_power_of_2(d_model))
```

- `triton.next_power_of_2(d_model)`：Triton 的 `tl.arange` 要求长度是 2 的幂，所以特征维向上取整（如 `d_model=100` → 128）。
- `max(16, ...)`：`tl.dot` 硬性要求每个维度 ≥ 16，`d_model=1` 时也补到 16。多出来的维度靠掩码读写，不影响结果。

```python
if BLOCK_D <= 64:
    BLOCK_M, BLOCK_N, num_warps = 64, 64, 4
elif BLOCK_D <= 128:
    BLOCK_M, BLOCK_N, num_warps = 32, 32, 4
else:  # BLOCK_D == 256
    BLOCK_M, BLOCK_N, num_warps = 16, 16, 4
```

按特征维选 tile 大小。`BLOCK_M`＝每个 program 处理多少 query 行，`BLOCK_N`＝内层循环每次吃多少 key 行。维度越大 tile 越小，目的是把寄存器/共享内存占用压在 T4 的限制内（T4 每块共享内存上限 64KB，这套配置实测编译后占 24–49KB）。

```python
log2_gamma = math.log2(min(max(float(gamma), 1e-300), 1.0))
```

**在主机端预先算好 `log2(γ)`**，内核里就不用再算 log（每个元素省一条超越函数）。钳位到 `(0, 1]` 是防御性的：题目保证 `0 < γ ≤ 1`，钳位后 `log2` 结果恒为有限值且 ≤ 0。注意这是 Python 双精度算的，比设备端 fp32 算 log 更准。

```python
scale = 1.0 / math.sqrt(d_model)
```

`1/√d_model` 缩放因子，同样在主机端算好。

```python
grid = (triton.cdiv(seq_len, BLOCK_M),)
```

一维网格：`ceil(seq_len / BLOCK_M)` 个 program。`seq_len=4096, BLOCK_M=64` → 64 个 program，每个负责 64 行 query。

```python
_decay_causal_attn_fwd[grid](
    Q, K, V, output,
    Q.stride(0), Q.stride(1), ...
    seq_len, d_model,
    log2_gamma, scale,
    BLOCK_M=..., BLOCK_N=..., BLOCK_D=...,
    num_warps=4, num_stages=1,
)
```

启动内核。传 `stride` 而不硬编码 `* d_model`，这样**非连续张量也正确**。`num_warps=4`＝每 program 128 线程；`num_stages=1`＝不做软件流水双缓冲——T4 共享内存紧张，且 fp32 FMA 路径本身偏计算 bound，稳妥优先。

---

## 第二部分：内核 `_decay_causal_attn_fwd`（设备端）

### 参数签名

```python
@triton.jit
def _decay_causal_attn_fwd(
    Q, K, V, Out, ... ,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
```

- 指针：`Q, K, V, Out` 是张量基地址。
- `tl.constexpr`：编译期常量。BLOCK_* 写进类型签名后，Triton 为每种组合编译一份专用机器码（循环展开、寄存器分配都按定长优化）。
- `seq_len, d_model, log2_gamma, scale` 是**运行时**标量——换 γ、换序列长不会触发重编译（这点和你之前那版把 `GAMMA` 声明成 constexpr 不同）。

### ① 确定本 program 负责哪些行

```python
pid_m = tl.program_id(axis=0)
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_d = tl.arange(0, BLOCK_D)
```

- `pid_m`：当前 program 编号（0 到 grid-1）。
- `offs_m`：它负责的 64 个 query 行号，即 `[pid*64, pid*64+1, ..., pid*64+63]`，对应公式里的 `n`。
- `offs_d`：特征维下标 `[0..BLOCK_D-1]`。

```python
m_mask = offs_m < seq_len
d_mask = offs_d < d_model
qd_mask = m_mask[:, None] & d_mask[None, :]
```

两个一维掩码广播成二维 `[BLOCK_M, BLOCK_D]`：处理 `seq_len` 不整除 64、`d_model` 不是 2 的幂时的越界位置。`[:, None]` 是加一维用于广播，和 NumPy 一样。

### ② 载入 Q tile

```python
q = tl.load(Q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
            mask=qd_mask, other=0.0)
```

指针算术：`行号 × 行stride + 列号 × 列stride`，得到 `[64, BLOCK_D]` 的地址网格，一次性加载。掩码外补 0——补 0 的行算出来也是 0，最后 store 时会掩掉，无害。

### ③ 累加器与因果上界

```python
acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
```

输出累加器 `[64, BLOCK_D]`，全程 fp32。

```python
hi = tl.minimum((pid_m + 1) * BLOCK_M, seq_len)
for start_n in range(0, hi, BLOCK_N):
```

**因果裁剪的关键**：本块最大行号是 `(pid_m+1)*BLOCK_M - 1`，它最多只能看到同样位置的 key，所以 m 循环到 `hi` 就停——对角线右侧的块整个跳过。相比"扫全部再掩掉"省约一半计算。`hi` 是运行时值，Triton 支持运行时边界的循环。

### ④ 载入 K 块、算注意力分数

```python
offs_n = start_n + tl.arange(0, BLOCK_N)
kv_mask = (offs_n < seq_len)[:, None] & d_mask[None, :]
k = tl.load(K + offs_n[:, None] * stride_km + ..., mask=kv_mask, other=0.0)
```

`offs_n` 是当前 64 个 key 行号，即公式里的 `m`。同样掩码加载。

```python
s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
```

- `tl.dot(q, k^T)`：`[64, BLOCK_D] × [BLOCK_D, 64]` → `[64, 64]`，一次算出本块 64 个 query 对 64 个 key 的**全部**点积——这就是公式里的 `Q[n]·K[m]`。
- `input_precision="ieee"`：强制全精度 fp32。**这是 T4 特化的关键**：T4（Turing）没有 TF32 tensor core，显式 ieee 让它老实走 FP32 FMA，精度与 fp32 参考实现一致，避免任何精度意外。
- 补 0 的特征维对点积贡献为 0，所以填充不影响结果。

### ⑤ 衰减掩码（数值稳定的核心）

```python
diff = offs_m[:, None] - offs_n[None, :]       # [64, 64]，即 n - m
exponent = tl.where(diff >= 0, diff * log2_gamma, float("-inf"))
s = s * tl.exp2(exponent)
```

- `diff`：每个 `(n, m)` 对的距离 `n - m`。
- `n ≥ m`：指数 `= (n-m)·log2γ`。因为 `log2γ ≤ 0`、`n-m ≥ 0`，**指数恒 ≤ 0**，`exp2` 结果在 `(0,1]` 内——绝不会溢出。这就是公式里的 `γ^(n-m)`（用 `exp2` 是因为 `γ^x = 2^(x·log2γ)`，且 GPU 上 ex2 是单条 SFU 指令，比 exp 快）。
- `n < m`（未来位置，只出现在对角线块内）：指数置 `-inf`，`exp2(-inf)` **精确等于 0**，不需要额外的布尔掩码乘法。
- 为什么不用 `γ^n · γ^(-m)` 分解？`seq_len=8192` 时 `γ^(-m)` 会 overflow 到 inf，再乘 `γ^n` 得 NaN。这个写法是全长度安全的。
- γ=1 时 `log2γ=0`，所有 `n≥m` 位置衰减为 1，自然退化为普通因果掩码。

### ⑥ 乘 V 并累加

```python
v = tl.load(V + ..., mask=kv_mask, other=0.0)
acc = tl.dot(s, v, input_precision="ieee", acc=acc)
```

- 第二次 `tl.dot`：`[64, 64] × [64, BLOCK_D]` → 加权求和 `Σ_m weights[n,m]·V[m]`，对应公式的最后一步。
- `acc=acc`：直接累加进累加器，等价于 `acc += tl.dot(s, v)` 但少一次中间拷贝。
- 越界的 key 行（`offs_n ≥ seq_len`）：K 补 0 → 分数为 0，V 也补 0 → 贡献为 0，无需额外处理。

### ⑦ 写回

```python
tl.store(Out + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
         acc, mask=qd_mask)
```

按同样的掩码写回，越界行（`seq_len` 尾部）和填充特征维不落盘。

---

## 数据流回顾

以 `seq_len=4096, d_model=64` 为例：64 个 program 并行，第 `pid` 个 program 拿着 64 行 Q，内层循环最多 `pid+1` 次（早的块循环少、晚的块循环多），每次循环做两次 64×64×64 的 `tl.dot`。总计算量约 `2·2·S²/2·D ≈ 4.3 GFLOP`，其中近一半被因果裁剪省掉，在 T4 的 8.1 TFLOPS FP32 上是毫秒级。

需要的话，我可以再补一张分块计算的示意图，或者讲讲如果要进一步提速（比如对角块单独处理省掉 `tl.where`、或者换 fp16 中间精度上 tensor core）可以怎么改。
