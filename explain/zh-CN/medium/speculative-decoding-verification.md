### 1. 核心定义与序列并行映射

```python
import triton
import triton.language as tl

@triton.jit
def _speculative_decoding_verification_kernel(
    draft_tokens_ptr, draft_probs_ptr, target_probs_ptr, uniform_samples_ptr, output_tokens_ptr,
    B, T, V, BLOCK_SIZE_V: tl.constexpr
):

```

* **`@triton.jit`**: 装饰器，告诉 Triton 编译器这是一个需要被即时编译（JIT）到 GPU 上执行的内核函数（Kernel）。
* **指针参数 (`_ptr`)**: 传入的是显存地址的起始点。在 Triton 中，我们通过指针运算来读取数据。
* **`BLOCK_SIZE_V: tl.constexpr`**: 声明这是一个在编译期就确定的常量（我们后面设为 1024）。Triton 会利用这个常量进行循环展开和内存对齐优化。

```python
    pid = tl.program_id(0)
    if pid >= B:
        return
    b = pid

```

* **`pid = tl.program_id(0)`**: 获取当前运行的 Block ID。在一维网格配置下，每个 Block 处理一个序列（Sequence）。
* **`if pid >= B: return`**: 边界保护，防止因为线程块向上取整而导致越界访问。
* **`b = pid`**: 将其赋值给 `b`，代表当前的 batch/sequence 索引。

### 2. 初始化与主循环控制

```python
    for idx in range(T + 1):
        tl.store(output_tokens_ptr + b * (T + 1) + idx, 0)
    accepted_all = True

```

* **`for idx ... tl.store(...)`**: 根据题目要求，如果一个 Token 被拒绝，其后续的 Token 以及 Bonus Token 都不会产生。所以我们一开始干脆把当前序列的完整输出（长度为 $T+1$）全初始化为 $0$。
* **`accepted_all = True`**: 状态标识。因为 Triton 不支持在动态循环中使用 `break`，我们用这个变量来控制是否继续验证后续的 Token。

```python
    for i in range(T):
        if accepted_all:

```

* **`for i in range(T):`**: 从左到右遍历推测的 $T$ 个 Token。
* **`if accepted_all:`**: 只有在之前没有发生过拒绝（Reject）时，才执行验证逻辑。如果变成了 `False`，这个 `for` 循环其实还在跑（空转），但由于条件不满足，不会发生任何显存读写，极大地降低了性能损耗。

### 3. 接受概率计算 (Acceptance Probability)

```python
            t_i = tl.load(draft_tokens_ptr + b * T + i)
            p_i_ti = tl.load(draft_probs_ptr + b * T * V + i * V + t_i)
            q_i_ti = tl.load(target_probs_ptr + b * T * V + i * V + t_i)

```

* **`t_i`**: 读取当前位置 $i$ 上的推测 Token ID。
* **`p_i_ti` / `q_i_ti**`: 计算 1D 内存偏移量 `b * (T * V) + i * V + t_i`，读取草稿模型（$p$）和目标模型（$q$）对该 Token 的预测概率。

```python
            alpha_i = q_i_ti / p_i_ti
            if alpha_i > 1.0:
                alpha_i = 1.0
            u_i = tl.load(uniform_samples_ptr + b * (T + 1) + i)

```

* **`alpha_i = q_i_ti / p_i_ti`**: 计算接受比率 $\alpha$。题目约束了 $p > 0$，所以不用担心除零错误。
* **`if alpha_i > 1.0: alpha_i = 1.0`**: 实现数学公式中的 $\min(1, \frac{q}{p})$。
* **`u_i`**: 取出预先生成好的均匀分布随机数，用于进行掷骰子。

### 4. 接受或拒绝的分支

```python
            if u_i < alpha_i:
                tl.store(output_tokens_ptr + b * (T + 1) + i, t_i)
            else:
                accepted_all = False

```

* **`if u_i < alpha_i:`**: 如果随机数小于 $\alpha$，代表**接受（Accept）**。直接将 `t_i` 写入输出数组的第 $i$ 个位置。
* **`else:`**: 否则**拒绝（Reject）**。将 `accepted_all` 置为 `False`。这会触发当前的重采样逻辑，并让后续所有的 $i$ 迭代变成空转。

### 5. 重采样 Pass 1：计算分布总和

```python
                sum_adj = 0.0
                for v_offset in range(0, V, BLOCK_SIZE_V):
                    v_idx = v_offset + tl.arange(0, BLOCK_SIZE_V)
                    v_mask = v_idx < V
                    # ... [指针计算] ...
                    p_val = tl.load(p_ptr, mask=v_mask, other=0.0)
                    q_val = tl.load(q_ptr, mask=v_mask, other=0.0)
                    
                    adj_val = tl.where(q_val > p_val, q_val - p_val, 0.0)
                    adj_val = tl.where(v_mask, adj_val, 0.0)
                    sum_adj += tl.sum(adj_val, axis=0)

```

* *背景：由于词表 $V$ 可能很大（如 131,072），单个线程无法一次性读完。我们使用 `BLOCK_SIZE_V`（如 1024）对词表进行分块读取。*
* **`v_idx = ...` & `v_mask = v_idx < V**`: 生成 1024 长度的索引向量，并创建越界保护掩码（Mask），防止在最后一块超出 $V$ 时读取到非法显存。
* **`adj_val = tl.where(...)`**: 并行计算 1024 个词的修正概率 $\text{adj}(v) = \max(0, q(v) - p(v))$。
* **`sum_adj += tl.sum(adj_val, axis=0)`**: 使用 Triton 内置的高效归约函数，将这 1024 个浮点数求和，并累加到全局总和中。

```python
                r = tl.load(uniform_samples_ptr + b * (T + 1) + T)
                is_uniform = sum_adj <= 0.0
                target_r = tl.where(is_uniform, r * V, r * sum_adj)

```

* **`r = ...`**: 加载预留给位置 $T$ 的随机数，用作逆 CDF 采样的基准。
* **`is_uniform = sum_adj <= 0.0`**: 极少数情况下，调整后的分布全为 0（即目标模型完全认同草稿模型，甚至更低），需要回退到均匀分布。
* **`target_r = r * sum_adj`**: 巧妙的数学转换！我们不去除以 `sum_adj` 把数组归一化（避免除法带来的性能下降和精度丢失），而是直接把随机数 $r$ 乘上总和，效果等价。

### 6. 重采样 Pass 2：无分支的 Inverse CDF (逆变换采样)

```python
                running_sum = 0.0
                chosen_k = V 
                
                for v_offset in range(0, V, BLOCK_SIZE_V):
                    v_idx = ... # 同上
                    v_mask = ... # 同上
                    
                    if is_uniform:
                        adj_val = tl.where(v_mask, 1.0, 0.0)
                    else:
                        # 重新计算 adj_val (省略重复代码)

```

* **`chosen_k = V`**: 极其重要的“哨兵值（Sentinel Value）”。由于 Triton 不能用 `break` 提前退出循环，我们要用 $V$ （一个永远不可能在 $0 \dots V-1$ 词表里的索引）来代表“**目前还没找到满足条件的 Token**”。
* **`if is_uniform:`**: 如果是均匀分布，每个合法词的权重就是 1.0；否则重算一遍 `adj_val`（相比于存入显存再读出，重新计算在 GPU 上反而更快，这叫算术强度优化）。

```python
                    chunk_cumsum = tl.cumsum(adj_val, axis=0)
                    total_cumsum = running_sum + chunk_cumsum
                    
                    cond = (total_cumsum >= target_r) & v_mask
                    v_idx_selected = tl.where(cond, v_idx, V)
                    min_idx = tl.min(v_idx_selected, axis=0)
                    
                    chosen_k = tl.where((chosen_k == V) & (min_idx < V), min_idx, chosen_k)
                        
                    running_sum += tl.sum(adj_val, axis=0)

```

* **`chunk_cumsum = tl.cumsum(...)`**: 计算当前 1024 长度块内的前缀和。
* **`total_cumsum = running_sum + chunk_cumsum`**: 加上之前块的累加和，恢复全局的 CDF 累积量。
* **`cond = (total_cumsum >= target_r)`**: 寻找累加概率第一次跨过目标值 `target_r` 的位置。
* **`v_idx_selected = tl.where(cond, v_idx, V)`**: 满足条件的保留原本的词索引，不满足条件的设为 $V$。
* **`min_idx = tl.min(...)`**: 找出当前块内第一个满足条件的最小索引。如果这个块里全都不满足，`min_idx` 就会是 $V$。
* **`chosen_k = tl.where((chosen_k == V) & (min_idx < V), min_idx, chosen_k)`**: 无分支状态更新。翻译过来就是：**只有当我之前一直没找到（`chosen_k == V`），并且当前块找到了目标（`min_idx < V`）时，我才把 `chosen_k` 更新为你。否则，保持原样。** 这样完美模拟了寻找“第一个”匹配项的过程。

```python
                chosen_k = tl.where(chosen_k == V, V - 1, chosen_k)
                chosen_k = tl.maximum(0, tl.minimum(chosen_k, V - 1))
                
                tl.store(output_tokens_ptr + b * (T + 1) + i, chosen_k)

```

* **`chosen_k == V` 处理**: 浮点数累加偶尔有精度截断，可能导致到最后都没跨过 `target_r`，此时安全降级，把最后一个词 $V-1$ 给它。
* **`tl.maximum(0, tl.minimum(chosen_k, V - 1))`**: 就是整数版的 `clamp`（裁剪函数），确保生成的 Token ID 绝对合法，防止非法访存崩溃。
* **`tl.store(...)`**: 把重采样出来的 Token 写入输出数组。由于随后 `accepted_all` 是 `False`，后续的 $i$ 位置将自动跳过，保留最初始化的 $0$。

### 7. 奖励 Token (Bonus Token) 提取

```python
    if accepted_all:
        r = tl.load(uniform_samples_ptr + b * (T + 1) + T)
        running_sum = 0.0
        bonus_k = V  
        
        for v_offset in range(0, V, BLOCK_SIZE_V):
            # ... [省略与 Pass 2 完全相同的读取与掩码逻辑] ...
            q_val = tl.where(v_mask, q_val, 0.0)
            
            chunk_cumsum = tl.cumsum(q_val, axis=0)
            total_cumsum = running_sum + chunk_cumsum
            
            cond = (total_cumsum >= r) & v_mask
            v_idx_selected = tl.where(cond, v_idx, V)
            min_idx = tl.min(v_idx_selected, axis=0)
            
            bonus_k = tl.where((bonus_k == V) & (min_idx < V), min_idx, bonus_k)
            running_sum += tl.sum(q_val, axis=0)
            
        bonus_k = tl.where(bonus_k == V, V - 1, bonus_k)
        bonus_k = tl.maximum(0, tl.minimum(bonus_k, V - 1))
        
        tl.store(output_tokens_ptr + b * (T + 1) + T, bonus_k)

```

* **`if accepted_all:`**: 只有当 $T$ 个草稿 Token 完美通关时，才进入这里。
* **为什么只有一趟循环？**: 题目约束里说了，$q$ 是一个合法的概率分布，其累加和天生就是 $1.0$。所以我们**根本不需要像拒绝时那样先算一遍 sum**，直接用给定的随机数 $r \in [0, 1)$ 进行逆 CDF 累加匹配即可，逻辑与 Pass 2 极其相似，直接拿下了最后的 Bonus Token。

### 8. Python 调度入口

```python
def solve(draft_tokens, draft_probs, target_probs, uniform_samples, output_tokens, B, T, V):
    grid = (B,)
    _speculative_decoding_verification_kernel[grid](
        draft_tokens, draft_probs, target_probs, uniform_samples, output_tokens,
        B, T, V, BLOCK_SIZE_V=1024
    )

```

* **`grid = (B,)`**: 定义启动的 Thread Block 数量，正好是 Batch 里的 Sequence 数量。
* **`BLOCK_SIZE_V=1024`**: 在这里将编译期常量传入。1024 是大多数 GPU 极其适应的规约长度，能在不过度消耗 SRAM 共享内存的前提下，把张量核心（Tensor Cores / SIMD 单元）喂饱。
