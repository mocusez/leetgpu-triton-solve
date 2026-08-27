当前实现把 BFS 的**每一层扩展**放到一个 Triton kernel 中执行，而 Python 负责维护层数、切换队列以及判断结束条件。

## 1. 整体执行流程

初始时：

```text
current_frontier = [起点]
visited[起点] = 1
depth = 0
```

然后循环执行：

```text
第 depth 层 frontier
        │
        ▼
启动一个 Triton kernel
        │
        ├─ 检查当前层每个节点的四个邻居
        ├─ 标记新访问节点
        ├─ 把新节点写入 next_frontier
        └─ 如果邻居是终点，设置 found 标志
        │
        ▼
CPU 读取 counters
        │
        ├─ found == 1：返回 depth + 1
        ├─ next_count == 0：返回 -1
        └─ 否则交换两个 frontier，depth += 1
```

也就是说，Triton kernel 每次负责扩展完整的一层。

---

## 2. Python 侧的 GPU 数据结构

`solve()` 中创建了这些 GPU 张量：

```python
visited = torch.zeros(total_cells, ...)
current_frontier = torch.empty(total_cells, ...)
next_frontier = torch.empty(total_cells, ...)
counters = torch.zeros(2, ...)
```

它们的作用分别是：

| 张量 | 作用 |
|---|---|
| `visited[i]` | 第 `i` 个格子是否已访问，0 表示未访问，1 表示已访问 |
| `current_frontier` | 当前 BFS 层中的节点编号 |
| `next_frontier` | 下一 BFS 层中的节点编号 |
| `counters[0]` | 下一层节点数量 |
| `counters[1]` | 是否发现终点 |

节点不使用 `(row, col)` 两个数字存储，而是使用一维编号：

```python
node = row * cols + col
```

这样 frontier 中只需要存储一个 `int32`。

---

## 3. Triton kernel 如何映射线程

kernel 启动时：

```python
launch_grid = (
    triton.cdiv(current_count, BLOCK_SIZE),
)
```

其中：

```python
BLOCK_SIZE = 128
```

每个 Triton program 最多处理当前 frontier 中的 128 个节点：

```python
pid = tl.program_id(axis=0)
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
active = offsets < current_count
```

例如：

```text
current_count = 300
BLOCK_SIZE = 128
```

会启动 3 个 program：

```text
program 0：处理节点 0～127
program 1：处理节点 128～255
program 2：处理节点 256～299
```

最后一组中超过 `current_count` 的线程会被 `active` 屏蔽。

---

## 4. 读取当前层节点

```python
node = tl.load(
    current_frontier_ptr + offsets,
    mask=active,
    other=0,
)
```

每个有效线程读取一个当前层节点，然后恢复二维坐标：

```python
row = node // cols
col = node - row * cols
```

之后分别检查四个方向：

```python
node - cols  # 上
node + cols  # 下
node - 1     # 左
node + 1     # 右
```

边界通过条件判断，例如：

```python
active & (row > 0)
active & (row + 1 < rows)
active & (col > 0)
active & (col + 1 < cols)
```

越界方向不会真正读取 `grid`。

---

## 5. 如何判断邻居可以访问

每个方向都会调用 `_visit_neighbor()`。

首先读取网格：

```python
is_free = tl.load(
    grid_ptr + neighbor,
    mask=allowed,
    other=1,
) == 0
```

这里的逻辑是：

- `allowed=False`：不读取邻居；
- `grid[neighbor] == 0`：可以访问；
- `grid[neighbor] == 1`：障碍物。

因此：

```python
candidate = allowed & is_free
```

表示“在边界内并且是空格”。

---

## 6. 如何用原子操作防止重复入队

多个当前层节点可能同时拥有同一个邻居。例如：

```text
节点 A 的右邻居是 X
节点 B 的左邻居也是 X
```

如果没有保护，`X` 可能会被加入下一层多次。

代码使用：

```python
old_value = tl.atomic_xchg(
    visited_ptr + neighbor,
    1,
    mask=candidate,
)
```

`atomic_xchg` 会做两件事：

1. 原子地把 `visited[neighbor]` 设置为 1；
2. 返回修改之前的值。

因此：

```python
claimed = candidate & (old_value == 0)
```

表示当前线程是第一个访问该节点的线程。

例如两个线程同时访问 `X`：

```text
线程 1：看到旧值 0，成功认领 X
线程 2：看到旧值 1，不再入队
```

这样每个格子最多只会进入 frontier 一次。

---

## 7. 如何紧凑地写入下一层

成功认领的线程在 `BLOCK_SIZE=128` 的向量中可能不是连续的，例如：

```text
claimed = [1, 0, 1, 0, 0, 1]
```

如果直接按照线程位置写入，会产生空洞。因此代码先计算：

```python
claimed_int = claimed.to(tl.int32)
claimed_count = tl.sum(claimed_int, axis=0)
```

得到当前 program 新发现了多少个节点。

然后执行一次全局原子加法：

```python
base_pos = tl.atomic_add(
    counters_ptr,
    claimed_count,
)
```

这会为当前 program 在 `next_frontier` 中预留连续空间。

例如：

```text
counters[0] 原来是 20
当前 program 发现了 3 个节点
```

那么：

```text
base_pos = 20
counters[0] 变成 23
```

当前 program 可以使用位置：

```text
20、21、22
```

其他 program 会通过自己的 `atomic_add` 获得不同的位置。

---

## 8. `tl.cumsum` 的作用

当前 program 内部还需要给每个 claimed 节点分配一个唯一位置。

```python
exclusive_prefix = (
    tl.cumsum(claimed_int, axis=0) - claimed_int
)
```

例如：

```text
claimed_int       = [1, 0, 1, 0, 1]
cumsum            = [1, 1, 2, 2, 3]
exclusive_prefix  = [0, 1, 1, 2, 2]
```

最终位置：

```python
output_pos = base_pos + exclusive_prefix
```

如果 `base_pos=20`，那么三个 claimed 节点会写入：

```text
20、21、22
```

最后：

```python
tl.store(
    next_frontier_ptr + output_pos,
    neighbor,
    mask=claimed,
)
```

只有 `claimed=True` 的线程会真正写入。

这相当于在 GPU 上完成了一次并行 compaction，生成的 `next_frontier` 中没有空洞。

---

## 9. 如何发现终点

每个新认领的邻居都会检查：

```python
found = claimed & (neighbor == end_idx)
```

然后整个 program 聚合成一个标志：

```python
found_count = tl.sum(found.to(tl.int32), axis=0)
found_flag = (found_count > 0).to(tl.int32)
```

最后写入：

```python
tl.atomic_or(
    counters_ptr + 1,
    found_flag,
)
```

如果任何一个 program 发现终点：

```text
counters[1] = 1
```

---

## 10. 为什么每层的答案是正确的

一次 kernel 只会处理 `current_frontier` 中的节点。

而 `current_frontier` 中的所有节点距离起点都是 `depth`。

因此，在这一层中发现终点时，终点距离一定是：

```python
answer = depth + 1
```

不会出现更长的路径提前覆盖结果，因为：

- BFS 一层一层扩展；
- 每一层完整执行结束后才进入下一层；
- 终点第一次被发现时就是最短距离。

---

## 11. 每层结束后的同步

kernel 启动后，Python 执行：

```python
counters_cpu = counters.cpu()
```

这个操作有两个作用：

1. 等待当前 Triton kernel 完成；
2. 把 GPU 上的两个计数器复制到 CPU。

然后读取：

```python
next_count = int(counters_cpu[0].item())
found = int(counters_cpu[1].item())
```

根据结果判断：

```python
if found:
    answer = depth + 1
elif next_count == 0:
    answer = -1
else:
    进入下一层
```

所以这里每层有一次很短的 CPU-GPU 同步，只传输 `counters` 中的 8 字节。

---

## 12. 为什么交换两个数组

进入下一层之前：

```python
current_frontier, next_frontier = (
    next_frontier,
    current_frontier,
)
```

这样：

- 刚生成的下一层变成新的当前层；
- 原来的当前层变成下一次使用的临时数组；
- 不需要重新分配 GPU 内存。

然后：

```python
current_count = next_count
depth += 1
```

继续下一轮 BFS。

---

## 13. 复杂度

设：

- `V = rows × cols`
- `D = 起点到终点的最短距离`

每个格子最多访问一次，每个格子最多检查四个方向，因此 GPU 上的扩展工作量是：

```text
O(V)
```

显存占用：

```text
O(V)
```

但 Python 需要为每个 BFS 层启动一次 kernel，因此总开销还包含：

```text
O(D) 次 kernel 启动和同步
```

在一般的 500×500 网格中性能较好；如果地图被构造成一条非常长的单通道路径，`D` 可能接近 `V`，此时 kernel 启动次数会成为主要开销。