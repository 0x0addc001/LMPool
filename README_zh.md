# LMPool：面向 LLM 推理的分布式 KV Cache 池化方案

**基于 [Mini-vLLM](https://github.com/Wenyueh/MinivLLM) 构建** | 原型阶段

---

## 目录

1. [概述](#1-概述)
2. [架构](#2-架构)
3. [组件](#3-组件)
   - 3.1 [Global Scheduler（全局调度器）](#31-global-scheduler全局调度器)
   - 3.2 [Global Block Manager（全局块管理器）](#32-global-block-manager全局块管理器)
   - 3.3 [Local Scheduler（本地调度器）](#33-local-scheduler本地调度器)
   - 3.4 [Local Block Manager（本地块管理器）](#34-local-block-manager本地块管理器)
   - 3.5 [Model Runner（模型执行器）](#35-model-runner模型执行器)
   - 3.6 [KV Transfer（KV 传输）](#36-kv-transferkv-传输)
   - 3.7 [Sequence（序列）](#37-sequence序列)
4. [配置与运行](#4-配置与运行)
5. [当前状态与未来工作](#5-当前状态与未来工作)

---

## 1. 概述

LMPool 将集群内多张 GPU 的 HBM 抽象为一个逻辑统一的全局 KV Cache 池，它在 Mini-vLLM 的 Paged Attention 基础上，扩展了跨 GPU 的块级前缀感知路由和冷热/拓扑感知驱逐。

### 1.1 问题

在 vLLM 原始的 Paged Attention 中，每张 GPU 独立管理自己的显存，存在三个局限：

| 局限             | 现象                                   | 后果                   |
| ---------------- | -------------------------------------- | ---------------------- |
| 无法跨卡复用前缀 | 多个请求共享相同前缀，但各卡各自存一份 | 显存浪费，有效吞吐下降 |
| OOM 时无弹性     | 本地 HBM 耗尽 → OOM 或触发 CPU swap    | 延迟飙升或请求中断     |
| 冷热分布失衡     | 本地 HBM 逐渐充满冷块，热块被挤到 CPU  | 延迟持续上升           |

### 1.2 方案

将多 GPU 的 HBM 抽象为统一的分布式显存池：

1. **逻辑统一**：`GlobalBlockManager` 维护跨 GPU 的全局页表，记录每个 KV 块的物理位置
2. **前缀去重**：块级 hash 链编码前缀，跨 GPU 查重，相同前缀只存一份
3. **冷热/拓扑感知**：LRU 驱逐 + 拓扑优先的 swap（NVLink > PIX > NODE）
4. **控制面/数据面分离**：`GlobalScheduler` 做决策，`kv_transfer` 做 NCCL 搬运

---

## 2. 架构

每个 GPU 进程运行一个独立的 `LLMEngine` 实例，拥有自己的 `Scheduler`、`BlockManager` 和 `ModelRunner`。`GlobalBlockManager`（权威副本在 rank 0）和 `GlobalScheduler` 作为跨 GPU 协调层叠加在其上。

```
┌──────────────────────────────────────────────────────┐
│                    控制面 (Control Plane)             │
│  ┌──────────────────┐  ┌─────────────────────────┐   │
│  │ GlobalScheduler  │  │  GlobalBlockManager     │   │
│  │ - route_sequence │  │  - global_page_table    │   │
│  │ - rebalance      │  │  - free_blocks_per_gpu  │   │
│  └────────┬─────────┘  └────────────┬────────────┘   │
└───────────┼─────────────────────────┼────────────────┘
            │                         │
┌───────────┼─────────────────────────┼────────────────┐
│           ▼      数据面 (Data Plane) ▼                │
│  ┌──────────────┐  NVLink/NCCL  ┌──────────────┐     │
│  │    GPU 0     │◄─────────────►│    GPU 1     │     │
│  │ - Scheduler  │  swap_out/in  │ - Scheduler  │     │
│  │ - BlockMgr   │               │ - BlockMgr   │     │
│  │ - ModelRunner│               │ - ModelRunner│     │
│  └──────────────┘               └──────────────┘     │
└──────────────────────────────────────────────────────┘
```

---

## 3. 组件

### 3.1 Global Scheduler（全局调度器）

**文件**：`src/lmpool/engine/global_scheduler.py` → `GlobalScheduler`

`GlobalScheduler` 是跨 GPU 决策层。初始化时持有 `GlobalBlockManager`（用于页表查询）、本地 `BlockManager`（用于 hash 计算）以及可选的 `ModelRunner`（用于执行 KV 传输）引用。它对外暴露两个主入口：请求路由（`route_sequence`）和显存重平衡（`rebalance`）。

#### 3.1.1 Routing（`route_sequence`）

##### 3.1.1.1 路由决策

决定一个新到来的序列应该在哪个 GPU 上执行。算法按优先级顺序执行以下六步：

1. **计算前缀 hash**：调用 `_compute_prefix_hash(seq)`，该方法只对序列的完整块（partial 尾块排除在外）做哈希。对第 *i* 个完整块，调用 `BlockManager.compute_hash(block_tokens, prev_hash)`，将第 *i-1* 块的 hash 链入当前块的输入。如果序列没有任何完整块，返回 `None`，路由直接跳到第 6 步。

2. **查询全局页表**：调用 `gbm.lookup_prefix(prefix_hash)`，返回 `List[BlockLocation]`，每个条目携带 `(gpu_id, block_id, hash, last_access_time)`。若结果为空，跳到第 6 步。

3. **按 GPU 聚合命中块数**：遍历 `BlockLocation` 列表，累计 `gpu_hit_count[gpu_id] += 1`。

4. **加权打分**

对每个候选 GPU：`score = hit_count × topo_weight`。拓扑权重由 `_get_topo_weight(my_rank, target_gpu)` 计算：

| 关系             | 权重 |
| ---------------- | ---- |
| 同 GPU           | 3.0  |
| NVLink 直连      | 2.0  |
| 同 Socket (PIX)  | 1.5  |
| 跨 Socket (NODE) | 1.0  |

5. **选择得分最高且空闲块足够的 GPU**

遍历打分候选，只有满足 `gbm.get_free_blocks_count(gpu_id) >= seq.num_blocks` 的 GPU 才合格。如果得分最高的候选 GPU 没有足够空闲块，它依然被返回，预期后续 `rebalance()` 会腾出空间。

6. **选空闲最多的 GPU兜底**

当没有前缀 hash 或没有任何命中时，`_select_most_free_gpu` 扫描所有 rank 返回空闲块数最大的 GPU，相同时优先本地 rank。

##### 3.1.1.2 Hash 链与块级前缀匹配

第 *i* 个块的 hash 编码了从 block 0 到 block i 的全部 token 内容：

```
hash_0 = xxhash64(tokens[0 : block_size])
hash_1 = xxhash64(hash_0.to_bytes(8) + tokens[block_size : 2*block_size])
...
hash_k = xxhash64(hash_{k-1}.to_bytes(8) + tokens[k*block_size : (k+1)*block_size])
```

两个序列若共享长度为 *k × block_size* 的前缀，必然产生相同的 `hash_k`。查询 `global_page_table[hash_k]` 即可找到所有持有该前缀的 GPU，无需比对 token 序列本身。

只有完整块会被哈希。Partial 块始终赋予 hash `-1`，避免序列生长过程中产生虚假命中。

#### 3.1.2 Swapping（`rebalance`）

##### 3.1.2.1 交换决策

`rebalance(gpu_id, needed_blocks)` 在某个 GPU 需要 `needed_blocks` 个空闲块但本地不足时调用，负责编排跨 GPU 的 swap 来腾出空间。

1. **获取驱逐候选**：调用 `gbm.select_eviction_candidates(gpu_id, needed_blocks)`，返回 `[(local_block_id, target_gpu_id), ...]`（三级驱逐策略见 §3.2.3）。

2. **数量校验**：若 `len(candidates) < needed_blocks`，直接返回 `False`。

3. **按目标 GPU 分组**：将候选列表按 `target_gpu` 分组为 `groups: dict[int, List[int]]`。

4. **执行 NCCL 传输**：对每组，拥有 `gpu_id` 的 rank 调用 `_execute_swap_out(blocks, gpu_id, target_gpu)`（委托给 `ModelRunner.execute_swap_out`），拥有 `target_gpu` 的 rank 调用 `_execute_swap_in_accept(blocks, gpu_id, target_gpu)`（委托给 `ModelRunner.execute_swap_in`）。两侧在传输前后均调用 `dist.barrier()` 同步。

5. **更新全局页表**：对每个被驱逐的块调用 `gbm.free_global(gpu_id, [local_block])`，再通过 `gbm.record_block_transfer()` 更新空闲计数和页表条目。

6. **抢占兜底**：当 `rebalance()` 仍无法满足空间需求时（例如所有远端 GPU 也满了），`preempt_for_rebalance(running_sequences, gpu_id, needed_blocks)` 作为最后手段：按顺序遍历 `running_sequences`，将最短的序列标记为 `WAITING`，释放其所有块，直到累计释放量满足 `needed_blocks`。被抢占的序列的 `block_table` 被清空，`num_cached_tokens` 重置为 0，等待下次重新调度。

---

### 3.2 Global Block Manager（全局块管理器）

**文件**：`src/lmpool/engine/global_block_manager.py` → `GlobalBlockManager`

`GlobalBlockManager` 是分布式 KV Cache 池的权威注册中心。rank 0（可通过 `master_rank` 配置）持有所有状态的主副本，其他 rank 持有本地缓存并定期刷新。

#### 3.2.1 属性

| 属性                  | 类型                             | 描述                               |
| --------------------- | -------------------------------- | ---------------------------------- |
| `global_page_table`   | `Dict[int, List[BlockLocation]]` | `prefix_hash →` 所有副本的物理位置 |
| `free_blocks_per_gpu` | `List[int]`                      | 每 GPU 的空闲块计数                |
| `block_access_time`   | `List[Dict[int, float]]`         | 每 GPU、每块的最近访问时间（LRU）  |
| `block_hash`          | `List[Dict[int, int]]`           | 每 GPU `block_id → hash` 映射      |
| `master_rank`         | `int`                            | 权威主节点的 rank（默认 0）        |
| `nvlink_pairs`        | `Set[Tuple[int, int]]`           | NVLink 直连 GPU 对                 |
| `socket_groups`       | `List[List[int]]`                | 按 CPU Socket 的 GPU 分组          |

`BlockLocation` 数据类携带四个属性——`gpu_id`、`block_id`、`hash`、`last_access_time`——是描述 KV 块物理位置的规范表示。

#### 3.2.2 分布式内存分配

##### 3.2.2.1 前缀去重

当本地 `BlockManager` 为新序列分配块时，最后一个完整块的 hash 会通过 `_commit_alloc(gpu_id, block_ids, hashes)` 提交到 `GlobalBlockManager`。

`_commit_alloc(gpu_id, block_ids, hashes)` 是核心写路径。对每个块，依次：

1. `free_blocks_per_gpu[gpu_id] -= 1`
2. `block_access_time[gpu_id][bid] = now`（用于 LRU）
3. `block_hash[gpu_id][bid] = h`
4. 在 `global_page_table[h]` 末尾追加 `BlockLocation(gpu_id, bid, h, now)`

若该 hash 已存在于页表中（即其他 GPU 上已有相同前缀的副本），则新增一条 `BlockLocation` 条目——两个副本均被注册，路由时可选择拓扑最近的那个。

`BlockManager.allocate(seq)` 驱动这一过程：遍历新序列的所有块，计算链式 hash，检查本地 `hash_to_block_id` 是否命中——命中则复用（`ref_count++` 并累加 `seq.num_cached_tokens`），未命中则分配新块。最终通过 `gbm._commit_alloc` 将最后一个完整块的 hash 注册到全局。

##### 3.2.2.2 冷热/拓扑感知分配

当本地空闲块不足时，调用 `BlockManager.allocate_with_swap(seq)` 而非直接 `allocate`：

1. **检查本地空闲**：若 `can_allocate(seq)` 为真，直接调用 `allocate()` 返回
2. **计算缺口**：`shortage = seq.num_blocks - len(free_block_ids)`
3. **获取驱逐候选**：调用 `gbm.select_eviction_candidates(rank, shortage)`，返回拓扑感知的 `(local_block, target_gpu)` 对（见 §3.2.3）
4. **释放冷块**：对每个 `ref_count == 0` 的候选块，调用 `_deallocate_block` 归还到 `free_block_ids`，再调用 `gbm.record_block_transfer()` 更新全局页表
5. **正常分配**：腾出足够空间后调用 `allocate(seq)`

Decode 阶段的对应入口为 `append_with_swap(seq)`，逻辑相同但每次只需腾出 1 个块。

#### 3.2.3 冷热/拓扑感知驱逐（`select_eviction_candidates`）

`select_eviction_candidates(gpu_id, num_blocks) → List[Tuple[int, int]]` 以三级递进策略为每个冷块找到 swap 目标：

**第一级 选出本地最冷的块**

对 `block_access_time[gpu_id]` 按时间戳升序排序，取前 `num_blocks` 个作为 `cold_blocks`。

**第二级 为每个冷块找有空闲的目标 GPU**

按 `_get_target_gpu_order(gpu_id)` 返回的拓扑优先级顺序迭代候选目标：

1. **优先级 1**：NVLink 直连
2. **优先级 2**：同 Socket GPU（按空闲块数降序）
3. **优先级 3**：跨 Socket GPU（按空闲块数降序）

找到第一个 `free_blocks_per_gpu[target] > 0` 的目标即停止，并临时将该目标的空闲计数减 1，防止后续块重复占用同一 slot。

**第三级 递归驱逐 / 覆盖写入**

若所有目标 GPU 都无空闲块：

1. 调用 `_select_remote_victim(target)` 在拓扑最近目标上选出 LRU 最冷的块
2. 从 `block_access_time`、`block_hash` 和 `global_page_table` 中删除该 victim 的条目，等效于递归驱逐
3. 目标因此腾出一个空闲 slot，可接收本地冷块
4. 若远端也完全为空（无块可选），则直接覆盖写入，即目标块计数不变，旧数据在传输到达时被覆盖

`_get_target_gpu_order(gpu_id)` 构建顺序如下：

```python
ordered = []
# 第一级：NVLink 直连，带宽最高
partner = nvlink_partner.get(gpu_id)
if partner: ordered.append(partner)

# 第二级：同 Socket GPU，按空闲块数降序
same_socket.sort(key=lambda g: free_blocks_per_gpu[g], reverse=True)
ordered.extend(same_socket)

# 第三级：跨 Socket GPU，按空闲块数降序
other_socket.sort(key=lambda g: free_blocks_per_gpu[g], reverse=True)
ordered.extend(other_socket)
```

#### 3.2.4 前缀查找（`lookup_prefix`）

`lookup_prefix(prefix_hash) → List[BlockLocation]`

1. 若 `prefix_hash` 不在 `global_page_table` 中，直接返回 `[]`
2. 取出所有匹配的 `BlockLocation`，按 NVLink 亲和性打分排序：NVLink 伙伴上的块得分 2.0，同 Socket 得分 1.5，其他得分 1.0，分数越高排越前
3. 返回排序后的列表供路由器直接取最优选项

注意：`lookup_prefix` 的排序权重是从**调用者**（当前 rank）视角衡量目标 GPU 的远近，与驱逐时的目标排序方向相同但基准不同。

#### 3.2.5 全局页表同步

`GlobalBlockManager` 采用 **master-push 同步模型**：

**`update_gpu_state(gpu_id, free_blocks, block_hashes)`**

Master-only 状态摄入边界。Worker 在每次分配、追加、抢占或序列结束后通过 `Scheduler._sync_local_state_to_global()` 调用此方法。它原子地替换 master 对 `gpu_id` 的状态视图：先清除该 GPU 在全局页表中的所有旧条目，再从新的 `block_hashes` 快照重新插入所有条目。

**`reserve_blocks(gpu_id, num_blocks)`**

在请求被路由到远端 GPU 后、远端 worker 尚未调用 `update_gpu_state` 之前，乐观地将 `free_blocks_per_gpu[gpu_id]` 减少 `num_blocks`，防止在短暂的状态延迟窗口内向同一 GPU 过度路由。

**`broadcast_page_table()`**

先调用 `gather_local_state()`——通过 `dist.all_gather_into_tensor` 收集每个 rank 当前的 `free_blocks_per_gpu` 值；再通过 `dist.broadcast_object_list` 将完整的 `(global_page_table, free_blocks_per_gpu, block_access_time, block_hash, master_rank)` 元组从 `master_rank` 广播到所有 rank。非 master rank 覆盖本地缓存。

**`maybe_sync()`**

内部计数器递增，每 `sync_interval`（默认 10）轮调度后调用一次 `broadcast_page_table()`（目前在 `Scheduler` 中被注释掉，需要重新启用才能激活跨 GPU 前缀复用）。

#### 3.2.6 Master 故障迁移

`GlobalBlockManager` 包含故障检测功能，但尚未启用：

1. `check_master_health()`：master 刷新自身 `master_heartbeat` 时间戳；非 master rank 调用 `_broadcast_master_heartbeat()` 接收心跳，若 `now - heartbeat > heartbeat_timeout`（默认 100 s）则发起选举
2. `_elect_new_master()`：简化轮转策略：`new_master = (old_master + 1) % world_size`
3. `set_master_rank(new_master)`：手动指定新管理节点，用于灾后重配置

#### 3.2.7 块生命周期

```
                    ┌──────────┐
          allocate  │   FREE   │  deallocate
       ┌────────────│          │◄───────────┐
       │            └─────┬────┘            │
       │                  │                 │
       │                  │ swap_in         │
       │                  ▼                 │
       │            ┌──────────┐            │
       │ ++ref > 0  │  ALLOC   │ --ref <= 0 │
       └───────────►│          │────────────┘
   (前缀命中复用)    └─────┬────┘  (序列完成/抢占)
                          │
                          │ swap_out
                          │ 
                          ▼
                    ┌──────────┐
                    │  REMOTE  │
                    │          │
                    └──────────┘
```

---

### 3.3 Local Scheduler（本地调度器）

**文件**：`src/lmpool/engine/scheduler.py` → `Scheduler`

本地调度器管理 `waiting` 和 `running` 双端队列，与 `BlockManager` 配合做内存分配决策。注入 `global_scheduler` 后激活两处扩展 hook。

#### 3.3.1 Prefill 阶段

**远程路由**：对 `waiting` 队首的每个序列，若 `remote_gpu_id` 未设置，调用 `global_scheduler.route_sequence(seq)`。若返回的目标是其他 GPU，则将序列从 `waiting` 弹出，状态设为 `RUNNING`，加入 `scheduled_sequences`，不做本地块分配，同时调用 `gbm.reserve_blocks(target_gpu, seq.num_blocks)` 乐观地减少目标 GPU 的空闲计数。实际块分配将在目标 rank 上进行。

**Swap 辅助分配**：若本地路由的序列因空闲块不足无法分配，调用 `block_manager.allocate_with_swap(seq)`，内部触发 `gbm.select_eviction_candidates` 并驱逐冷块腾出空间。若仍失败，prefill 循环 break，序列留在 `waiting`。

#### 3.3.2 Decode 阶段

**Append 失败时的重平衡**：当 `block_manager.can_append(seq)` 返回 `False` 时，调用 `global_scheduler.rebalance(self.rank, 1)`。若 rebalance 成功，将序列压回 `running` 队首，下次迭代重试；若失败，执行原有抢占逻辑（最短运行序列被抢占）。

#### 3.3.3 状态同步

每次分配、追加、抢占或序列结束后，`_sync_local_state_to_global()` 将本地 `BlockManager` 的 `(free_count, block_hashes)` 快照推送到 master `GlobalBlockManager`，通过 `gbm.update_gpu_state(rank, free_count, block_hashes)` 调用实现。

---

### 3.4 Local Block Manager（本地块管理器）

**文件**：`src/lmpool/engine/block_manager.py` → `BlockManager`

每个 GPU 进程拥有一个 `BlockManager` 实例，管理该 GPU 上的物理 KV Cache 块：`free_block_ids`（deque）、`used_block_ids`（set）、本地 `hash_to_block_id` 前缀缓存，以及指向 `GlobalBlockManager` 的引用。

**`compute_hash(token_ids, prefix_hash_value)`**

以 numpy `int32` 数组的二进制表示为输入计算 `xxhash64`。若 `prefix_hash_value != -1`，先将其 8 字节小端编码馈入 hasher，实现 hash 链化。

**`allocate(seq)`**

遍历序列的所有块。对每个完整块，计算链式 hash，查 `hash_to_block_id` 是否本地命中：命中则 `ref_count++` 并累加 `seq.num_cached_tokens`；未命中则从 `free_block_ids` 取一个新块，调用 `block.update(h, token_ids)` 并写入 `hash_to_block_id`。Partial 尾块始终分配新块，hash 保留 `-1`。所有块处理完成后，若 `gbm` 不为 `None`，调用 `gbm._commit_alloc` 注册最后一个完整块。

**`deallocate(seq)`**

对 `seq.block_table` 中每个块 `ref_count -= 1`；降至 0 的块归还到 `free_block_ids` 并从 `hash_to_block_id` 移除。最后清空 `seq.block_table` 和 `seq.num_cached_tokens`。

**`append(seq)`**

追加一个新 token 后调用：

1. 若新 token 恰好填满一个块（`num_tokens % block_size == 0`）：计算并存储该块的 hash，写入 `hash_to_block_id` 并通知 `gbm._commit_alloc`
2. 若新 token 是新块的第一个 token（`num_tokens % block_size == 1`）：从 `free_block_ids` 分配一个新块
3. 其他情况：token 写入已有 partial 块，无需操作

**`can_allocate(seq)`**：`len(free_block_ids) >= seq.num_blocks`

**`can_append(seq)`**：若下一个 token 会开启新块，检查 `free_block_ids` 非空；否则直接返回 `True`

---

### 3.5 Model Runner（模型执行器）

**文件**：`src/lmpool/engine/model_runner.py` → `ModelRunner`

`ModelRunner` 持有模型权重、CUDA graph 捕获、KV cache 张量和采样器，是 KV cache 物理内存分配和跨 GPU 数据传输的唯一执行点。

#### 3.5.1 KV Cache 分配（`allocate_kv_cache`）

Warmup forward pass 后，计算 `available_mem = free_mem × gpu_memory_utilization - (peak_warmup - current)`，再除以每块字节代价（`block_size × 2 × num_layers × num_kv_heads × head_dim × dtype_bytes`）得到 `num_available_kv_blocks`。

多 GPU 模式下，通过 `dist.all_reduce(..., op=MIN)` 让所有 rank 取最保守的块数，保证 block table 大小全局一致。

KV cache 以单张量 `(2, num_layers, max_cached_blocks, block_size, num_kv_heads, head_dim)` 分配，再按层切片赋给每个 attention 模块的 `k_cache` / `v_cache` 属性。

#### 3.5.2 权重加载与广播

rank 0 从磁盘加载权重，随后两次 `dist.barrier()` 之间 rank 0 遍历 `model.parameters()` 对每个参数调用 `dist.broadcast(param.data, src=0)`，确保所有 rank 在进入主循环前持有相同权重。

#### 3.5.3 CUDA Graph 捕获（`capture_cudagraph`）

对 Decode batch size `[1, 2, 4, 8, 16, 32, ...]`（上限 `max_num_seqs`）各捕获一张 CUDA graph，存入 `self.graphs[batch_size]`。推理时 `run_model` 找到最小的能容纳当前 batch 的 graph 并原地更新输入张量后 replay，消除 CPU launch overhead。

#### 3.5.4 远程块拉取（`_swap_in_remote_blocks(seq)`）

在 `run()` 的 model forward 之前，对所有 `pending_swap_in` 非空的序列执行：扫描 model modules 定位 `k_cache` 张量，调用 `kv_transfer.swap_in(remote_gpu, remote_blocks, local_gpu, kv_cache, ...)`，将返回的 `local_blocks` 写入 `seq.block_table` 的对应前缀位置，最后清除 `pending_swap_in`、`is_remote_prefix`、`remote_gpu_id`。

#### 3.5.5 Swap 执行（`execute_swap_out` / `execute_swap_in`）

两个方法均先调用 `_get_kv_cache()` 定位 KV cache 张量，再分别委托给 `kv_transfer.swap_out` 和 `kv_transfer.swap_in`。由 `GlobalScheduler._execute_swap_out` / `_execute_swap_in_accept` 调用。

---

### 3.6 KV Transfer（KV 传输）

**文件**：`src/lmpool/engine/kv_transfer.py`

实现基于 NCCL `send`/`recv` 的两个跨 GPU 块迁移原语。Tag 编码 `block_id × 10000 + layer_idx × 2 + is_k` 保证 K/V 张量即使在多块并发传输时也不发生 tag 冲突。

#### 3.6.1 驱逐（`swap_out`）

将源 GPU 上的冷 KV 块搬运到目标 GPU：

1. **块索引协商**：源端通过 `_send_block_list` 发送待驱逐块列表；若指定了 `target_free_blocks`，一并发送；目标端接收后将分配好的目标 block id 回传
2. **逐层逐块传输**：对每层 `0..num_layers-1`，源端读取 `layer_kv[0, src_block]`（K）和 `layer_kv[1, src_block]`（V），调用 `dist.send`；目标端分配零缓冲区，调用 `dist.recv`，再 `copy_` 写入 `layer_kv[0/1, dst_block]`
3. **全局 barrier**：对标量零张量调用 `dist.all_reduce`，确保两端传输完成前不返回

#### 3.6.2 拉取（`swap_in`）

从远端 GPU 拉取 KV 块到本地，协议是 `swap_out` 的镜像：本地发送想要的远端块列表 → 远端回传映射 → K/V 数据逐层反向传输。

单 GPU 场景下，两端是同一 rank，NCCL 调用自动退化为无操作（`target_blocks = blocks_to_evict`）。

---

### 3.7 Sequence（序列）

**文件**：`src/lmpool/engine/sequence.py` → `Sequence`

为全局池化新增三个字段：

| 字段               | 类型        | 描述                                           |
| ------------------ | ----------- | ---------------------------------------------- |
| `is_remote_prefix` | `bool`      | 是否使用了远程 GPU 上前缀的 KV cache           |
| `remote_gpu_id`    | `int`       | 远程前缀所在的 GPU rank；-1 表示所有块都在本地 |
| `pending_swap_in`  | `List[int]` | 等待从远端拉取的物理块 ID 列表                 |

三个字段均包含在 `__getstate__` / `__setstate__` 中，确保通过 `multiprocessing.Queue` 跨进程传递时不丢失。

---

## 4. 配置与运行

### 4.1 关键配置项

| 配置项                            | 类型                    | 描述                                     |
| --------------------------------- | ----------------------- | ---------------------------------------- |
| `world_size`                      | `int`                   | 参与池化的 GPU 数量                      |
| `enable_global_pool`              | `bool`                  | 启用全局 KV Cache 池                     |
| `gpu_memory_utilization`          | `float`                 | 可用 GPU 显存比例（值越小越早触发 swap） |
| `swap_threshold`                  | `float`                 | 触发 swap 的 GPU 显存使用率阈值          |
| `global_page_table_sync_interval` | `int`                   | 页表广播间隔（调度周期数）               |
| `nvlink_topo.pairs`               | `List[Tuple[int, int]]` | NVLink 直连 GPU 对                       |
| `nvlink_topo.sockets`             | `List[List[int]]`       | 按 CPU Socket 的 GPU 分组                |

### 4.2 运行命令

```bash
# 安装 uv 包管理器
curl -LsSf https://astral.sh/uv/install.sh | sh

# 同步依赖
uv sync

# 双卡（NVLink 直连）：
CUDA_VISIBLE_DEVICES=0,2 uv run python main.py

# 单卡：
CUDA_VISIBLE_DEVICES=0 uv run python main.py

# 八卡：
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run python main.py
```

---

## 5. 当前状态与未来工作

### 5.1 当前状态

| 功能              | 状态       | 说明                                        |
| ----------------- | ---------- | ------------------------------------------ |
| 多 GPU 异步推理 | ✅ 已完成     | 多个 rank 独立调度、执行、采样                 |
| 块级前缀感知路由决策  | ✅ 已完成     | `route_sequence` 已实现                     |
| 全局页表同步/前缀复用| 🛠️ 进行中 | `maybe_sync` 暂时注释，本地页表独立未同步     |
| 冷热/拓扑感知驱逐决策 | ✅ 已完成 | `select_eviction_candidates` 已实现      |
| 跨 GPU 块迁移原语| 🛠️ 进行中 | `swap_out`/`swap_in` 待联调                    |
| Benchmarks | ❌ 未实现 | 吞吐量、延迟、前缀命中率等指标，与基线对比 |
| Tests | ❌ 未实现 | 各组件单元测试 |

### 5.2 未来工作

1. **全局页表同步与前缀复用**：恢复 `maybe_sync()` 调用，使 `lookup_prefix` 能跨 GPU 命中；修改 `BlockManager.allocate` 接受 `BlockLocation` hint，直接引用远端已有物理块而非重新分配。
2. **跨 GPU 块迁移端到端联调**：构造 Rank 0 → Rank 1 的 NCCL send/recv 场景，验证 `swap_out`/`swap_in` 无死锁且搬运前后 KV 数据一致。
3. **Benchmarks**：构造高并发、长共享前缀的压测场景，量化相比单卡基线的吞吐量、TTFT、前缀命中率提升。
4. **Tests**：覆盖路由决策（命中/未命中/空闲不足）、驱逐策略（三级 fallback）、页表同步一致性、swap 原语正确性。