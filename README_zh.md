# LMPool：面向 LLM 推理的分布式 KV Cache 池化方案

[English](./README.md) | [简体中文](./README_zh.md)

---

## 目录

1. [概述](#1-概述)
2. [架构](#2-架构)
3. [运行逻辑](#3-运行逻辑)
4. [组件](#4-组件)
5. [实现](#5-实现)
6. [配置与运行](#6-配置与运行)
7. [测试](#7-测试)
8. [评估](#8-评估)
9. [当前状态与未来工作](#9-当前状态与未来工作)

---

## 1. 概述

LMPool 将集群内多张 GPU 的 HBM 抽象为一个逻辑统一的全局 KV Cache 池。它在 [Mini-vLLM](https://github.com/Wenyueh/MinivLLM) 的 Paged Attention 基础上，扩展了 KV Cache 感知的跨 GPU 路由和 NVLink KV transfer。

### 1.1 问题

| 局限 | 现象 | 后果 |
| --- | --- | --- |
| 无法跨卡复用前缀 | 共享前缀在不同 GPU 上重复存储 | 显存浪费、吞吐下降 |
| 显存压力下缺少弹性 | 本地 HBM 耗尽后触发 OOM 或 CPU fallback | 延迟飙升或请求中断 |
| 冷热分布失衡 | 冷块占据 HBM，热块被挤出 | 延迟持续上升 |

### 1.2 方案

1. `GlobalBlockManager` 维护跨 GPU 的全局页表。
2. 块级 hash 链编码前缀，支持前缀复用决策。
3. `GlobalScheduler` 负责请求路由与重平衡编排。
4. `kv_transfer` 负责基于 NCCL 的 KV transfer。
5. `LLMEngine` 作为 launcher / supervisor，启动独立控制进程和每卡数据面 worker。

---

## 2. 架构

当前实现将编排与执行分离：

- `LLMEngine`：launcher / supervisor
- `control_plane_process`：独立控制进程
- `data_plane_process`：按 rank 划分的数据面 worker

每个 worker 拥有自己的 `Scheduler`、`BlockManager` 和 `ModelRunner`。控制进程持有权威的 `GlobalScheduler` 与 `GlobalBlockManager` 状态，用于路由和重平衡编排。

![fig_architecture.png](/assets/fig_architecture.png)

---

## 3. 运行逻辑

1. `LLMEngine` 接收 prompt，并构造 `Sequence`。
2. `ControlPlaneClient` 计算所有完整前缀块的累积 hash chain。
3. 控制进程调用 `GlobalScheduler` 选择目标 rank。
4. `LLMEngine` 将 `Sequence` 转发给目标 worker。
5. worker 通过 `Scheduler` 和 `ModelRunner` 执行 prefill / decode。
6. 如果显存压力过高，worker 向控制进程请求 rebalance。
7. 控制进程下发 `prepare -> execute -> publish -> finalize` transfer 计划；
   所有目标块发布成功前，worker 始终保留源块。

---

## 4. 组件

### 4.1 控制面

**文件**

- `src/lmpool/engine/control_plane.py`
- `src/lmpool/engine/global_scheduler.py`
- `src/lmpool/engine/global_block_manager.py`

控制面是独立进程，负责路由决策、全局页表更新、重平衡编排和 worker 心跳监测。
control epoch 会丢弃旧控制进程的迟到响应；worker epoch 和单调递增的快照版本会拒绝
陈旧 worker 状态。心跳超时的 worker 会从路由候选和全局页表查询中移除，直到它重新
上报完整快照。

### 4.2 数据面

**文件**

- `src/lmpool/engine/data_plane.py`
- `src/lmpool/engine/scheduler.py`
- `src/lmpool/engine/block_manager.py`
- `src/lmpool/engine/model_runner.py`
- `src/lmpool/engine/kv_transfer.py`

每个数据面进程绑定一张 GPU，负责本地调度、KV 块分配、模型执行，以及控制面下发的 KV transfer 任务。

### 4.3 Sequence（序列）

**文件**：`src/lmpool/engine/sequence.py`

`Sequence` 携带 token ids、block table、completion token，以及远程前缀等全局池化元信息。

### 4.4 Global Scheduler（全局调度器）

**文件**：`src/lmpool/engine/global_scheduler.py`

`GlobalScheduler` 是跨 GPU 决策层，当前运行在独立控制进程中。它对外暴露两个主入口：

- `route_sequence_meta()`：请求路由
- `plan_rebalance()`：transfer 编排

路由策略：

1. 只对完整块计算累积 hash chain
2. 在全局页表中计算每张 GPU 从第 0 块开始的最长连续命中
3. 优先选择扣除队列压力后可复用块得分最高的 GPU
4. 如果没有命中，则回退到负载更低且有效容量足够的 GPU；有效容量为
   `空闲块 + 依赖安全的可回收 cache`
5. 路由后只对连续前缀未覆盖的新块进行乐观 reserve

每个 ingress reserve 会按 sequence 保留到首次 prefill 提交，防止并发请求重复承诺
同一批可回收块。

foreground transfer 对 `no_plan`、`no_target_space`、`stale_source` 等结构性
失败使用按 rank 指数退避，避免容量状态尚未变化时重复提交相同的失败请求。

当前路由权重如下：

| 关系 | 权重 |
| --- | --- |
| 同 GPU | 2.0 |
| NVLink 直连伙伴 | 1.0 |

### 4.5 Global Block Manager（全局块管理器）

**文件**：`src/lmpool/engine/global_block_manager.py`

`GlobalBlockManager` 维护：

- `global_page_table`：hash 到物理块位置
- `free_blocks_per_gpu`：每卡空闲容量
- 根据 ready-block 前缀 DAG 计算的依赖安全可回收容量
- 已路由但尚未提交 prefill 的逐 sequence 乐观块预留
- 每个 block 的访问频率和最近访问时间：用于 LFU 优先、LRU 次序选择
- `block_hash`：每卡 block hash 快照
- `block_generation`：物理 block id 每次复用时递增的代次，用于拒绝引用已回收
  block id 的陈旧 transfer 计划
- `block_parent_hash`：用于驱逐时保持有效前缀链的父 hash 关系

当前权威状态保存在控制进程中，worker 通过带版本号的状态消息回传本地快照。
控制进程重启后会向每个 worker 请求完整快照，在快照到达前该 rank 不参与路由。

### 4.6 Local Scheduler（本地调度器）

**文件**：`src/lmpool/engine/scheduler.py`

本地调度器维护 `waiting` 和 `running` 队列，并与 `BlockManager` 协作。

- Prefill：调度等待序列、分配 KV 块、执行前向推理
- Decode：追加 token 并继续运行
- 显存压力：向控制进程请求 rebalance

### 4.7 Local Block Manager（本地块管理器）

**文件**：`src/lmpool/engine/block_manager.py`

每个 worker 拥有一个 `BlockManager` 管理本地 KV cache 块。

主要职责：

- 计算链式 block hash
- 分配 block，并通过叶子约束的 LFU/LRU 回收冷缓存 block
- 追加 decode token
- 维护本地前缀缓存状态

完整且带 hash 的 block 在活跃引用数降为 0 后仍保留为前缀缓存；partial block 会立即释放。
缓存 block 会继续出现在全局页表中并保持可驱逐状态，直到容量压力触发回收。驱逐只选择
前缀链叶子，避免保留的后继 block 失去连续复用所需的祖先；先按访问频率排序合法叶子，
频率相同时再按最近访问时间排序。

### 4.8 Model Runner（模型执行器）

**文件**：`src/lmpool/engine/model_runner.py`

`ModelRunner` 持有模型权重、CUDA graph、KV cache 张量和采样器，是前向推理与 KV 迁移的执行点。

### 4.9 KV Transfer（KV 传输）

**文件**：`src/lmpool/engine/kv_transfer.py`

通过 NCCL `send` / `recv` 实现 block 级迁移。block 仍是放置单位，但一个 transfer plan
内的全部层和 K/V block 会打包成单个连续 P2P payload，并只执行一次阻塞 P2P 数据收发。

transfer 使用幂等的四阶段协议。`prepare` 校验源块 hash 和物理 block generation，锁定
源块以防本地回收，并预留目标块；`execute` 将 KV 写入仍对本地/全局前缀查询隐藏的目标块；
`publish` 在源块仍被锁定时发布全部有效目标块；所有 publish ACK 到齐后，`finalize` 才为
move-style 计划释放源块并上报新页表；`abort` 解锁源块并回收隐藏的目标预留。锁只覆盖
transfer 状态切换，不进入模型 forward 或逐 token decode 热路径。control epoch 变化时，
worker 会先清理未完成计划，再上报恢复快照。

### 4.10 Sequence（序列）

**文件**：`src/lmpool/engine/sequence.py`

`Sequence` 额外携带：

- `is_remote_prefix`
- `remote_gpu_id`
- `pending_swap_in`：legacy 字段名，表示待 transfer-in 的远端块

这些字段可通过 `multiprocessing.Queue` 跨进程传递。

---

## 5. [实现](./src/lmpool/)

---

## 6. 配置与运行

### 6.1 关键配置项

| 项目 | 类型 | 说明 |
| --- | --- | --- |
| `world_size` | `int` | 参与池化的 worker GPU 数量 |
| `enable_global_pool` | `bool` | 启用全局 KV Cache 池 |
| `use_control_plane_process` | `bool` | 启动独立控制进程 |
| `gpu_memory_utilization` | `float` | 可用 GPU 显存比例 |
| `heartbeat_interval` | `float` | 控制面与数据面的心跳周期 |
| `heartbeat_timeout` | `float` | 控制面 / worker 存活检测超时 |
| `nvlink_topo.pairs` | `List[Tuple[int, int]]` | 可选的 NVLink 直连 GPU 对；如果不配置，代码会尽量从 `nvidia-smi topo -m` 自动解析 |
| `route_prefix_hit_weight` | `float` | 全局路由中可复用前缀块的正向权重 |
| `route_queue_pressure_weight` | `float` | worker waiting/running 队列压力的惩罚权重 |
| `route_free_block_weight` | `float` | 空闲 KV block 的轻量 tie-breaker 权重 |
| `route_load_bypass_threshold` | `float` | 绕过 prefix owner 所需的最小 token 等价成本优势 |
| `route_prefill_cost_weight` | `float` | 预计完成成本模型中每个缺失 prefix token 的成本 |
| `route_reclaim_cost_weight` | `float` | 使用本地可回收 KV 容量时计入的附加成本 |
| `foreground_transfer_cost_weight` | `float` | 时间域 transfer 成本的整体倍率 |
| `foreground_transfer_min_benefit_ratio` | `float` | 预计节省 prefill 毫秒数与 transfer 毫秒数的最小比值 |
| `foreground_transfer_bandwidth_gib_s` | `float` | transfer admission 使用的实测有效带宽 |
| `foreground_transfer_fixed_latency_ms` | `float` | 每个计划的固定协议与协调开销 |
| `foreground_transfer_interference_multiplier` | `float` | 打包、解包和推理干扰倍率 |
| `foreground_prefill_token_time_ms` | `float` | 每个未缓存 prompt token 的预计重算耗时 |
| `foreground_future_reuse_discount` | `float` | 从历史叶前缀访问折算未来复用的折扣 |
| `foreground_transfer_ewma_alpha` | `float` | 源端实测 transfer 额外开销的 EWMA 权重 |
| `enable_kv_transfer_prewarm` | `bool` | worker ready 前初始化配置的 NVLink P2P communicator |
| `route_cache_queue_slack` | `float` | cached prefix route 可接受的最大预计完成成本余量 |

### 6.2 运行

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# 双卡示例
CUDA_VISIBLE_DEVICES=0,2 uv run python main.py

# 单卡示例
CUDA_VISIBLE_DEVICES=0 uv run python main.py
```

---

## 7. [测试](./tests/)

---

## 8. [评估](./benchmarks/)

`benchmarks/` 下只保留三个面向论文论证的可执行入口：

- `benchmark_kv_routing.py`：locality / routing-only 消融
- `benchmark_kv_transfer.py`：独立 NCCL/NVLink KV 数据通路
- `benchmark_e2e.py`：五配置 session-handoff 系统对比

论文实验请使用
[`benchmarks/PAPER_RUNBOOK.md`](./benchmarks/PAPER_RUNBOOK.md) 中固定的环境记录、测试和命令矩阵。

端到端脚本报告以下场景：

- `single-gpu`
- `multi-gpu`
- `multi-gpu-kv-routing`
- `multi-gpu-kv-transfer`
- `multi-gpu-lmpool`

报告指标包括：

- throughput，单位为生成 token/s
- goodput，单位为满足 `--goodput-e2e-sla-ms` 的生成 token/s
- mean / p95 TTFT
- mean / p95 TTPT
- mean / p95 端到端延迟
- GPU 利用率 mean / p95
- GPU 显存利用率 mean / p95
- 数据面请求命中率与 token 复用比例
- 控制面请求命中、owner 选择和匹配 block 比例
- transfer / copy count 和 rebalance success / failure count

当前 `multi-gpu` 基线采用在线 round-robin 分发。控制面场景如果要观察 prefix reuse，建议使用
`--submit-window 4` 或 `--submit-window 8`，因为只有前序请求完成 prefill 并上报全局页表后，
后续请求的路由决策才可能产生前缀命中。
当前 benchmark 的 TTFT 来自 data-plane worker 上报的 first-token 事件。local prefix hit 只统计
每个请求第一次 prefill 的本地命中，排除抢占后重试产生的命中；同时单独报告首次 cached-token
比例、prefill attempt、preemption 和重复处理的 prefill token，因此 round-robin 与 routing 的
真实 locality 收益可以直接比较。
控制面 route hit 和 prefix-owner hit 会单独报告。
五个场景通过 `--kv-block-budget` 使用相同的每 rank KV 容量请求值；prefix diagnostics
还会报告每个 rank 实际运行时 block 容量、workload 理论命中上界、路由匹配 block 比例和
stale-route 比例，避免将容量差异或陈旧页表误判为策略收益。
对于 `memory-skew`，benchmark 还会按确定性的预热、施压、复用三个阶段运行，并分别报告
发送 block 数、源端保留 block 数、源端释放 block 数、链式 transfer 计划数，以及复用阶段的
热点前缀 transfer 比例、请求命中率和 token 复用率。这样可以区分“尝试过 foreground transfer”和“确实释放容量并
保住可复用 KV”的 transfer。诊断表还会报告实际字节数、源端 transfer 时间、有效 GiB/s、
预计 transfer 成本和预计节省的 prefill 时间。
为了隔离验证 transfer，`session-handoff` 先只在 source 侧建立前缀，再让同一批会话跨
NVLink pair 继续。`--handoff-prefix-groups` 控制独立会话数；`--handoff-warmup-prompts`
控制建缓存阶段长度。将其设为 prefix group 数后，其余请求都用于多轮 reuse，避免 50/50
warmup 阶段稀释被测收益。
worker 会向控制面上报每个 block 的访问频率和最近访问时间。
控制面会从这些快照中提取最大热点前缀链，并为每个 NVLink pair 维护持久候选队列。
在 workload phase 边界，ingress 会提供尚未提交请求的逐前缀需求计数；控制面把相同方向、
相同 NVLink pair 的候选合并成有界批次，并且只在两端队列压力都较低时下发。一个批次只执行
一次 prepare/execute/publish/finalize 协议和一个连续 KV payload。phase 会等待已接受的放置计划完成，
等待时间仍计入 serving elapsed，因此 background transfer 是 reuse 到来前的主动放置，
而不是等 reuse 请求出现后才进行迟到复制。
被拒绝的 placement 决策会按前缀、NVLink pair、有效预测复用需求和目标容量缓存；状态未变
时，重复的 block-state 上报不会再次执行相同 admission。benchmark JSON 会分别记录 prompt、
cached 和实际参与计算的 uncached prefill token、placement 等待时间，以及每个 NVLink pair
的候选生命周期。成功计划还会把 dispatch 到 commit 的完整耗时反馈给该 pair 的成本模型。
worker 还会把已完成的 uncached-prefill 耗时反馈到逐 rank EWMA，使 placement admission
比较真实重算成本与启动校准、线上执行一致的 all-layer transport 成本。副本提交成功后会创建
受预测需求约束的 placement lease。lease 会把一半预测 reuse 需求固定分配给 replica；该配额
和另一半需求显式分配给仍然有效的 source；奇数余量在相邻 prefix 间交替归属两端。后续请求
按这些配额消费，从而既保留 prefix reuse，又让每个 NVLink pair 的两张 GPU 共同执行 reuse
阶段，避免只在 source 或 replica 单端串行执行。
未预测 eviction 时，admission 只计算目标端第一次冷 miss 可避免的 prefill，不再把同一收益
乘以所有未来 reuse 次数。
每个 NVLink pair 使用独立 NCCL process group，已完成 prepare 的计划跳过旧 block-ID 协商；
data-plane 同时等待 ingress/control 两条队列，避免空闲 worker 因等待请求队列而延迟 transfer。
首次 dispatch-to-commit 样本与启动校准先验进行 EWMA 融合，不再由一次冷启动抖动覆盖先验。
foreground transfer 按
“单位目标缺失 block 可带来的前缀复用价值”排序完整前缀链，并用最近访问时间打破平局；
admission 使用保守的 wall-clock 成本模型，不再把同一请求在前缀链每个 block 上的访问次数
重复相加；transfer 完成后目标 block 继承源端访问频率元数据。
对于 topology-blind baseline 和 transfer-only，memory-skew 的复用请求会落到 NVLink pair
另一侧。serving 计时在 worker ready 和代表性 KV payload 的 P2P prewarm 后开始；完成的
source transfer 会将实测额外延迟通过每个 NVLink pair 独立的保守 EWMA 回流 admission，
且同一 pair 同时最多执行一个 foreground plan。transfer 完成后会增加或移动全局页表位置，
后续 routing 可以选择新的 prefix owner，但仍会比较负载成本，不会被强制路由过去。
路由负载预留会计入预计 decode 工作。当 prefix owner 明显比 NVLink partner 更忙时，系统
可以直接 spill，并由该请求在 partner 自然建立缓存。主动副本由已完成访问统计和 ingress
排队需求独立规划，因此 routing 不会因为期待一个尚未完成的 copy，而把当前请求继续留在
已经过载的 owner 上。
`locality` workload 默认使用 16 组不同的长共享前缀，并按固定 seed 打乱请求顺序；可通过
`--locality-prefix-groups` 调整。多前缀组可以避免 round-robin 仅靠在每张 GPU 上复制同一个
热点前缀，就获得与 routing 接近的稳态命中率。
benchmark 默认忽略 EOS，保证每条请求执行相同的 decode 工作量。论文实验建议显式设置
`--seed`，并使用 `--repetitions 3` 或更高次数报告 mean/std。

具体用法见 `benchmarks/README.md`。

---

## 9. 当前状态与未来工作

| 功能 | 状态 | 说明 |
| --- | --- | --- |
| 多 GPU 异步推理 | 已完成 | 多个 rank 独立调度、执行、采样 |
| 控制面路由 | 已完成 | `route_sequence_meta` 已实现 |
| NVLink 驱逐决策 | 已完成 | `select_eviction_candidates` 已实现 |
| Benchmarks | 已完成 | 已增加共享前缀压测和基线对比 |
| Tests | 已完成 | 已补充模块级单测与 NCCL 集成测试 |

未来工作：

1. 继续扩展共享前缀基准测试到更长 trace 和更复杂负载。
2. 继续更新 README 和补充注释
