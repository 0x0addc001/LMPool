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
7. 控制进程下发 transfer 计划，worker 执行 NCCL 搬运并回报状态。

---

## 4. 组件

### 4.1 控制面

**文件**

- `src/lmpool/engine/control_plane.py`
- `src/lmpool/engine/global_scheduler.py`
- `src/lmpool/engine/global_block_manager.py`

控制面是独立进程，负责路由决策、全局页表更新、重平衡编排和 worker 心跳监测。

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
- `block_parent_hash`：用于驱逐时保持有效前缀链的父 hash 关系

当前权威状态保存在控制进程中，worker 通过状态消息回传本地快照。

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

通过 NCCL `send` / `recv` 实现 block 级迁移，按层搬运 K/V 张量。

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
| `foreground_transfer_cost_weight` | `float` | 每个 transferred block 折算的等价 prefill block 成本 |
| `foreground_transfer_min_benefit_ratio` | `float` | 预测节省 prefill 与 transfer 成本的最小比值 |
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

`benchmarks/` 下提供共享前缀压测脚本，场景包括：

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
保住可复用 KV”的 transfer。
worker 会向控制面上报每个 block 的访问频率和最近访问时间。foreground transfer 按
“单位目标缺失 block 可带来的前缀复用价值”排序完整前缀链，并用最近访问时间打破平局；
transfer 完成后目标 block 继承源端访问频率元数据。
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
