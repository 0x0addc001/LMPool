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

LMPool 将集群内多张 GPU 的 HBM 抽象为一个逻辑统一的全局 KV Cache 池。它在 [Mini-vLLM](https://github.com/Wenyueh/MinivLLM) 的 Paged Attention 基础上，扩展了 KV Cache 感知的跨 GPU 路由和驱逐。

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
4. `kv_transfer` 负责基于 NCCL 的 KV 迁移。
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
2. `ControlPlaneClient` 计算完整块前缀 hash。
3. 控制进程调用 `GlobalScheduler` 选择目标 rank。
4. `LLMEngine` 将 `Sequence` 转发给目标 worker。
5. worker 通过 `Scheduler` 和 `ModelRunner` 执行 prefill / decode。
6. 如果显存压力过高，worker 向控制进程请求 rebalance。
7. 控制进程下发 swap 计划，worker 执行 NCCL 搬运并回报状态。

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

每个数据面进程绑定一张 GPU，负责本地调度、KV 块分配、模型执行，以及控制面下发的 swap 任务。

### 4.3 Sequence（序列）

**文件**：`src/lmpool/engine/sequence.py`

`Sequence` 携带 token ids、block table、completion token，以及远程前缀等全局池化元信息。

### 4.4 Global Scheduler（全局调度器）

**文件**：`src/lmpool/engine/global_scheduler.py`

`GlobalScheduler` 是跨 GPU 决策层，当前运行在独立控制进程中。它对外暴露两个主入口：

- `route_sequence_meta()`：请求路由
- `plan_rebalance()`：swap 编排

路由策略：

1. 只对完整块做 hash
2. 在全局页表中查找前缀命中
3. 优先选择前缀命中分数最高的 GPU
4. 如果没有命中，则回退到空闲块最多的 GPU
5. 路由后进行乐观 reserve

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
- `block_access_time`：用于 LRU 选择
- `block_hash`：每卡 block hash 快照

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
- 分配 / 释放 block
- 追加 decode token
- 维护本地前缀缓存状态

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
- `pending_swap_in`

这些字段可通过 `multiprocessing.Queue` 跨进程传递。

---

## 5. [实现](./src/lmpool/README_zh.md)

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

## 7. [测试](./tests/README.md)

---

## 8. 评估

`benchmarks/` 下提供共享前缀压测脚本，场景包括：

- `single-gpu`
- `multi-gpu`
- `multi-gpu-kv-routing`
- `multi-gpu-kv-swapping`
- `multi-gpu-lmpool`

报告指标包括：

- throughput
- goodput
- mean / p95 TTFT
- mean / p95 TTPT
- mean / p95 端到端延迟
- GPU 利用率 mean / p95
- GPU 显存利用率 mean / p95
- 前缀命中率

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
