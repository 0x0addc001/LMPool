# LMPool CV & QA

## 简历条目

### 一行版

**LMPool | 多 GPU LLM 推理 KV Cache 调度系统**：设计基于 KV 局部性的全局路由和基于 NVLink 的事务式 KV copy/move；在 6 张 RTX 3090、Qwen3-0.6B/1.7B 上，5x 共享前缀下吞吐分别提升 11.8%/21.3%，平均 TTFT 分别降低 18.6%/25.2%。

### 三条版

- 使用 Python、PyTorch、NCCL 和多进程 control/data plane 实现多 GPU KV Cache 协调系统；全局元数据与本地物理 block 所有权分离，通过版本、reservation、prepare--publish 和 rollback 保证 incomplete replica 不参与路由。
- 实现 locality- 和 load-aware KV routing：联合连续 prefix match、token-equivalent queue work、容量与 reclaim pressure 评分。在 6 张 RTX 3090 的内部消融中，5x 共享前缀下，Qwen3-0.6B/1.7B 吞吐提升 11.8%/21.3%，平均 TTFT 降低 18.6%/25.2%，未缓存 prefill token 降低 59.9%/61.0%。
- 实现 pair- 和 payload-aware NVLink KV transfer：打包全 layer K/V tensor，采用 NCCL P2P copy/move、P95 profile 和在线 residual EWMA 做准入；验证了 1--64 block 数据路径、background placement 后的 87 个 copied block 和 95 次 lease route，以及 0.6B memory skew 下安全释放 35 个 source block。

## STAR 整理

### 场景（Situation）

数据并行 LLM 服务中，每张 GPU 都有独立的 KV cache。Round-robin 会把具有已缓存共享前缀的请求分配到其他 GPU，导致重复 prefill；但把所有请求都送到 prefix owner 又会形成热点和排队。直接搬运 KV 也并非总是有利，因为 transfer 会占用链路、目标显存和同步开销。

### 任务（Task）

我的目标是实现一个多 GPU KV cache 协调原型：优先将请求发给已有可用 KV prefix 且负载可接受的 GPU；当 KV placement 与需求或容量不再匹配时，仅在收益覆盖成本时通过 NVLink 复制或迁移完整 KV block chain。同时需要保证多进程下 metadata 与物理 block 状态的一致性。

### 行动（Action）

我将系统拆为 main process、独立 control plane 和每 GPU 一个 data plane。Global Scheduler 根据连续 ready-prefix match、排队 token、decode work、有效容量和 reclaim pressure 选择路由目标，而不是单纯最大化 hit rate。对于 transfer，我实现了 source validation、target reservation、NCCL packed P2P transfer、publication 和 finalization 四阶段事务；destination 在 publication 前不可被路由。成本模型按 NVLink pair 和 payload size 使用离线 P95 profile，并以 source/placement residual 的非负 EWMA 保守更新。为避免只证明功能，我设计了长共享前缀 routing、load skew 和 memory skew 三类受控 workload，并为 block manager、scheduler、control plane 和 transfer 添加单元与多进程测试。

### 结果（Result）

在 6 张 RTX 3090 的五次内部消融中，5x 共享前缀 routing 在 Qwen3-0.6B/1.7B 上分别带来 11.8%/21.3% 吞吐提升和 18.6%/25.2% 平均 TTFT 降低，未缓存 prefill 约降低 60%。Transfer 路径完成了逐字节验证，load skew 中执行三次 background placement、复制 87 个 block 并通过 lease 路由 95 个后续请求；memory skew 中 0.6B 安全释放 35 个 source block。我也保留了负结论：已完成的 transfer 本身并不保证平均 E2E 或吞吐提升，这使系统的性能主张限定在证据支持的范围内。

## 面试逐字稿

### 90 秒项目介绍

我做的项目叫 LMPool，解决的是多 GPU LLM 服务中的 KV cache placement 问题。数据并行部署时，每张 GPU 的 KV cache 是独立的。Round-robin 很均衡，但可能错过其他 GPU 上已缓存的共享前缀，造成重复 prefill。相反，严格把请求送到 prefix owner 虽然命中率高，却可能把热点集中到一张卡上。

我的做法是把问题拆成两部分。第一部分是 locality-aware routing：路由器不只看 prefix hit，还看连续可复用 prefix、队列中的 token、decode work、容量和回收压力，选择预计完成成本最低的 GPU。第二部分是 cache fluidity：当 KV 的位置不再匹配需求或容量时，系统才考虑通过 direct NVLink copy 或 move 完整 KV block chain，并用实测链路 profile 和在线观测来判断 transfer 是否值得。

工程上，我将 control plane 与每张 GPU 的 data plane 分开。Control plane 管理全局 page table 和 transfer plan；Local Block Manager 管理物理 block。Transfer 是 prepare、execute、publish、finalize 的事务。目标端收到 KV 后并不能立刻被路由，只有发布完整 hash chain 和 ready state 后才可见。

在 6 张 RTX 3090 上，5x 共享前缀的内部实验中，routing 在 Qwen3-0.6B 和 1.7B 上分别提升 11.8% 和 21.3% 吞吐，并分别降低 18.6% 和 25.2% 的平均 TTFT。Transfer 路径验证了 background copy、placement lease 和 source release，但我没有把它写成普遍 throughput 提升，因为实验显示 transfer 完成并不必然改善 E2E。这是我在这个项目中最重视的工程原则：让设计、实验和结论严格对齐。

### 3 分钟 STAR 深挖

项目开始时，我观察到 prefix caching 在单 GPU 内有效，但在多 GPU data parallel 环境里常常失效：缓存和请求调度是两个独立决策。一个请求即使有共享前缀，也可能被 round-robin 分到没有该 prefix 的 GPU；如果简单改成 owner routing，又会产生热点。我的任务是构建一个可以同时处理 locality、负载和 KV placement 的系统，而不是只提高一个 cache-hit 指标。

我先定义了 ownership boundary。Global Block Manager 只保存版本化的全局 metadata；每个 Local Block Manager 独占物理 allocation、reference count 和 KV tensor。这样可以避免跨进程共享 Python block manager 带来的并发问题。随后，我实现了 route cost：它把 token-equivalent queued work、按 block 对齐的 missing prefill work 和 reclaim pressure 加起来。连续 prefix match 会直接减少 missing prefill term，因此 locality 转化为计算节省，而不是额外的 hit-rate bonus。对于负载过高的 owner，系统允许有界 spill，但要求额外 recomputation cost 在阈值内。

Transfer 部分是最难的。直接调用 send/recv 不足以保证状态正确，因为 destination 可能收到数据但还没有完整 metadata。我的解决方案是四阶段 transaction：prepare 时锁定 source chain 并 reserve destination block；execute 用 pair-local NCCL communicator 传一次打包的全 layer K/V tensor；publish 时 destination 写入完整 hash chain 和 ready snapshot；finalize 时 copy 保留 source，move 才释放 dependency-safe、unreferenced suffix。任何失败都会释放 destination reservation 而保留 source。准入上，我用每个 NVLink pair 的 1--64 block P95 profile，加上 transaction residual 和非负 EWMA，只有预期节省的 prefill time 超过保守 transfer cost 才执行。

结果上，routing 是最稳定的性能收益：5x 共享前缀下，两种 Qwen3 模型吞吐分别提升 11.8% 和 21.3%，TTFT 分别降低 18.6% 和 25.2%。Transfer 的机制也确实执行：load skew 中有 87 个 copied block 和 95 个 lease-routed request，memory skew 的 0.6B 场景安全释放了 35 个 source block。但我发现 completed transfer 不等于端到端性能一定更好，decode contention 可能抵消 prefill 节省。因此我最后把 transfer 的结论写成机制正确性和适用边界，而没有过度声称。这次经历让我学会了把性能优化拆成可验证假设，并用负结果收紧系统主张。

## 高频追问与回答

### 为什么不只做 routing？

Routing 只能利用已经在目标 GPU 上存在的 KV。当请求负载或 KV capacity 与现有 placement 不匹配时，单纯 routing 可能把请求集中到 owner，或无法释放 source capacity。Transfer 提供的是数据流动性，但它必须通过成本和可执行性检查，不能替代 routing。

### 为什么不总是 transfer？

Transfer 有 gather、NCCL、scatter、reservation、同步和显存干扰成本。如果 future reuse 不足，重新 prefill 反而更便宜。系统因此以 pair-specific P95 profile 和在线 residual 为基础，只接受节省 prefill 成本覆盖 transfer 成本的 plan。

### 如何避免并发状态错误？

物理 block 只由对应 Local Block Manager 修改；control plane 事件循环串行更新全局 metadata。Transfer 使用 generation/version check、reservation、prepare--publish visibility boundary、idempotent phase 和 abort rollback。未发布 replica 永远不会被路由。

### 最大的困难是什么？

最大的困难不是把 KV tensor 从一张卡复制到另一张卡，而是判断什么时候复制值得，以及如何证明它没有破坏 serving correctness。我的解决方法是把数据路径、admission 和 publication 分离：先用 microbenchmark 校准 payload cost，再用 transaction protocol 保证 visibility，最后通过受控 workload 分别验证 routing、background placement 和 source release。

### 你会如何继续这个项目？

下一步会用公开或匿名化多轮 trace 评估 online demand prediction，并报告 precision、recall、lead time 和 invalid-copy rate；在相同模型、KV budget 和 arrival process 下与至少一个生产 serving engine 比较；同时对 block size、模型 KV geometry、GPU 数及成本模型的 profile、EWMA 和 safety ratio 做 ablation。
