# LMPool QA

## 1. 一个请求进入 LMPool 后经历什么完整过程？

请求首先由 LLMEngine 接收并完成 tokenization，随后构造 Sequence。Sequence 按完整 prompt block 计算链式 prefix hash，partial tail 不参与全局复用。入口侧的 ControlPlaneClient 使用 requester rank $-1$ 请求全局路由，因此候选集合是所有健康 GPU，而不是某个初始 GPU 的 NVLink partner。

这里的 `requester_rank` 表示“从哪张 GPU 的视角发起路由”，不是最终执行 rank。若请求已经位于某个 worker，非负 rank 会让候选集合受该 worker 及其直连 NVLink partner 约束。新请求来自 LLMEngine，还没有落到任何 GPU，因此用哨兵值 $-1$ 表示“无本地 GPU”。Global Scheduler 在这种情况下把所有健康、已发布新鲜状态的 replica 都作为候选，并令它们具有相同的入口拓扑权重。选出目标后，LLMEngine 才把 Sequence 发到对应 data-plane queue。$-1$ 不是一个真实 rank，也不参与 NCCL。

Global Scheduler 从 Global Block Manager 获取每个候选 GPU 的连续 ready-prefix 长度、等待和运行 token、活跃 sequence、free blocks 及 dependency-safe reclaimable capacity。它计算 route cost，并对目标 GPU 建立临时容量 reservation。LLMEngine 将完整 Sequence 放入目标 data-plane process 的队列。

目标 Local Scheduler 接收 Sequence 后，Local Block Manager 对 ready local prefix block 增加引用计数，并为缺失 block 分配物理 ID。若真实 shortage 仍存在，Scheduler 才允许 foreground transfer 进入成本与可执行性检查。Model Runner 执行 prefill，将新 KV 写入本地 cache，再进入逐 token decode。Worker 把 first token、完成 token、运行指标以及版本化 block/load snapshot 返回给 LLMEngine 和 control plane。请求完成后引用被释放；完整 prefix 在 ref-count 归零后继续缓存，直到被复用或 dependency-safe reclamation 回收。

TLDR：LMPool 先用全局元数据决定“请求去哪”，再由目标 worker 决定“本地复用还是分配”；只有真实 shortage 或已获准的预测性 placement 才进入 transfer。KV tensor 始终在 Model Runner 之间传输，control plane 只处理 metadata 和 transaction phase。

## 2. Routing 和 transfer 分别在哪里做决策？

Routing 由 control-plane process 中的 Global Scheduler 决定，输入是 Global Block Manager 保存的 versioned prefix、load 和 capacity snapshot。Transfer 也由 Global Scheduler 规划，但物理 block reservation、gather/send/receive/scatter 和 publish 由两个 data-plane process 执行。

关键区别是 routing 改变 request placement，transfer 改变 KV placement。两者共享元数据和成本视图，但不能互相替代。

## 3. 为什么入口 routing 可以看所有 GPU，而 transfer 只在 NVLink pair 内？

入口请求还没有 initial GPU，因此可以在所有健康 replica 中选择最低预计成本。物理 transfer 必须有明确 source 和 destination，而且当前系统只支持配置或发现出的 direct NVLink pair。全局 routing 不意味着全局任意 GPU 间都能直接搬 KV。

## 4. Prefix match 为什么必须连续？

后续 block 的 attention 依赖之前所有 token。即使某个后部 block hash 单独存在，如果前面的 hash chain 不完整，请求也不能从中间安全恢复。LMPool 因此只统计从第一个完整 block 开始的最长连续 ready chain。

## 5. Route cost 的物理含义是什么？

$Q_g$ 表示 candidate GPU 上 token-equivalent 的等待、pending 和 running work；$M_g$ 是按 block 取整的 missing prefill estimate；$A_gS$ 表示需要 reclaim 的容量压力。Global Scheduler 最小化

$$
C_{\mathrm{route}}(g)=Q_g+w_pM_g+w_eA_gS.
$$

这不是“hit 越高越好”的 reward，而是让 prefix reuse 直接减少预计需要执行的 prefill work。

## 6. Strict prefix-owner routing 为什么不够？

一个热 prefix 的 owner 可能收到大量请求。全部送给 owner 会提高 hit rate，却可能降低 batch parallelism 并增加 queueing。LMPool 允许 bounded spill：只有 owner 的负载明显更高且额外 recomputation 在阈值内时，才绕过 owner。

这里的额外 recomputation 是 spill rank 相比 owner 多执行的 missing-prefill 和 reclaim work，连同两边的 queue work 一起形成 route cost。它不把 transfer cost 直接加进 spill 的 route cost，因为“立即把本次请求 spill 后在目标重算”和“先复制 KV 再服务未来请求”是两个不同动作。Global Scheduler 会单独判断预计复用收益能否覆盖实测 transfer cost；foreground transfer 由真实 block shortage 触发，background transfer 需要热点候选和通过收益门限的放置计划。因此不能把 bounded spill 描述为隐式执行了一次 transfer。

## 7. Foreground transfer 和 background transfer 有什么区别？

Foreground transfer 是 demand-driven。Local Scheduler 在真实 block shortage 时只请求 shortage 数量，不请求整条 sequence 的全部 blocks。

Background transfer 是 placement-driven。它针对预计将来会在另一 GPU 复用的完整 hot chain，在请求真正到达前复制副本。候选仍需通过 cooldown、pair idleness、target capacity 和 benefit gate。

二者使用同一个 transactional data path，但 benefit estimate 不同。Foreground 可以使用折扣后的历史复用；background 最多计算第一次可避免的 cold prefill，因为 destination 第一次 miss 后也会自行预热。

## 8. Copy 和 move 有什么区别？

Copy 在 destination 建立新副本，同时保留 source。它适用于 source block 仍有引用或未来两侧都需要复用的情况。

Move 在 destination publish 成功后，释放 source 上 dependency-safe、unreferenced 的后缀。被引用或作为其他 retained block 祖先的 block 不能 move-release。

## 9. KV block 在 Block Manager 之间传输吗？KV cache 是怎么 version 的？

不直接在 Block Manager 对象之间传输。Local Block Manager 负责物理 block ID、reservation、reference、generation 和 readiness。Model Runner 从 source KV tensor gather 数据，通过 pair-local NCCL 发送到 destination Model Runner，再 scatter 到 destination reserved blocks。Block Manager 随后 publish metadata。

系统没有给整份 KV tensor 维护一个单一版本号，而是使用四层身份与时序信息。第一，链式 prefix hash 标识逻辑内容，相同 hash 表示相同的完整 token-prefix chain。第二，每个物理 block ID 都有 `generation`；同一个 ID 每次重新分配时 generation 加一，transfer plan 同时携带 block ID、hash 和 generation，从而避免 ID 被释放后复用所造成的 ABA 错误。第三，每个 worker 每次上报 block/load snapshot 都递增 `state_version`，control plane 丢弃小于等于当前版本的乱序快照。第四，worker restart 会产生新的 `worker_epoch`，control-plane restart 会产生新的 `control_epoch`；epoch 改变时旧 page table 和旧消息不再可信，必须等待完整新快照后才能重新路由到该 worker。

因此，hash 回答“这是什么 KV”，generation 回答“这个物理槽位还是不是 plan 创建时的那一代”，state version 回答“快照是否比已知状态更新”，epoch 回答“消息是否仍属于当前进程生命周期”。

## 10. KV tensor 如何打包？

对 $B$ 个 block，packed tensor 形状是

$$
[L,2,B,S,H_{kv},D],
$$

依次表示 layer、K/V、block、block token、KV head 和 head dimension。Source 使用 indexed gather 形成连续 tensor，NCCL 只发送一次 payload，destination 再 indexed scatter。Hash、generation 和 block ID 留在控制协议中，不进入 tensor payload。

## 11. 为什么使用 prepare-execute-publish-finalize？

核心是 visibility boundary。Destination receive 完成并不代表 block 已经可被路由复用。Prepare 先 reserve；execute 只移动数据；publish 在 local metadata ready 后将 replica 暴露给 Global Block Manager；finalize 最后决定保留或释放 source。任何 publish 前失败都不会产生可见的半成品副本。

Visibility boundary 是“内部状态从不可被其他请求观察，转变为可被全局路由使用”的提交边界。在 LMPool 中，这条边界是 publish，而不是 NCCL receive 返回。Execute 完成后，destination HBM 中可能已经有正确字节，但 block 仍带有 pending-publish 状态，不会出现在 worker 的 ready-block snapshot 中，也不会进入 Global Block Manager 的可路由 page table。Publish 完成 block registration、设置 `kv_ready`、清除 pending 状态并上报新版本快照。只有随后 control plane 接受该快照，新的 replica 才能被 prefix lookup 命中。这样可以防止请求读到“数据已到但元数据、父链或容量状态尚未提交”的半完成副本。

## 12. 并发读写是否安全？

每个 worker 的 block mutation 在单线程 event loop 中串行执行。Control plane 的状态 mutation 也由单一事件循环串行化。Per-client receive lock 防止多个调用者消费同一 response queue。Snapshot 带 worker epoch 和 version，过期更新会被拒绝。Transfer plan 按 plan ID 和 phase 幂等。

这不等于分布式共识。Launcher failure 会停止服务，worker failure 会丢失该 worker 的物理 cache。

## 13. 当前 transfer cost 如何估计？

当前 paper suite 不再使用固定 `40 ms` transaction prior。它按实际 payload bytes 查询对应物理 NVLink pair 的 `1/2/4/8/16/32/64` block P95 曲线；中间 plan 采用分段线性插值，超出最大点时采用末段斜率。

静态成本为校准得到的 transaction residual P95 加上 `1.2 × data-path P95`；没有 residual profile 时回退为零，而不是手工常数。运行时 source 和 dispatch-to-publish observation 会按 pair 和 size bucket 更新非负 EWMA。Admission 取 profile 与在线观测中的最大值。当前仍缺少 held-out prediction error 和 residual/safety-ratio sensitivity，这是论文明确承认的 limitation。

更具体地说，成本模型先根据 plan 的 block 数计算 payload bytes，再从对应 NVLink pair 的 1/2/4/8/16/32/64-block P95 曲线得到 $T_{\mathrm{data}}$。静态冷启动估计为

$$
T_{\mathrm{static}}=40\text{ ms}+1.2T_{\mathrm{data}}.
$$

一次成功 source operation 会产生 `source observation`，它测量 source gather、P2P 和相关执行路径的实际耗时，并把“实际值减去 profile 数据路径值”的非负 residual 更新到该 pair、该二次幂 size bucket 的 EWMA。一次完整 plan 从 control-plane dispatch 到 destination commit 的耗时会产生 `dispatch-to-publish observation`；它把“完整放置耗时减去静态估计”的非负 residual 更新到另一个 pair-by-bucket EWMA。EWMA 用近期样本平滑瞬时抖动；residual 截断为非负值，避免一次异常快的样本让系统比离线 P95 prior 更激进。

Admission 使用

$$
T_{\mathrm{xfer}}=\max(T_{\mathrm{static}},T_{\mathrm{source}},T_{\mathrm{placement}})
$$

是为了选择最保守的可用估计，而不是把同一段耗时重复相加。随后只有 $T_{\mathrm{save}}\geq\rho T_{\mathrm{xfer}}$ 且 source generation、direct-pair topology、target capacity 等条件仍成立时，plan 才被接受。

“缺少 held-out prediction error”是指尚未用一批运行校准 profile/EWMA，再在完全独立的 plan 上报告 predicted versus observed cost 的 MAE、P95 error 和 underprediction rate。因此，多尺寸数据路径和 transaction residual 都已测量，但完整 serving-cost 预测仍缺少独立验证。

## 14. 为什么不能只用 microbenchmark 带宽估算 transfer？

Microbenchmark 测量的是相对空载的数据路径。Serving 中还有 scheduler coordination、目标容量 reservation、与模型 kernel 的竞争、publish 和 block registration。只使用 payload bytes / bandwidth 会系统性低估短 plan 的完整事务成本。

## 15. 为什么测量 1/2/4/8/16/32/64 blocks 带宽？

这些点按二次幂覆盖小 foreground plan 到大 coalesced background plan。二次幂能够在有限测量次数下覆盖数量级变化，并与运行时 size bucket 对齐。实际 plan 不被限制在这些点上，而是使用相邻点 piecewise-linear interpolation，超出范围时使用边界或末段斜率。

二次幂采样把 1 到 64 blocks 的范围压缩为七个校准点，同时保证 payload 每次翻倍，便于观察固定开销被摊薄的过程。若测量每个整数大小，需要 64 组实验；只测 1/2/4/8/16/32/64 则能以七组实验覆盖同一数量级范围。运行时 EWMA 也把实际 plan 向上归入下一个二次幂 bucket，例如 5--8 blocks 共用 8-block bucket，避免每个整数 plan 都只有很少样本。

实际 latency lookup 与 EWMA bucket 是两件事。若 plan 位于两个测量点之间，例如 6 blocks，数据路径 latency 在 4-block 与 8-block 点之间按 payload bytes 做线性插值。若 plan 小于最小测量点，当前实现使用第一个测量点的 latency，保持保守。若 plan 大于最大测量点，当前最终实现使用最后两个点，即 32 和 64 blocks，计算非负末段斜率并线性外推；只有当这两点的归一化 latency 恰好相同、斜率为零时，才退回使用“最后一点 latency / bytes”的平均斜率。也就是说，最大档之外最终使用的是末段斜率，不是把 latency 固定在 64-block 边界值。

## 16. 4-block 带宽是不是线上固定 batch size？

不是。4 blocks 只是 profile 的一个校准档。Foreground batch 由真实 shortage 决定；background 可以把同一 directed pair 上的多个 chain coalesce。论文不再用单一 4-block bandwidth 代表所有 plan。

## 17. 为什么 block size 是 256，而 vLLM 常见配置更小？

256 是当前 Mini-vLLM 原型和 paper workload 的固定配置，能形成足够大的完整 transfer unit 并降低 metadata 数量，但会增加 tail waste 和碎片化风险，也会改变 transfer amortization。论文没有证明 256 是最优值。16/64/256-token sensitivity 是 future work，不应把当前结果外推到其他 page size。

## 18. Routing 是否会被 transfer 影响？Transfer 是否会被 routing 影响？

（1）会，而且这是设计目标。Destination publish 后，Global Block Manager 的 page table 出现新 replica。Placement lease 随后给 source 和 destination 分配 request quota，使 routing 能利用复制出的 KV。没有 publish 前不可见，没有 lease 时普通 cost-based route 仍可选择副本。

Placement lease 不是 GPU memory lock，也不是 KV ownership 的永久转移。它是 control plane 在一次 benefit-checked background copy 成功后，为某个 prefix leaf 创建的短期路由配额记录。记录包含 source GPU、replica GPU、两侧剩余 request quota、下一次优先选择哪一侧以及所属 NVLink pair。后续匹配该 prefix 的请求在进入普通 route-cost 计算前，先按 source/replica 交替消费这些配额；每次仍重新检查目标上是否存在连续 ready chain 且容量是否可用。配额耗尽后 lease 被删除，routing 回到普通成本模型。

Lease 解决的是“复制完成但请求仍集中在原 owner”问题。Transfer 改变 Global Block Manager 可见的 replica 集合，lease 则把已准入的 reuse quota 分摊到 source 和 destination，使复制出的 KV 真正转化为并行服务能力。

（2）会，Routing 会改变每个 rank 的请求、prefix access、queue pressure、free capacity 和已有 replica 分布，这些状态直接影响后续 transfer 的候选链、source/destination 和预期收益。若 routing 已经把请求送到可复用 owner 且负载可接受，就没有必要 transfer；若出现真实 shortage，或热点候选通过 background placement 的收益门限，transfer 才可能有价值。普通 route hit 和 bounded spill 本身不会无条件触发 transfer。当前论文配置中，foreground transfer 由真实 shortage 触发，background transfer 由热点、容量、冷却和成本门限共同触发，workload 不指定 target rank。

## 19. 为什么 transfer-only cached-token ratio 很高但 throughput 没提高？

在当前 load-skew 中，transfer-only 完成 87 个 block 的复制，但没有 placement-lease route。完整 LMPool 记录到 95 次 lease route，mean TTFT 显著更低；不过它的 mean TPOT 更高，因此减少 cold prefill 不会自动转化为更高 throughput 或更低 mean E2E。

这里的 rank request ratio 是一次 scenario 中“提交请求数最多的 rank / 提交请求数最少的非空 rank”。值为 1 表示请求完全均匀；3.3 表示最忙 rank 收到的请求约为最闲 rank 的 3.3 倍。它衡量 request placement skew，不是 TP、DP 或模型并行度。该指标也不能单独证明负载均衡，因为请求长度可能不同，所以论文同时检查 per-rank tokens、execution time 和 GPU utilization。

## 20. 为什么 routing-only 有时与完整 LMPool 一样好？

如果 routing 已经能把请求送到 prefix owner，并且 owner 没有严重过载，那么 transfer 的增量空间很小。当前 memory-skew 中，即使 foreground move 安全释放了 source block，完整系统也与 routing-only 接近。论文因此把它表述为机制与安全性证据，不主张所有 metric 都显著优于 routing-only。

## 21. 为什么 load-skew workload 没有体现 transfer 优势？

当前 load-skew 使用窄热点集，避免 round robin 过早在所有 worker 上自然复制同一前缀，并确实触发三次 background placement。不过复制出来的 KV 会与 prefill 和 decode 调度竞争：TTFT 与 uncached prefill 降低，但完整系统的 TPOT 可能升高。因此它说明的是可测量 trade-off，而不是“所有 transfer workload 都提高聚合吞吐”。

## 22. 为什么 memory-skew workload 也没有优势？

Memory pressure 只是必要条件之一。还需要 source 存在可安全复制或移动的完整 reusable chain，destination 有空间，并且 future reuse 足够覆盖 transaction cost。Qwen3-1.7B memory-skew 中 cost gate 没有接受 transfer，因此它验证的是 rejection boundary，不是 transfer speedup。

Rejection boundary 是策略从“接受 transfer”切换到“拒绝并本地继续”的条件边界，不是一个单独常数。在当前系统中，它由一组门共同定义：是否存在完整且 generation 仍有效的 source chain，是否有 direct NVLink pair，pair 是否空闲，target 是否有可保留的容量，candidate 是否重复或仍处于 cooldown，以及预计 saved prefill 是否超过安全比率乘以保守 transfer cost。任一门失败，plan 都落在 rejection side。Memory-skew 结果说明“有显存压力”本身没有越过这条边界；它还需要可复用 KV、可行目标和足够未来需求同时成立。

## 23. 当前 load-skew workload 是否使用 oracle？

不使用。workload 只控制 prompt identity、group frequency 和 arrival order，不指定 target rank，也不向 scheduler 提供 exact future request count。Background transfer 根据 observed hotness、load pressure、capacity、cooldown 和正常 benefit gate 决策。在线 demand predictor 仍是 future work，因为当前热点信号不是经过验证的 forecast。

Oracle 指实验向系统提供真实运行中通常无法提前完整知道的信息。当前 workload 不提供这类信息。Lease routing 是 admitted background copy 成功后的短期配额路由：它在将匹配请求送向 replica 前仍会检查 ready prefix 和 capacity，并在 quota 耗尽后失效。

## 24. 与 vLLM 或 SGLang 相比，LMPool 的差异是什么？

vLLM PagedAttention 和 SGLang RadixAttention 主要解决单个 engine 内的 block allocation 和 prefix reuse。LMPool 研究 data-parallel replica 之间的 admission-time placement：全局路由选择在哪个独立 allocator 上执行，并在 direct NVLink pair 上以成本门控方式复制或移动完整 prefix chain。

当前论文没有生产级 vLLM/SGLang end-to-end baseline，因此不能声称整体性能优于它们。贡献是机制与受控证据，而不是生产系统 SOTA。

## 25. 与 Mooncake、LMCache、PegaFlow 的差异是什么？

这些系统主要提供跨 instance、跨节点或跨存储层的 KV visibility 和持久容量，常使用 RDMA、host DRAM 或 SSD。LMPool 聚焦单机 scale-up 场景中的 peer HBM cache，并把 request route、load/capacity 和 measured pair-local transfer cost 放在同一 admission decision 中。

二者可以互补：远端或持久 cache 解决 capacity 和 cross-node reuse，LMPool 的原则可用于决定优先本地路由、NVLink peer transfer，还是回退到更远层级。

Admission decision 是在昂贵操作真正开始前，基于当前状态决定“允许还是拒绝”的门控过程。对 request routing，它判断哪个 rank 具有足够有效容量并预留该容量；对 transfer，它判断 source identity、topology、target space、pair availability 和 benefit/cost 是否全部满足，并据此 admit、defer 或 reject plan。将二者放在同一控制面并不意味着用一个公式完成所有动作，而是让 route placement 和 KV placement 读取同一份 versioned locality、load、capacity 与 cost 状态，避免一个模块制造另一个模块无法兑现的决策。

## 26. 与 Aqua 的差异是什么？

Aqua 主要针对 burst memory pressure 和 active inference state paging，利用其他 GPU 的空闲 HBM 降低 preemption cost。LMPool 使用 content-addressed complete prefix chain，目标是 future prefix reuse，并将 copy placement 与 request routing 绑定。二者的 trigger 和 objective 不同。

## 27. 为什么只用 Qwen3-0.6B 和 1.7B？

它们能在六张 24 GiB RTX 3090 上以多个 data-parallel replica 完整运行，并允许重复实验。两者共享 KV geometry，因此 transfer payload 相同，主要差异是 compute intensity。这个选择适合机制隔离，但不足以证明大模型普适性。更大模型、GQA/MLA 差异和不同 KV bytes/token 属于后续实验。

在当前结果中，Qwen3-1.7B 更能体现“避免 prefill”的价值。5x routing workload 中，0.6B/1.7B 的 throughput 分别提升 11.8%/21.3%，TTFT 分别降低 18.6%/25.2%。原因是两者 KV geometry 相同、同一 plan 的 transfer bytes 基本相同，但 1.7B 每个 uncached prefill token 的计算更贵，因此 saved compute / transfer cost 比例更高。

这不表示模型越大，LMPool 一定越好。更大的模型可能改变 KV heads、head dimension、dtype、block payload、可用 HBM、模型并行方式以及通信与 kernel 干扰。若 KV bytes 增长快于可避免的计算，或模型必须使用 tensor parallel 而不再是独立 DP replica，transfer 可能更贵。更可靠的判断变量是预计节省的 prefill 时间相对于完整 transfer、queueing 和 interference 成本的比值，而不是参数量本身。

## 28. 为什么没有生产 trace 和生产 baseline？

当前工作优先构造可控 workload，用于区分 routing、transfer、memory pressure 和 naturally replicated locality。代价是 external validity 不足。不是回避，而是明确论文结论是 prototype mechanism study；下一步是 public multi-turn trace、online forecast 和相同 KV budget 下的 production engine baseline。

## 29. GPU utilization 越高越好吗？

不是单独越高越好。高 utilization 只有在 throughput、latency 和 goodput 同时改善时才说明资源被有效利用。当前 suite 中，如果 utilization 提高但 latency 恶化，应解释为拥塞而不是收益；论文不再把 utilization 单独作为性能提升证据。

GPU memory utilization 也不是越高越好。过低说明容量闲置，过高会增加 admission failure、reclamation 和 preemption 风险。论文固定 per-worker KV budget 来保证 baseline 公平。

## 30. Prefix hit 指标为什么有 route、owner、local 和 cached tokens？

Route hit 表示 control plane 在路由时发现至少一个匹配 prefix。Owner hit 表示最终目标属于当时记录的 prefix owner。Local request hit 表示 data plane 初始 prefill 实际复用了至少一个 local ready block。Cached-token ratio 是复用 prompt token 数占全部 prompt token 数的比例。

前三者是 request-level binary rate，cached-token ratio 是 token-weighted work reduction。一个请求只命中一个 block 和命中七个 blocks 都算一次 request hit，因此论文主要使用 cached-token ratio 与 uncached tokens 解释计算节省。

## 31. 上一个 scenario 的 KV cache 会影响下一个 scenario 吗？

不会。每个 scenario 和 trial 都创建新的 worker processes、Model Runner 和 Local Block Manager。进程退出后 GPU KV cache 随之释放。结果之间不会共享运行时 prefix state，但模型权重可能继续存在于操作系统或 Hugging Face 文件缓存中；计时从 worker 初始化和 warm-up 完成后开始。

这里的 model warm-up 是正式 serving 计时前运行少量合成 prefill/decode，使 CUDA context、模型权重、KV cache allocation、kernel path 和 NCCL communicator 完成初始化，避免把一次性启动开销计入每请求 latency。它不同于 workload warm-up phase。当前 load-skew 与 memory-skew 的 workload warm-up 用于构造后续可能被 reuse 或 transfer 的 prefix chain，属于场景构造，并与 reuse-phase 指标分开报告。

## 32. Control plane 是单点故障吗？

当前是。LLMEngine 可以监控并重启 control-plane process，worker snapshot 可重建 global metadata，但这不是 consensus 或 high availability。Launcher failure 会停止服务。生产版本需要 replicated metadata、leader election 和 launcher failover。

## 33. 系统最大的工程困难是什么？

最困难的是同时保证“数据已经传完”和“元数据允许被路由”之间的一致性。早期问题包括 destination 提前可见、source block 仍被引用却被释放、重复 phase 导致 double free，以及 P2P ordering 或 rendezvous 造成 hang。解决方法不是增加一个全局锁，而是定义单一 ownership、generation/version check、prepare--publish visibility boundary、idempotent plan phase 和 abort rollback，并用故障与并发测试覆盖。

P2P ordering 指 source 与 destination 必须以一致的 pair、tensor shape、direction 和调用顺序进入匹配的 NCCL send/recv。若一个 rank 先等待 plan A，而另一个 rank 先执行 plan B，或者一侧发送而另一侧没有进入对应 receive，双方会互相等待。Rendezvous 指所有参与 process-group 初始化的 rank 必须使用同一 store/init method、world size 和唯一可用 endpoint；端口被旧进程占用会触发 `EADDRINUSE`，缺 rank、错 world size 或错 endpoint 则可能一直等待。

一个 Python 全局锁只能序列化拿到同一锁的线程，不能跨独立 worker process 自动保证 NCCL 调用匹配，也不能证明 block 在等待期间没有被释放并复用；若把所有 route、allocation 和 transfer 都串行化，还会破坏并发和吞吐。系统因此使用更窄的协议约束：Local Block Manager 是物理块的单一 owner；hash+generation 防止 ABA；worker snapshot version 和 epoch 拒绝旧状态；prepare 锁 source 并 reserve target；publish 是唯一 visibility boundary；同一 `(plan_id, phase)` 重放返回缓存结果而不重复 mutation；任一阶段失败都 abort，保留 source 并释放 target reservation。测试分别注入重复 phase、stale generation、失败 publish、进程重启和并发 response consumption，而不是只测试正常路径。

## 34. 研发这个项目最大的收获是什么？

第一，优化数据通路之前必须先减少不必要的数据移动，因此 routing 和 transfer 不是两个独立 feature，而是一个共同成本决策。第二，cache hit rate 不是最终目标，必须同时观察 uncached work、queueing、rank distribution、throughput 和 latency。第三，分布式 GPU 系统的性能机制必须建立在明确的数据 ownership 和 publication semantics 上，否则一次看似成功的 NCCL receive 仍可能产生错误的全局状态。

Uncached work 是请求在目标 worker 上没有可复用的连续 ready KV 时仍需执行的 prompt prefill，包括对应 token、block allocation 和模型计算；论文使用 uncached prompt tokens 与 prefill time 来近似这部分实际工作。Publication semantics 定义“谁在什么条件下可以看见一项状态变化”：destination 数据只有在 block 注册、`kv_ready`、父链、generation 和新 snapshot 都提交后才可被全局 route；source 只有在 destination publish 后才可能 finalize/reclaim。这些规则把“字节传输成功”和“系统状态可安全使用”分开。

更广泛的收获是建立以可测量代价分析 tradeoff 的能力：locality 与负载均衡、transfer 与 recomputation、较大 block 的批量效率与内部碎片、保守 admission 的安全性与 missed opportunity、强一致性保护与控制面开销、集中式元数据的简单性与高可用性，以及热点信号的机制隔离能力与生产真实性。真正的系统优化不是让单个指标最大，而是在明确 correctness boundary 后，用 workload、成本模型、消融和 negative result 判断哪一侧更值得。

## 35. 如果只能补一个实验，应该补什么？

优先补成本模型与 forecast 的联合有效性实验：在一个公开多轮 trace 上实现在线 demand estimate，同时记录每个 candidate plan 的 predicted cost、observed cost 和后续实际 reuse。报告 prediction precision/recall、invalid-copy rate、cost MAE/P95 error，以及 no-transfer、always-transfer、profile-only、profile+EWMA 的 throughput/TTFT。该实验直接回答系统能否在未知流量中正确发现并选择有利 transfer。

Forecast 是对未来 KV reuse demand 的估计，至少应回答四个问题：哪个 prefix chain 会再次出现、预计出现多少次、请求更可能落到哪个 rank/pair、以及距离首次 reuse 还有多少 lead time。Background transfer 只有在 forecast 足够早时才能在请求到达前完成，足够准时才能避免复制无人使用的 KV。当前 suite 不提供 exact future-prefix counts；生产版本需要从会话 ID、历史 prefix access、route hits、arrival pattern 或应用提示中在线估计，并报告 precision、recall、lead time、invalid-copy rate 以及错误 forecast 对 throughput/latency 的影响。
