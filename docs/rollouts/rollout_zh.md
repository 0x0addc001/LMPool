# Rollout

本文是 [rollout.md](rollout.md) 的中文版本。它保留当前有效决策的完整事实链，并按主题归纳早期的探索性迭代。代码路径、配置项、消息字段、指标名和实验工件名称保持英文，以便与实现和结果文件逐项对应。英文文档仍是历史逐条记录的权威来源。

每项决策按同一结构记录：决策需求、决策计划、决策实现和决策结果。

## 早期决策概览（2026-07-12 至 2026-07-25）

### 术语、基准与可观测性

项目统一以 `transfer` 指代跨 GPU 的 KV 移动，并保留 `swap_in`、`swap_out` 等旧内部 API 以维持兼容性。基准入口被整理为 `benchmark_e2e.py`、`benchmark_kv_transfer.py` 和 `benchmark_kv_routing.py`，原有脚本保留为兼容实现。端到端结果随后补充了 P90/P95 延迟、每 rank 请求与 token 分布、GPU 利用率、内存利用率、路由命中、owner 命中、本地命中、前台与后台转移状态等指标；图表也按 workload 命名并采用统一的论文配色。

### 路由从前缀命中扩展到负载约束

早期实验表明，仅按共享前缀选择 owner 会将请求集中到少数 rank。为此，`GlobalBlockManager` 开始接收 waiting、running 和 pending 序列状态；`GlobalScheduler` 在最长连续前缀收益之外，估计队列压力、解码工作、有效剩余容量和乐观的待准入负载。路由缓存仅在目标仍持有前缀、容量足够且负载未明显高于可选目标时复用。后续修正还保证了 routed prefix promise 能穿过本地 admission，从而避免“全局已选中前缀 owner、局部又改送其他 rank”的语义断裂。

### KV 容量、前缀链与事务安全

KV block budget 被改为真实的容量上限，而非仅影响基准参数。Block manager 维护完整前缀链和最长连续前缀，以增量 KV 容量而不是仅按 token 数进行 admission。解码跨 page 边界前会检查容量，完成请求的前缀 block 在 LRU 回收前仍可复用。前台 move 只释放依赖安全的叶子后缀；copy 保留源 block，适用于后台复制，不被当作前台腾挪成功。

### 前台与后台转移分离

系统将 request-critical 的 foreground move 与不阻塞当前请求的 background copy 分开统计和决策。前台路径只在存在真实容量短缺、可执行计划、直接 NVLink 邻居、有效 source generation、空闲且有容量的目标时执行；否则回退到本地回收。后台路径根据内容热点、预测或观测到的复用、负载差和冷却时间挑选候选，并通过 placement lease 将后续请求导向已发布副本。该分离避免了 copy 在未释放源空间时被误记为容量缓解。

### 成本模型从固定延迟演进为按 payload 校准

直接 NVLink microbenchmark 从 1、2、4、8、16、32、64 个 block 测量各 GPU pair 的 P95 数据通路延迟和有效带宽。`GlobalScheduler` 根据计划 block 数计算 payload，并对相邻档位做分段线性插值。静态数据通路先验结合 transaction residual；source operation 与 dispatch-to-publish 的观测按 pair 和 size bucket 更新非负 EWMA。admission 使用可用静态与在线估计中的保守值，而不是固定 `40 ms` 常数。该模型还保留 topology、容量、generation、reservation 和 lease 等可执行性条件。

### 并发与故障处理

控制面与数据面明确了 source ownership、generation/version check、prepare--publish visibility boundary、幂等 plan phase 和 abort rollback。源 block 直到 move commit 后才释放；copy 永远不释放源 block；abort 只释放目标预留。并发测试覆盖了重复消息、过期 generation、reserve/release、busy target、失败回滚和 worker 异常。每次多进程 benchmark 使用独立 rendezvous store，避免连续 trial 的端口冲突。

### 工作负载与证据边界

基准拆分为 routing、load-skew、memory-skew 和 NVLink transfer profile。routing 使用多组长共享前缀以验证避免重复 prefill；load-skew 使用窄热点集和较长 reuse burst 以验证后台 copy 与 lease；memory-skew 制造源端容量压力以验证安全 move 与 source-block release。端到端主张只建立在可重复的长前缀 routing 结果上。转移的完成、copy 数和 source release 是机制证据，不被表述为普适的吞吐或 E2E 优势。

## 2026-07-27：基于主动 Move 的 Memory-Skew 工作负载

### 决策需求

此前的 memory-skew trace 要么在请求已遇到容量压力后才尝试 foreground transfer，要么使用保留源 block 的 copy-style background placement。前者位于请求关键路径，后者不释放源空间，二者都不能证明 capacity offload。

### 决策计划

构造不包含 rank hint 的四阶段 trace：建立可复用的长会话前缀，用无关的 anchor-only 流量制造源端压力，在排空屏障期间移动安全的会话后缀，然后重放会话。控制面必须仅依据拓扑、容量和 worker 空闲状态选择 NVLink peer。

### 决策实现

新增 `--memory-skew-proactive-move` 与 `--background-transfer-mode move`。在 move mode 下，global block manager 计算依赖安全的叶子后缀；转移发送目标注册所需的完整前缀链；源端仅在目标发布后释放该后缀。基准在 warm-up barrier 发布未来需求，只在压力排空后 flush proactive move，并在 memory-skew 场景禁用 foreground rebalance。独立脚本以单候选、64-block transfer budget 启用该模式。

### 决策结果

该工作负载将主动释放源容量与请求触发的转移分开测量。有效运行必须报告已释放的 source block，并让预测复用经 committed target lease 路由；请求内容和 workload 均不选择 source 或 target rank。

## 2026-07-27：为 Memory-Skew Offload 使用内容热点复用

### 决策需求

首个 proactive move 运行完成了一个计划并释放源 block，但只有五个 reuse request 使用 placement lease。该 trace 验证了正确性，却没有足够受影响请求来测量稳定的服务收益。

### 决策计划

将压力和随后复用集中在同一内容级 session group。这对应于一类热点会话：短共享 anchor 制造 cache 压力，长 session suffix 仍然具有价值。trace 不选择 worker rank，常规路由确定 source owner，控制面确定 NVLink target。

### 决策实现

新增 `--memory-skew-reuse-hot-groups` 和 `--memory-skew-reuse-hot-share`。独立 memory-skew 脚本默认使用一个 hot pressure anchor 和一个 hot reuse session group，二者均为全量 share。脚本还提高 load-bypass 阈值，使路由在压力阶段将 hot anchor 保留于 cache owner，从而形成 offloading 所要缓解的 locality-induced capacity skew。

### 决策结果

一次被接受的 move 可通过 target lease 服务大多数 reuse request，而不是只服务六分之一。该工作负载仍不依赖 rank；只有同时报告非零 released block 且相对于 routing-only 在 reuse phase 有实质改善时，才可作为 offload 证据。

## 2026-07-27：使 Suite 的 Memory-Skew 与主动 Move 语义一致

### 决策需求

preflight 与 paper script 仍调用已淘汰的 foreground-only memory-skew trace，并禁用了 background placement。其 acceptance condition 期待 foreground rebalance，因此没有验证独立 move benchmark 所使用的实现。

### 决策计划

恢复已完成 move 的独立 workload 分布，并让 preflight 与 paper suite 调用同一 proactive move trace。将其作为机制 gate，而不是端到端性能优越性的主张。

### 决策实现

独立默认值恢复为两个 pressure-hot group、80\% share、uniform reuse 和常规 route-load bypass 阈值。两个 suite 均使用六组、12 个 warm-up request、30 个 pressure request、零 trigger request、128-block budget 和 move-mode background placement。preflight gate 只要求已完成的 LMPool move、已释放的 source block 和至少一个 lease route。

### 决策结果

suite 不会再悄然执行另一套 foreground-only workload。通过的 memory-skew 结果证明 proactive move transaction 与后续 routing path，但在没有独立重复证据时，不得表述为吞吐优势。

## 2026-07-27：移除冗余的 Session Warm-Up Replica

### 决策需求

使用六个 session group 和 12 个 warm-up request 时，proactive move trace 存在不确定性。它在压力前创建了重复 session chain，导致部分 NVLink target 无法接收完整 chain，并产生 `no_target_space` 拒绝。

### 决策计划

在 warm-up 阶段仅为每个 session group 建立一次。这样既保留由内容决定的正常 owner，也避免冗余 target residency；pressure 与 reuse phase 保持 rank-agnostic。

### 决策实现

将 standalone、preflight 和 paper-suite 中 memory-skew 的 warm-up 长度从 12 改为 6。没有增加 target selection、route hint 或拓扑特定的 workload 内容。

### 决策结果

已验证运行完成五个 proactive move，释放 90 个 source block，并经 placement lease 路由全部 30 个 reuse request。该配置成为 suite 的 memory-skew 机制验证默认值。

## 2026-07-27：定义 Load-Skew 的 Reuse 边界

### 决策需求

preflight load-skew 在分配 phase label 时出现 `NameError: trigger_end`。load-skew 只有 warm-up 和 reuse phase，但共享提交路径引用了 memory-skew 专用的 trigger boundary。

### 决策计划

在 pressure boundary 使用空区间表示不存在的 trigger phase，使通用 phase labeling 无需增加专门 dispatch path。

### 决策实现

在 load-skew phase setup 中设置 `trigger_end = pressure_end`。现有 label logic 会跳过该空 trigger interval，并将所有 post-warm-up request 标为 reuse。

### 决策结果

load-skew 不再在 worker 执行前失败，memory-skew 仍保留显式的可选 trigger phase。

## 2026-07-27：使用窄热点集验证 Load-Skew Replica

### 决策需求

preflight load-skew 使用 24 个 hot prefix 和 48 个 warm-up request，重复前缀占满潜在 replica target。LMPool 只完成一个 background copy 和一个 lease route，因此 trace 衡量了路由，却没有测试基于 replica 的 load relief。

### 决策计划

使用小型 content-hot set，每个 prefix 仅 warm-up 一次，并使用长 reuse burst。这样 NVLink peer 能保留 replica 容量，每次被接受的 copy 也有足够后续 request 摊销成本。workload 仍只选择 prompt identity，控制面选择全部 owner 与 target。

### 决策实现

新增 `benchmarks/run_load_skew.sh`。默认 trace 使用三个 hot prefix group、三个 warm-up request、189 个 reuse request、64 次 prompt repeat、192-block budget 和 96-request submit window。background placement 每个 plan 使用一个 32-block candidate，forecast threshold 为一，最多 64 个预期 reuse。

### 决策结果

独立运行会在该配置进入 preflight 或 paper suite 前，暴露 copy count、完成的 background placement、lease route 以及 reuse-phase throughput 与 latency。

## 2026-07-27：提升已验证的窄热点集 Load-Skew Trace

### 决策需求

独立窄热点集 load-skew 运行完成两个 background placement，复制 58 个 block，并经 placement lease 路由 63 个 request。preflight 与 paper suite 仍使用旧的广热点集 trace，后者 target capacity 不足，不能稳定执行 replica placement。

### 决策计划

令 suite workload 与已验证 trace 完全相同，同时保留特定模型的 SLA 值和 suite repetition count。resume check 必须包含全部 workload-shaping 参数，避免复用旧广热点集 artifact。

### 决策实现

更新 `run_preflight_suite.sh` 与 `run_paper_suite.sh`：使用三个 prefix group、三个 warm-up request、三个 full-share hot group、192 个总 request、64 次 prompt repeat 和 96-request submission window。两个脚本均使用一个 32-block copy candidate、一请求 hot threshold、零 minimum load-skew gate 和 64-reuse forecast；artifact check 同时验证这些参数。

### 决策结果

preflight 与 paper run 现在测量独立 workload 已验证的 background-copy 与 placement-lease path，而不是另一条 capacity 不同的广热点集 trace。

## 2026-07-28：将文档与当前 Paper Suite 同步

### 决策需求

paper、README、slide、script、runbook、report 与 oral Q&A 仍混合当前实现细节、已淘汰 session-handoff 结果和旧的固定 `40 ms` transfer prior，使文档主张强于当前 `20260727T231622Z` 数据支持的范围。

### 决策计划

使用最新的五 trial paper suite 作为工作负载、图表和结论的唯一来源。将长前缀 routing 作为主要端到端性能结果；除非 load-skew 和 memory-skew 展示可重复 serving gain，否则只将其作为 transfer mechanism 与 boundary evidence。以可复现的聚合图替换 slide table。

### 决策实现

新增 `docs/paper/figures/generate_suite_results.py`，读取当前 suite JSON 并生成 routing、skew 与 NVLink-profile 的 PNG/PDF 图。两个 README、paper、paper README、10/15 分钟 script、runbook、report 和 Q&A 使用一致的 workload：1x/3x/5x long-prefix routing、three-group load skew、six-group memory skew。移除 session-handoff performance claim，用 calibrated residual/profile model 替代固定延迟描述，并在文字与图中明确 transfer trade-off。

### 决策结果

文档记录现在区分稳定的 routing improvement 与已验证的 transfer execution。5x prefix length 下，routing 在两种模型上将 uncached prefill 降低约 60\%，并改善 throughput 与 tail latency。load skew 验证三个 background placement、87 个 copied block 和 95 条 lease route；memory skew 在 0.6B 上验证安全的 source-block release。二者均不被表述为普适的 mean-E2E 或 throughput improvement。

## 2026-07-28：使 Paper Figure 自包含数值

### 决策需求

首批同步 suite figure 含有 error bar，但读者必须查看 JSON 或正文才能得到精确数值；slide 也过度依赖 table，趋势和不确定性不够直观。

### 决策计划

保留与证据匹配的标准学术图表类型：policy comparison 使用分面 grouped bar，transfer calibration 使用 pair-wise line chart。为每个 bar 加上 numerical mean label，保留 95\% confidence interval，并标注所有 1--64 block bandwidth sample。

### 决策实现

扩展 `generate_suite_results.py`，增加 error-aware bar annotation 和 point annotation。为 paper、slide、README asset 和 report 重新生成 routing、skew 和 transfer-profile 图，沿用共享的 Google-style policy palette：round robin 为灰色、routing 为蓝色、transfer-only 为绿色、LMPool 为黄色。

### 决策结果

每幅图无需另查 table 即可解释：bar height 给出精确 five-trial mean，whisker 给出 95\% confidence interval，calibrated NVLink point 给出实测 effective bandwidth。每个 facet 的指标仍处于可读的单一尺度，因此未使用对数或 broken axis。

## 2026-07-28：将候选图生成与文档发布分离

### 决策需求

图表设计可能需要视觉审阅，然后才应被 paper、README、slide 和 report 引用。若每次探索性 render 都直接写入文档 asset path，审阅边界就不清晰。

### 决策计划

为 suite plotting tool 增加 preview-only output mode。候选图写入归档实验 artifact 附近，而默认 mode 继续执行既有的同步发布行为。

### 决策实现

为 `generate_suite_results.py` 增加 `--preview-dir`。提供该参数时，工具只写入 PNG/PDF candidate figure，并跳过 paper、slide、README 和 report asset copy。当前 candidate 由 `20260727T231622Z` suite 生成。

### 决策结果

图表审阅现在显式且非破坏性。当前 candidate set 含有带 numerical label 与 confidence interval 的 routing、skew 和 transfer-profile chart，可在批准前独立检查。

## 2026-07-28：为 Figure Legend 预留空白区域

### 决策需求

candidate legend 与 bar label 和绘制数据重叠，降低 numerical annotation 与 policy mapping 的可读性。

### 决策计划

在 figure title 与 faceted axis 之间预留专用水平区域，将 legend 居中放置于该区域，而非嵌入 subplot 内部。

### 决策实现

将 routing 与 skew figure 从 constrained layout 改为显式 subplot margin。top margin 现在容纳 title 与 external legend，chart axis 位于 legend 下方。重新生成 candidate PNG/PDF，并审阅全部 paper caption：sentence caption 使用 sentence case 并加句号；figure/table 的 noun phrase 使用 Title Case 且不加句号。

### 决策结果

legend 不再覆盖 bar、error bar 或 numerical label。在审阅批准前，candidate figure 仍与文档发布分离。

## 2026-07-29：发布批准的 Figure 并加强 Evaluation 解释

### 决策需求

已审阅的 candidate figure 可以发布。evaluation 还需要明确解释某项 policy 为什么改善一个 metric 却不改善另一个，特别是 short-prefix routing、background placement 和 foreground source release。

### 决策计划

将已批准的 numerical figure 发布到所有文档 asset path。只有在 five-trial data 支持直接比较与机制解释时才扩充 evaluation；保留 negative 和 mixed outcome，不将其改写为正向 transfer result。

### 决策实现

使用 external legend、bar-value label、confidence interval 与 calibrated bandwidth label 重新生成默认 paper、slide、README 和 report asset。LaTeX evaluation paragraph 扩充为比较 1x 与 5x routing、load skew 下的 LMPool 与 routing-only/round robin，以及 memory skew 下的 transfer-only 与 routing-only。正文将 long-prefix routing gain 归因于避免连续 prefill，将 load-skew E2E trade-off 归因于更高的 decode TPOT，将 memory-skew outcome 归因于安全释放但缺乏足够的新增 avoided work。

### 决策结果

所有发布的文档 asset 均使用已审阅 figure。evaluation 同时陈述观察到的比较和数据支持的原因：routing 仅在 saved prefill 足够大时改善；background placement 可降低 TTFT，但可能增加 decode contention；完成 move 可验证 offload safety，却不自动证明 serving speedup。

## 2026-07-29：使 Paper 与校准后的 Transfer Model 对齐

### 决策需求

paper 仍描述了已淘汰的人工选择 `40 ms` transfer prior。实现与当前 paper-suite methodology 实际使用的是 pair-specific、payload-dependent 的 P95 data-path profile、transaction-residual profile 与 nonnegative online residual EWMA。若保留旧模型，系统描述将与 artifact 不一致。

### 决策计划

依据 systems-paper-writing guidance 审阅 paper，将 cost model 修正为实现所用模型，并修订 abstract、introduction、evaluation setup 和 limitation，形成清晰的 problem--gap--design--evidence 叙事。保留报告结果，在不声称普适 serving gain 的条件下陈述 mixed transfer outcome。

### 决策实现

将 abstract 重写为简洁的 systems summary，在 introduction 中加入 prose contribution mapping，以 calibrated pair-and-payload profile、transaction residual、size-bucket EWMA 和 GlobalScheduler 使用的 zero-residual bootstrap behavior 替代 fixed-prior formula。编辑还澄清 piecewise-linear profile 的 extrapolation，移除过期 session-handoff reference，并收紧 implementation、evaluation 与 limitation 的行文。

### 决策结果

paper 现与当前实现和 `20260727T231622Z` evaluation suite 一致：对 sufficiently long prefix 作出强 routing claim，将 background placement 和 move release 报告为已验证机制，并保留“完成 transfer 尚未展示普适 E2E 或 throughput benefit”这一实测边界。

## 2026-07-29：最终 Diagram 与 Transaction-Semantics 审计

### 决策需求

最终 paper audit 发现 transfer cost diagram 仍展示旧 fixed-latency prior；lifecycle diagram 把 abort 描述成恢复 source，但 source 在 successful move finalization 前本就不会释放；background-placement diagram 还使 ingress forecast 看起来是强制条件而非可选条件。

### 决策计划

将 paper figure 和 README asset 与当前 scheduler/transaction protocol 对照。仅修正事实不一致，使用 source script 重新生成 light/dark diagram，不改动纯样式细节。

### 决策实现

transfer cost diagram 改为显示 P95 transaction residual 与 interference-scaled data-path prior，以及 source/placement residual EWMA。background placement 改为使用 observed reuse 或 optional ingress demand。abort semantics 改为释放 target reservation 且保留 source。重新生成 paper、slide、report、README figure output，并更新对应中英文 README。

### 决策结果

当前 diagram set 与实现一致：publication 是 visibility boundary，copy 保留 source，move 仅在 commit 后释放 source，cost admission 使用 calibrated pair-and-payload profile，而非固定 `40 ms` prior。

## 2026-07-29：删除已退役的 Session-Handoff Figure

### 决策需求

未被引用的 `fig_results_summary.png` 和旧 report generator 仍展示已退役的 forecast-assisted session-handoff workload。当前 paper/report 使用 routing、load-skew、memory-skew 与 transfer profile suite，保留这些 artifact 可能使不受数据支持的结论被重新引入。

### 决策计划

核验已退役 figure 是否被 paper、slide、README 或 report 引用。仅在没有活动引用时删除，同时保留当前 suite figure。

### 决策实现

确认 paper 与文档均未引用退役 result summary。删除 paper/slide 中的 `fig_results_summary.png` 副本，以及读取旧 paper batch 的未使用 report generator 和两份 session-handoff output。

### 决策结果

仓库现在只暴露当前 suite 的 routing、skew 和 NVLink transfer profile figure。没有文档路径能够无意中重新生成退役的 session-handoff comparison。

## 2026-07-29：最终 Paper-Figure 审计与 MLSys Review

### 决策需求

最终提交需要 paper figure reference 与已提交 asset 一一对应的审计，并需要严格的 MLSys review，以区分证据缺口与仅靠行文可以修复的问题。

### 决策计划

删除所有未引用 paper figure，核验每个引用 PNG 都存在匹配的当前 README asset，写出具体 review，并仅收窄可能被误读为 transfer performance guarantee 的主张。

### 决策实现

审计 `example_paper.tex` 的所有 `\\includegraphics` reference。删除 paper/slide 中未使用的 `fig_absolute_metrics.png`、退役 session-handoff result figure 和 paper figure directory 中未引用的重复 PDF export。确认十个被引用的 paper figure 都存在，且对应 README asset byte-identical。新增 `docs/reviews/review_20260729.md`，并在 evaluation introduction 中明确 Q1 是主要端到端 performance claim，Q2--Q4 建立 data-path correctness、protocol behavior 和 transfer benefit boundary。report 直接引用同一 paper-suite figure，figure generator 不再生成未使用的 report copy 或重复 PDF export。

### 决策结果

paper 不含未引用 figure artifact。review 将缺少 production baseline/trace、缺少 cost-model ablation、以及尚未得到稳定 incremental transfer speedup 识别为剩余实质风险；这些是保留的 evidence gap，而不是通过措辞掩盖的内容。

## 2026-07-29：维护不重复 Results 的中文阅读版

### 决策需求

项目需要中文版 paper 用于本地阅读与展示，同时英文 MLSys source 仍是提交 artifact。重复保存 result figure 或 bibliography asset 会引入不必要的第二事实源。

### 决策计划

创建独立的中文 LaTeX 文档，翻译技术文字，但保留 equation、citation、numeric result 和带限定条件的 claim。通过同步脚本从英文 paper 的规范 figure/bibliography source 更新本地自包含副本，而不是人工维护另一套结果。

### 决策实现

新增 `docs/papers/paper_zh/paper_zh.tex`、README 与 `sync_assets.sh`。文档支持 `CJKutf8` 的 pdfLaTeX 与 Fandol 的 XeLaTeX，翻译 abstract、system design、workload、evaluation、limitation 和 related work；原研究图保留英文 label，并提供中文 caption。同步脚本复制十张规范 figure 与 bibliography，使目录可独立上传到 Overleaf。

### 决策结果

中文版可在 Overleaf 独立编译；可重复的同步脚本令十张 figure 和 bibliography 与英文 paper 对齐。英文 paper 仍是规范提交源。
