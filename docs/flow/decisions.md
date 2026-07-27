# Decisions

This document records implementation decisions in a STAR-like format:
decision demand, decision plan, decision implementation, and decision result.

## 2026-07-12: Establish Transfer-First Terminology

- Decision demand: The project needs a consistent paper-facing vocabulary. The
  previous code and docs mixed `swap`, `offload`, `migration`, and `transfer`,
  which makes the system story harder to explain.
- Decision plan: Use `transfer` as the external term for cross-GPU KV movement.
  Keep low-level compatibility wrappers where changing names would break tests
  or existing call sites, but expose benchmark metrics and documentation as
  transfer-oriented concepts.
- Decision implementation: Renamed benchmark-facing counters from `swap_count`
  to `transfer_count` in `benchmarks/shared_prefix_benchmark.py`, changed the
  summary table column from `swaps` to `transfers`, and renamed the benchmark
  scenario from `multi-gpu-kv-swapping` to `multi-gpu-kv-transfer`. Updated
  `README.md`, `README_zh.md`, `benchmarks/README.md`, and `tests/README.md`
  to use transfer-oriented wording. Kept internal legacy names such as
  `swap_in`, `swap_out`, and `pending_swap_in` for compatibility, but marked
  them as legacy API / field names in code comments and docs. Data-plane runtime
  stats now emit `transfer_count` while still emitting `swap_count` as a
  compatibility field.
- Decision result: New work will report `transfer_count` and transfer-related
  failure reasons. Existing internal functions can be renamed gradually after
  compatibility tests are updated.

## 2026-07-12: Route With Prefix Reuse And Queue Pressure

- Decision demand: Benchmarks showed high prefix-hit rates but higher latency in
  control-plane scenarios. A prefix-only route policy can overload the prefix
  owner and increase TTFT/E2E latency.
- Decision plan: Extend the global block snapshot with worker queue state, then
  include queue pressure in the route score. Prefix reuse remains the primary
  benefit, but routing should avoid sending all shared-prefix requests to a
  congested rank.
- Decision implementation: Added `waiting_sequences_per_gpu` and
  `running_sequences_per_gpu` to `GlobalBlockManager`, plus
  `get_queue_pressure()` with `waiting + 2 * running` as the current lightweight
  pressure estimate. Extended `ControlPlaneClient.report_block_state()` and
  `control_plane_process` message handling to carry `waiting_sequences` and
  `running_sequences`. Updated `data_plane_process.send_block_state()` and
  `Scheduler._sync_local_state_to_global()` to report queue sizes. Updated
  `GlobalScheduler.route_sequence_meta()` so prefix-hit scoring uses
  `hit_count * topo_weight * route_prefix_hit_weight - queue_pressure *
  route_queue_pressure_weight + free_blocks * route_free_block_weight`. Also
  changed no-prefix / no-hit fallback to prefer lower queue pressure before
  free-block count. Added config knobs `route_prefix_hit_weight`,
  `route_queue_pressure_weight`, `route_free_block_weight`, and
  `route_cache_queue_slack`.
- Decision result: Routing can now trade off prefix locality against queue
  pressure using a lightweight control-plane snapshot. This is still a static
  cost model, not a learned or calibrated latency predictor.

## 2026-07-12: Cache Repeated Prefix Route Decisions

- Decision demand: Shared-prefix workloads repeatedly route the same full-block
  prefix. Recomputing the full route decision through the control plane on every
  request adds fixed TTFT overhead.
- Decision plan: Add a small control-plane route cache keyed by prefix hash.
  Reuse a cached target only when the target still owns that prefix and has
  enough free blocks for the incoming sequence.
- Decision implementation: Added an in-process `route_cache` dictionary inside
  `control_plane_process`, keyed by `prefix_hash`. On a route request, the
  control plane first checks whether the cached target still has the prefix in
  `GlobalBlockManager.lookup_prefix()`, has enough free blocks, is still a
  valid local/NVLink candidate, and has queue pressure within
  `route_cache_queue_slack` of the least-loaded candidate. If all checks pass,
  the response returns `route_info.reason = "route_cache"`; otherwise the
  request falls back to `GlobalScheduler.route_sequence_meta()` and refreshes
  the cache only when a prefix hit is selected. Added tests for valid cache
  reuse and congested-cache bypass.
- Decision result: Warm shared-prefix requests take a shorter route-decision
  path when the cached owner is still viable, without bypassing queue-aware
  overload protection.

## 2026-07-12: Enforce Benchmark KV Block Budgets

- Decision demand: Transfer-path benchmarks need to create real KV block
  pressure. If the runtime ignores benchmark-provided `max_cached_blocks`, the
  `multi-gpu-kv-transfer` scenario can report `transfers = 0` even when the
  command line asks for a small transfer budget.
- Decision plan: Treat `max_cached_blocks` as an upper bound on automatically
  computed KV cache capacity instead of allowing `ModelRunner.allocate_kv_cache()`
  to overwrite it unconditionally.
- Decision implementation: Updated `ModelRunner.allocate_kv_cache()` to compute
  the memory-derived KV block capacity, read the configured
  `max_cached_blocks`, and use `min(memory_capacity, configured_max_blocks)` as
  the actual per-rank block count before the cross-rank MIN all-reduce. This
  preserves safety under low memory while making benchmark knobs such as
  `--eviction-max-cached-blocks` effective.
- Decision result: Transfer stress experiments can now intentionally constrain
  the KV block budget and should be able to trigger rebalance / transfer when
  request pressure exceeds local free blocks.

## 2026-07-12: Add Direct KV Transfer Microbenchmark

- Decision demand: End-to-end shared-prefix benchmarks mainly measure routing,
  prefix locality, queueing, and model execution together. They cannot prove the
  raw benefit of the transfer primitive when `transfers = 0`.
- Decision plan: Add a focused benchmark that isolates KV transfer from routing
  and decode. The benchmark should validate data correctness and report latency
  / bandwidth for the same KV tensor shape used by the model.
- Decision implementation: Added `benchmarks/kv_transfer_benchmark.py`. The
  script spawns two NCCL ranks, allocates synthetic KV cache tensors, fills
  source KV blocks on rank 0, transfers them into rank 1 using the existing
  `kv_transfer.swap_in` legacy API, validates copied K/V data, and reports
  mean latency, p95 latency, bytes per iteration, and effective GiB/s.
- Decision result: Transfer Principle can now be validated independently from
  routing. End-to-end benchmarks should still be used to show whether transfer
  helps under data-skew pressure, but the microbenchmark verifies the data-path
  primitive itself.

## 2026-07-12: Add Copy-Style Transfer For Pinned Prefix Blocks

- Decision demand: End-to-end transfer stress showed `rebalance_fail = pinned`
  and `transfers = 0`. Move-style transfer cannot release source blocks that are
  still referenced by live sequences, but hot shared prefixes are often exactly
  those pinned blocks.
- Decision plan: Add copy-style transfer as a separate transfer mode. Move mode
  still frees source space. Copy mode replicates KV blocks to an NVLink peer and
  keeps the source block live, improving cache fluidity without violating
  ref-count safety.
- Decision implementation: Added `mode` to rebalance plans and per-transfer
  records. `GlobalScheduler.plan_rebalance()` first tries existing move
  candidates. It only selects pinned source blocks from
  `GlobalBlockManager.block_hash` and emits `mode = "copy"` when the caller
  explicitly passes `allow_copy=True`; foreground allocation rebalance keeps the
  default `allow_copy=False` because copy does not free source space.
  `data_plane_process.execute_rebalance_plan()` skips pinned source rejection
  for copy transfers and does not release source blocks after the NCCL transfer.
  Runtime stats now include `transfer_copy_count`. `GlobalBlockManager.record_block_copy()`
  records a copied target location without deleting the source location. Tests
  cover copy plan generation, default no-copy foreground rebalance, and
  control-plane copy rebalance.
- Decision result: Copy-style transfer exists as an explicit replication path
  and as the foundation for future speculative/background transfer. It is not
  used as a default foreground memory-reclamation mechanism, because it would
  report success without freeing the local source blocks needed by the waiting
  request.

## 2026-07-12: Add E2E Orchestration Test

- Decision demand: Unit tests covered individual components, but there was no
  single test named as end-to-end coverage for ingress routing and worker result
  aggregation.
- Decision plan: Add a lightweight no-CUDA e2e test that exercises LLMEngine
  orchestration with mocked processes and tokenizer.
- Decision implementation: Added `tests/test_e2e.py`. The test constructs an
  `LLMEngine`, replaces the control-plane client with a fake router that sends a
  prompt to rank 1, injects first-token / prefill / runtime / finished worker
  messages, then verifies `engine.step()` returns all aggregated outputs.
- Decision result: The tests now include an explicit e2e orchestration smoke
  test without requiring model weights or CUDA.

## 2026-07-12: Normalize Benchmark Entry Names

- Decision demand: Benchmark file names were inconsistent with the paper-facing
  decomposition into routing, transfer, and end-to-end evaluation.
- Decision plan: Add canonical benchmark entry points while keeping old files as
  compatibility implementation modules.
- Decision implementation: Added `benchmarks/benchmark_e2e.py` as the canonical
  end-to-end wrapper over `shared_prefix_benchmark.py`, added
  `benchmarks/benchmark_kv_transfer.py` as the canonical wrapper over
  `kv_transfer_benchmark.py`, and added `benchmarks/benchmark_kv_routing.py`
  to run only `single-gpu`, `multi-gpu`, and `multi-gpu-kv-routing` scenarios.
- Decision result: New commands can use clean benchmark names, while existing
  commands using `shared_prefix_benchmark.py` and `kv_transfer_benchmark.py`
  continue to work.

## 2026-07-12: Add Background Speculative Copy Transfer

- Decision demand: Foreground copy-style transfer can move hot pinned prefix
  data, but it does not free source blocks. Using it to satisfy a blocked local
  allocation path makes latency worse and can report progress without creating
  the free space the current request needs.
- Decision plan: Keep foreground rebalance move-only by default, and add a
  separate background path for speculative copy. The control plane should return
  the route decision first, then opportunistically copy a small number of hot
  prefix blocks to an NVLink peer for future requests.
- Decision implementation: Added `enable_background_copy`,
  `background_copy_max_blocks`, and `background_copy_cooldown_s` control-plane
  knobs. `control_plane_process()` now inspects successful prefix-hit route
  decisions, picks the prefix owner as source, picks its NVLink peer as target,
  skips copies already present on the target, and enqueues a background
  `mode = "copy"` transfer plan through the existing two-phase
  prepare/execute protocol. Background plans have no request reply target, so
  completion and failure cleanup no longer assumes every rebalance has a
  foreground requester. `benchmarks/shared_prefix_benchmark.py` enables this
  path conservatively with one copied block and a cooldown, and exposes CLI
  knobs to disable or tune it. `tests/test_control_plane.py` now verifies that
  route response returns before the source/target ranks execute the background
  copy plan.
- Decision result: Transfer now has two clear roles: foreground move-style
  transfer for immediate space reclamation, and background copy-style transfer
  for speculative cache fluidity. The default benchmark path can measure
  whether proactive NVLink replication improves later prefix locality without
  turning the current request path into a blocking transfer path.

## 2026-07-12: Split Foreground Rebalance And Background Copy Metrics

- Decision demand: Benchmark output showed `copies > 0` together with
  `reb ok = 0`, which made it look as if all transfer work failed. The old
  `reb ok` column only counted foreground `ControlPlaneClient.rebalance()`
  responses, while successful background copy plans were only visible through
  `copies`.
- Decision plan: Keep foreground rebalance metrics separate from background
  speculative copy metrics. Foreground metrics should describe current-request
  space reclamation. Background metrics should describe async copy plans that
  may improve future locality but do not unblock the current allocation path.
- Decision implementation: `data_plane_process.execute_rebalance_plan()` now
  emits `background_copy_success` when a background copy plan executes on the
  source rank, and emits `background_copy_fail` with a reason when a background
  prepare fails. `benchmarks/shared_prefix_benchmark.py` aggregates the new
  counters into `ScenarioResult`, exports them to JSON, and renames the table
  columns to `fg ok`, `fg fail`, `bg ok`, and `bg fail`. Benchmark docs now
  explain the foreground/background metric split and recommend less constrained
  settings for validating speculative copy.
- Decision result: Benchmark summaries now distinguish failed foreground
  move-style rebalances from successful or failed background copy-style transfer
  plans. This makes it easier to diagnose whether poor results come from
  allocation pressure, missing target space, or speculative copies not turning
  into later local prefix hits.

## 2026-07-12: Make Speculative Copy Produce Real Local Prefix Hits

- Decision demand: Benchmarks showed high `route hit` and `owner hit` but
  `local hit = 0`. The control plane could route to a prefix owner, yet the
  data-plane worker did not observe reusable local cached tokens during prefill.
- Decision plan: Fix the prefix-reuse chain before changing route scores.
  Background copy should replicate an ordered prefix hash chain, not just the
  terminal prefix hash. Transfer-in blocks should also be reusable by local
  `BlockManager.allocate()` even though their original token ids are not stored
  in the transferred metadata.
- Decision implementation: `ControlPlaneClient.route_sequence()` now sends
  `prefix_hashes`, the cumulative hash for every full prompt block. The control
  plane preserves that chain in `route_info`, and background copy walks it in
  order to choose source blocks, bounded by `background_copy_max_blocks`.
  `BlockManager.register_swap_in_blocks()` now marks transferred blocks with
  `token_ids = None`, meaning the KV payload is trusted by hash. `BlockManager.allocate()`
  accepts that trusted hash match, but only increments `seq.num_cached_tokens`
  for contiguous prefix hits from block 0; non-contiguous hits can share block
  table entries but do not falsely reduce prefill input length.
- Decision result: Speculative copy can now copy the beginning of a shared
  prefix, which is the only part that can produce real local prefill hits. Tests
  cover trusted transfer-in reuse, non-contiguous hit accounting, and ordered
  prefix-chain copy planning.

## 2026-07-12: Report Local Hit At Request Level

- Decision demand: Stress benchmarks with heavy preemption/retry produced
  `local hit` values such as `0.20%`, which were hard to interpret. The metric
  was counted per prefill event, so repeated failed scheduling attempts inflated
  the denominator and hid whether a request ever observed local cached prefix
  tokens.
- Decision plan: Keep `local hit` as the user-facing cache reuse metric, but
  count it per request. A request should count as a local hit if any prefill
  attempt for that sequence reports cached prefix tokens.
- Decision implementation: `benchmarks/shared_prefix_benchmark.py` now tracks
  `prefill_seen_seq_ids` and `prefill_hit_seq_ids`, and computes
  `prefix_hit_rate = len(hit_seq_ids) / len(seen_seq_ids)`. `benchmarks/README.md`
  documents the request-level definition.
- Decision result: Future benchmark tables will show whether local reuse
  reached requests, without dilution from preemption/retry event counts. This
  makes A/B comparisons of background copy easier to interpret.

## 2026-07-12: Make Allocation Capacity Prefix-Reuse Aware

- Decision demand: After fixing request-level local-hit accounting, benchmarks
  still showed high `route hit` / `owner hit` but low `local hit`, plus many
  foreground rebalance failures. Requests routed to a prefix owner were still
  being rejected by local capacity checks before `BlockManager.allocate()` could
  reuse cached blocks.
- Decision plan: Change the prefill capacity check from "does this worker have
  free blocks for the whole sequence?" to "does this worker have free blocks
  for the blocks that are not already locally cached?" Rebalance shortage should
  use the same required-new-block count.
- Decision implementation: Added `BlockManager.num_required_new_blocks(seq)`,
  which walks the sequence's full-block hash chain and counts only blocks that
  are not reusable from `hash_to_block_id`. `BlockManager.can_allocate(seq)` now
  compares free blocks against that value. `Scheduler.schedule()` computes
  prefill rebalance shortage from `num_required_new_blocks(seq)` rather than
  `seq.num_blocks`. Tests now cover prefix-aware capacity in both
  `BlockManager` and `Scheduler`.
- Decision result: Requests routed to a prefix owner can now be admitted even
  when the worker lacks enough free blocks for the full prompt, as long as the
  missing portion fits. This should raise worker-side local hits and reduce
  unnecessary foreground rebalance failures.

## 2026-07-13: Allow E2E Benchmark To Scale Beyond Two GPUs

- Decision demand: Eight-GPU experiments were blocked because the end-to-end
  benchmark still hard-coded `world_size = 2` for multi-GPU, routing, transfer,
  and LMPool scenarios. Setting `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` alone
  did not start eight data-plane workers.
- Decision plan: Add an explicit benchmark `--world-size` parameter. Keep the
  default at two GPUs to preserve existing commands, but allow callers to opt
  into eight-GPU runs and validate that enough CUDA devices are visible.
- Decision implementation: `benchmarks/shared_prefix_benchmark.py` now parses
  `--world-size`, checks it against `torch.cuda.device_count()`, and passes it
  to `make_config()` for all multi-GPU scenarios. The benchmark comments and
  `benchmarks/README.md` now document `--world-size` and the need to quote
  semicolon-separated `--nvlink-pairs` values.
- Decision result: The canonical `benchmark_e2e.py` entry can now launch N
  data-plane workers for multi-GPU scenarios, including eight visible GPUs, as
  long as the model and NCCL setup fit the machine.

## 2026-07-13: Remove World Collective From Pairwise KV Transfer

- Decision demand: Four-GPU transfer stress runs appeared to hang after NCCL
  unbatched P2P warnings. The transfer primitive ended with `dist.all_reduce()`,
  which requires every rank in the process group to participate. In a larger
  world, only the source and target ranks of one NVLink pair enter the transfer
  function, so the collective can deadlock.
- Decision plan: Treat blocking NCCL send/recv pairs as the synchronization
  boundary for a pairwise transfer. Remove world-size collectives from
  `swap_in()` and `swap_out()` unless all ranks are explicitly orchestrated to
  participate.
- Decision implementation: Removed the final `dist.all_reduce()` from
  `src/lmpool/engine/kv_transfer.py` in both legacy transfer APIs. Added
  comments documenting why point-to-point transfer must not call a world
  collective inside a multi-pair process group.
- Decision result: KV transfer can now execute between one NVLink pair inside a
  larger data-plane world without waiting for unrelated ranks. This should
  unblock four-GPU and larger benchmark runs that trigger background copy or
  foreground transfer on only one pair at a time.

## 2026-07-13: Make Routing Baseline Transfer-Free

- Decision demand: The `multi-gpu-kv-routing` benchmark scenario still showed
  foreground rebalance failures. That mixed routing behavior with transfer /
  rebalance behavior and made the scenario unsuitable as a routing-only
  baseline.
- Decision plan: Add a scheduler-level switch for foreground rebalance and
  disable both foreground rebalance and background copy in the routing-only
  benchmark scenario. Keep those mechanisms enabled for `multi-gpu-kv-transfer`
  and `multi-gpu-lmpool`.
- Decision implementation: Added `Scheduler.enable_foreground_rebalance`,
  wired it from `config["enable_foreground_rebalance"]` in `data_plane_process`,
  and set `routing_config["enable_foreground_rebalance"] = False` plus
  `routing_config["enable_background_copy"] = False` in
  `benchmarks/shared_prefix_benchmark.py`. Added a scheduler regression test
  that verifies rebalance is not called when the switch is disabled. Updated
  benchmark docs to define `multi-gpu-kv-routing` as the routing-only baseline.
- Decision result: Future `multi-gpu-kv-routing` rows should report zero
  foreground rebalance, zero background copy, and zero transfer counts. This
  cleanly separates routing benefits from transfer/rebalance behavior.

## 2026-07-13: Add Token-Aware Load Score To Routing

- Decision demand: Four-GPU experiments showed that pure prefix-locality
  routing can improve TTFT and tail latency but may sacrifice throughput by
  routing too many shared-prefix requests to the same owner GPU. The existing
  queue pressure used only waiting/running sequence counts and did not represent
  prompt length or decode occupancy well enough.
- Decision plan: Extend worker state reports with token-level load and update
  global routing to use `locality_score - load_score + capacity_score`. Keep
  prefix locality as the primary signal, but allow the scheduler to bypass a
  prefix owner when its token-aware load is much higher than the least-loaded
  candidate.
- Decision implementation: `GlobalBlockManager` now stores
  `waiting_tokens_per_gpu` and `running_tokens_per_gpu` and exposes
  `get_load_score()`. `ControlPlaneClient.report_block_state()`,
  `Scheduler._sync_local_state_to_global()`, and `data_plane_process.send_block_state()`
  now propagate token counts. `GlobalScheduler` now has configurable
  `route_load_weight`, token weights, running-sequence weight, and
  `route_load_bypass_threshold`; route scoring subtracts token-aware load, and
  prefix-hit routes can return `reason = "prefix_hit_load_bypass"` when the
  owner is overloaded. Tests cover owner-load penalties and bypass behavior.
- Decision result: LMPool routing should be less likely to over-concentrate
  long shared-prefix requests on a few owner GPUs, improving throughput under
  request skew while preserving prefix locality when load is balanced.

## 2026-07-13: Gate Background Copy By Hot Prefix

- Decision demand: Benchmarks where routing-only already had high local hit
  showed that eager speculative copy added transfer overhead while improving
  local hit by less than one percentage point. Background copy needed to become
  selective instead of firing on the first prefix hit.
- Decision plan: Track route-time prefix hit counts in the control plane and
  only allow background copy after a prefix becomes hot. Also prevent multiple
  background copy plans from running concurrently on the same `src -> dst` pair.
- Decision implementation: Added `background_copy_hot_threshold` to the control
  plane and benchmark CLI. `control_plane_process()` now increments
  `prefix_route_hits[prefix_hash]` for prefix-hit routes and returns early until
  the threshold is reached. It also maintains `background_copy_inflight_pairs`
  and releases the pair when the background plan succeeds, fails, aborts, or is
  cleared after worker failure. Tests now cover threshold-gated copy behavior.
- Decision result: Speculative transfer is now hot-prefix gated. The default
  benchmark setting avoids eager copy overhead, while experiments can still set
  `--background-copy-hot-threshold 1` to reproduce the previous eager policy.

## 2026-07-13: Add P90 Latency Reporting And Visualization

- Decision demand: Load-skew and transfer-relief experiments need tail-latency
  visibility beyond the existing mean latency and p95 table column. The figure
  did not show P90, making it harder to tell whether a mechanism primarily
  improves average latency or tail latency.
- Decision plan: Add P90 latency fields to benchmark results, include
  `p90(e2e)` in the summary table, and draw P90 E2E in the latency subplot
  alongside mean TTFT, the then-current per-output-token proxy, and mean E2E.
- Decision implementation: Extended `ScenarioResult` with `p90_ttft_s`,
  `p90_ttpt_s`, and `p90_e2e_s`. Reused the existing `_percentile()` helper for
  both P90 and P95 calculations. Updated the summary table and PNG figure to
  include `p90(e2e)`. The `ttpt` field described here was later found to be an
  E2E-per-token proxy and was superseded by the decode TPOT decision below.
- Decision result: Future benchmark JSON, tables, and figures can directly
  compare mean latency against P90/P95 tail latency, which is necessary for
  evaluating load-skew relief and transfer-triggered tail improvements.

## 2026-07-15: Use Paper-Friendly Benchmark Figure Colors

- Decision demand: The E2E benchmark figure reused matplotlib default colors
  across subplots, making different metric groups harder to distinguish in
  paper-style figures.
- Decision plan: Assign explicit muted, colorblind-friendly palettes to each
  subplot group so throughput, latency, prefix-hit, and utilization metrics use
  distinct colors.
- Decision implementation: Updated `save_summary_figure()` in
  `benchmarks/shared_prefix_benchmark.py` with fixed Okabe-Ito / muted academic
  color groups, thin bar outlines, and light horizontal grid lines. The latency
  subplot keeps TTFT, the legacy per-output-token proxy, mean E2E, and P90 E2E
  as separate visible series. This palette was superseded by the brighter,
  synchronized visual palette decision below.
- Decision result: Future `--output-figure` PNGs are more suitable for paper
  drafts and easier to read when multiple metric groups appear in the same
  summary figure.

## 2026-07-15: Strengthen Load-Aware Routing And Rank Diagnostics

- Decision demand: Four-GPU E2E results showed that LMPool reduced TTFT and
  tail latency but still lost throughput to round-robin multi-GPU baselines.
  The likely causes were locality-heavy routing that underutilized some GPUs,
  repeated foreground transfer attempts that produced `no_plan`, and missing
  per-rank diagnostics to prove load skew.
- Decision plan: Make routing more load-sensitive by default, expose the load
  knobs through benchmark CLI, avoid foreground transfer for tiny one-block
  shortages in benchmark runs, and persist per-rank execution counters in the
  benchmark JSON.
- Decision implementation: Updated benchmark defaults to
  `route_load_weight=0.03`, `route_load_bypass_threshold=256`, and
  `route_cache_queue_slack=256`, then added matching CLI arguments. Added
  `foreground_transfer_min_blocks` and wired it into `Scheduler` through
  `data_plane_process`, so benchmark foreground transfer only fires when a
  shortage is large enough to justify control-plane and NCCL overhead. Added
  `foreground_transfer_fail_cooldown_s=2.0` for benchmark runs to avoid rapid
  repeated `no_plan` attempts. Preserved foreground transfer in
  `multi-gpu-kv-transfer` so that scenario can still isolate transfer behavior.
  Added rank attribution for `prefill_stats` and `runtime_stats` in `LLMEngine`,
  and added `rank_stats` to benchmark JSON with submitted requests, local prefix
  hits, execution tokens/time, transfer counts, and rebalance counters.
- Decision result: LMPool experiments can now trade locality for parallelism
  without source edits, repeated low-value foreground transfer attempts should
  drop, and saved JSON can explain whether throughput loss comes from rank load
  imbalance, transfer overhead, or cache-locality choices.

## 2026-07-15: Split Benchmark Workloads And Add Per-Rank GPU Metrics

- Decision demand: The previous optimization did not fully implement the
  workload split between locality-oriented routing experiments and
  load/memory-skew transfer experiments. It also lacked per-rank GPU utilization
  in JSON, so load concentration still required inference from global averages.
- Decision plan: Add an explicit workload selector to the E2E benchmark and
  make GPU metric sampling respect `CUDA_VISIBLE_DEVICES`. Attach per-rank GPU
  utilization and memory utilization to `rank_stats`.
- Decision implementation: Added `--workload {locality,load-skew,memory-skew}`.
  `locality` keeps the original single shared-prefix workload for routing,
  `load-skew` mixes a hot prefix with cold prefixes, and `memory-skew` uses a
  longer hot prefix to increase KV block pressure for transfer/rebalance
  experiments. Reworked `GpuMetricSampler` to map logical ranks to physical GPU
  IDs from `CUDA_VISIBLE_DEVICES`, sample only those GPUs, and expose
  `summarize_by_rank()`. The benchmark now merges per-rank GPU util and memory
  util into each scenario's JSON `rank_stats`.
- Decision result: Future experiments can separate routing locality claims from
  transfer/rebalance stress claims, and the saved JSON directly shows whether
  poor throughput comes from per-rank request skew, token skew, execution time,
  or GPU utilization imbalance.

## 2026-07-15: Account For Optimistic Route Load

- Decision demand: The locality benchmark showed `multi-gpu-lmpool` routing
  109 of 128 requests to rank 1 while other ranks were nearly idle. Route hit
  was high, but throughput collapsed because route cache/load scoring did not
  account for requests already routed but not yet reflected in worker
  block-state reports.
- Decision plan: Treat every routing decision as an optimistic waiting-load
  reservation in the authoritative control-plane state. The next worker
  `block_state` snapshot still overwrites the estimate, but consecutive route
  requests will see the pending load immediately.
- Decision implementation: Added `GlobalBlockManager.reserve_route_load()` to
  increment `waiting_sequences_per_gpu` and `waiting_tokens_per_gpu` after a
  route decision. `control_plane_process()` now calls it immediately after
  `reserve_blocks()`. Added a regression test that repeatedly routes a shared
  prefix before any worker report and asserts the targets are not sticky to a
  single cached owner.
- Decision result: Route cache and load-aware scoring now include in-flight
  routed work, so high-locality bursts should distribute across available ranks
  instead of collapsing onto the first prefix owner.

## 2026-07-15: Visualize Per-Rank Benchmark Diagnostics

- Decision demand: `rank_stats` in benchmark JSON exposed request and GPU
  imbalance, but reading raw JSON made it hard to quickly diagnose route
  collapse or underutilized ranks from experiment artifacts.
- Decision plan: Keep the existing summary figure unchanged and automatically
  save a second per-rank diagnostics figure whenever `--output-figure` is used.
- Decision implementation: Added `save_rank_stats_figure()` to
  `benchmarks/shared_prefix_benchmark.py`. It derives a sibling filename using
  the `_rank_stats` suffix and plots per-rank submitted requests, output tokens,
  GPU utilization, and local prefix hit rate across all scenarios with distinct
  muted paper-style colors.
- Decision result: Each benchmark run with `--output-figure foo.png` now also
  emits `foo_rank_stats.png`, making route skew and GPU imbalance visible
  without manually inspecting JSON.

## 2026-07-15: Make Route Cache Owner-Balanced

- Decision demand: Locality benchmark rank diagnostics showed LMPool could
  still collapse most requests onto one rank even when route hit was high.
  A single-target route cache made prefix locality behave like sticky routing.
- Decision plan: Keep the route cache as a fast path, but make it choose among
  all currently valid prefix owners using current optimistic load instead of
  blindly reusing the previously cached target.
- Decision implementation: Updated `control_plane_process()` so a cached prefix
  first gathers all owner GPUs that are valid candidates and have enough free
  blocks, then selects the lowest-load owner with a free-block tiebreaker. If
  the lightest owner is still too congested compared with the lightest
  candidate, routing falls back to full global scoring. Added a regression test
  covering multi-owner cache balancing.
- Decision result: Prefix locality remains a fast path, but repeated shared
  prefix requests should distribute across prefix owners instead of sticking to
  the first cached rank.

## 2026-07-15: Disable Route Cache By Default

- Decision demand: The latest locality run still showed poor LMPool throughput.
  Rank diagnostics showed requests were no longer confined to one GPU, but were
  still confined to existing prefix owners. The route-cache fast path bypassed
  full load-aware scoring, so non-owner GPUs were not seeded even when owners
  were overloaded.
- Decision plan: Keep route-cache code only as an opt-in test/experiment path
  and make the default control plane always use full route scoring. This keeps
  the runtime behavior simple: every request sees the same prefix/locality/load
  policy.
- Decision implementation: Added `enable_route_cache` config in
  `control_plane_process()` with default `False`. Existing route-cache tests now
  enable it explicitly. Added a global scheduler regression test showing ingress
  routing can bypass an overloaded prefix owner and send work to a free GPU.
- Decision result: Default LMPool routing no longer has a sticky cache fast path
  that can override load-aware routing. High-locality bursts should now spread
  beyond the first prefix owners when those owners accumulate optimistic load.

## 2026-07-15: Remove Duplicate-Replica Prefix Score Amplification

- Decision demand: Locality runs still showed routing collapse after disabling
  the route-cache fast path. Rank diagnostics showed most requests were sent to
  a single prefix owner. The root cause was that routing counted every physical
  copy of the same prefix hash on a GPU as a separate prefix hit, so routing more
  requests to one GPU created more duplicate copies and further increased that
  GPU's future score.
- Decision plan: Treat prefix locality as content presence, not duplicate
  physical replica count. For a given prefix hash, each GPU should contribute at
  most one hit to the routing score.
- Decision implementation: Changed `GlobalScheduler.route_sequence_meta()` to
  aggregate hit hashes as a set per GPU before computing `gpu_hit_count`.
  Updated the scheduler regression test so duplicate replicas on a remote GPU no
  longer outweigh an equivalent local prefix hit.
- Decision result: Prefix-hit score can no longer self-amplify merely because a
  GPU has served many duplicate requests. Load-aware routing should now be able
  to seed idle GPUs instead of being dominated by duplicate KV replicas.

## 2026-07-16: Preserve Pending Admission Load Across Worker Snapshots

- Decision demand: Locality benchmarks still concentrated requests on one or
  two ranks because newly routed work disappeared from the load estimate before
  the destination worker admitted it.
- Decision plan: Keep control-plane admission reservations separate from worker
  snapshots and clear them only when the worker receives the sequence.
- Decision implementation: Added pending sequence/token counters to
  `GlobalBlockManager`. Routing load and queue pressure include these counters,
  while `update_gpu_state()` cannot overwrite them. After receiving a batch,
  `DataPlaneProcess` first publishes a block-state snapshot containing the new
  waiting sequences, then sends sequence-specific `route_admitted` messages.
  FIFO ordering guarantees that the control plane installs real waiting load
  before clearing matching pending reservations. Added regression coverage for
  stale snapshots, unrelated acknowledgements, and the final handoff from
  pending to worker-reported load.
- Decision result: There is no zero-load observation window between route
  reservation and worker admission, so a synchronous burst cannot repeatedly
  route to an owner that only appears idle because its acknowledgement raced
  ahead of its state report.

## 2026-07-16: Match Rank Charts To Metric Semantics

- Decision demand: Connecting discrete rank IDs with lines obscured load skew.
- Decision plan: Use pies for additive shares and bars for independent rates.
- Decision implementation: Reworked `save_rank_stats_figure()` to render one
  row per scenario. Request and output-token shares use pie charts; GPU
  utilization and local prefix-hit rate use labeled bars.
- Decision result: Request concentration is directly visible without treating
  utilization and hit rates as parts of a whole.

## 2026-07-16: Stop Routing Into A Full Prefix Owner

- Decision demand: The `0957` locality result still sent 116 of 128 routing
  requests to one rank and re-executed prefill 1061 times, while three ranks
  stayed near 3% GPU utilization.
- Decision plan: Inspect the exact capacity-failure route branch before changing
  score weights. Preserve prefix-owner routing only while it is executable.
- Decision implementation: Changed `GlobalScheduler.route_sequence_meta()` so
  the `failed_gpus` branch first searches all topology-eligible candidates for
  enough free blocks and selects the lowest-load candidate. It returns
  `prefix_hit_needs_rebalance` only when no candidate can directly allocate the
  request. Replaced the test that required routing into an undersized owner and
  added separate fallback and all-candidates-full tests.
- Decision result: A full prefix owner can no longer absorb the entire locality
  workload while idle GPUs have capacity. Routing-only no longer depends on a
  disabled transfer path, and LMPool invokes transfer only for a real global
  capacity shortage.

## 2026-07-16: Make E2E Comparisons Reproducible

- Decision demand: The balanced `1021` run showed clear routing latency gains,
  but scenario output totals differed because temperature sampling could emit
  EOS early. Single-run throughput differences therefore mixed system behavior
  with output-length and runtime variance.
- Decision plan: Equalize generated work, seed every data-plane process, and
  support repeated trials with explicit variability reporting.
- Decision implementation: The E2E benchmark now defaults to `ignore_eos=True`,
  accepts `--seed` and `--repetitions`, and propagates a rank-specific stable
  seed before model initialization. Repeated scenarios are aggregated into
  mean results with throughput, goodput, TTFT, and E2E population standard
  deviations in JSON and a dedicated console table.
- Decision result: Every request performs the configured decode work, and paper
  comparisons can distinguish a stable gain from run-to-run noise.

## 2026-07-16: Retain Completed Prefix Blocks Until LRU Reclamation

- Decision demand: `route hit` and locality gains were timing-dependent because
  `BlockManager.deallocate()` deleted a complete hashed block as soon as its
  final active reference ended. The advertised prefix cache therefore retained
  no KV state across non-overlapping requests.
- Decision plan: Keep complete unreferenced KV blocks as evictable cache, reclaim
  them only under capacity pressure, and preserve transfer as LMPool's first
  pressure response.
- Decision implementation: Complete blocks with `ref_count == 0` now remain in
  `used_block_ids` and `hash_to_block_id` with an LRU timestamp; partial blocks
  are still released immediately. Added protected-prefix-aware local LRU
  reclamation. Scheduler first attempts configured foreground transfer, then
  reclaims cold local cache before preempting a live sequence. Added tests for
  reuse after request completion and LRU reclamation that protects the incoming
  sequence's cached prefix.
- Decision result: Prefix ownership and global page-table entries persist across
  requests, while cold cache remains reclaimable and repeated prefill is avoided
  when local cache pressure can be resolved without preemption.

## 2026-07-16: Use Multiple Long Prefix Groups In The Locality Workload

- Decision demand: With one shared prefix, round-robin warmed one replica on
  every GPU and reached the same 95.31% worker-local hit rate as routing in the
  six-GPU run, so final hit rate could not isolate routing locality.
- Decision plan: Replace the single hotspot with a configurable balanced set of
  long prefixes and decouple prefix order from the round-robin rank cycle.
- Decision implementation: Added `--locality-prefix-groups` with a default of
  16. Every locality group starts with a distinct stable marker and retains the
  configured long repeated body. Requests are distributed evenly across groups
  and shuffled with `--seed` before suffixes are attached. Made `benchmarks` an
  importable package, added generator regression tests, argument validation,
  and synchronized benchmark and repository documentation.
- Decision result: Round-robin must build redundant copies of several prefix
  groups across workers, while KV-aware routing can consolidate each group at
  its existing owners. Worker local-hit rate, prefill work, and cache footprint
  can now distinguish the two policies.

## 2026-07-16: Prevent Prefill-Decode Preemption Ping-Pong

- Decision demand: The six-GPU multi-prefix run submitted 128 requests but
  executed 4,791 to 7,090 prefill attempts per scenario. Waiting prefill could
  displace live decode work; the preempted sequence then returned to the front
  of the waiting queue and consumed the released blocks again.
- Decision plan: Bound admission by immediate decode growth, preserve running
  work when a new prompt does not fit, and avoid transfer attempts that only
  increase concurrency rather than resolve a real allocation shortage.
- Decision implementation: Added per-sequence remaining decode-block
  calculation and scheduler admission headroom for the next growth block of
  each active and incoming sequence. Prefill now reclaims cold unreferenced
  cache first, triggers foreground transfer only when the request itself lacks
  blocks, and falls through to decode instead of preempting a running sequence.
  Prefill and decode transfer failures share one capacity cooldown. In the
  exceptional decode victim path, the victim is queued at the back and the
  blocked decode receives the newly freed block immediately. Added scheduler
  tests for decode preservation and admission headroom.
- Decision result: Long-prompt admission can no longer create the immediate
  prefill/decode ping-pong responsible for repeated full-prompt execution, and
  transfer is not invoked solely to consume reserved decode capacity.

## 2026-07-16: Measure Initial Prefix Reuse Separately From Retries

- Decision demand: A request counted as a local prefix hit if any retry hit
  blocks left by its own earlier prefill, inflating round-robin local hit to
  85.94% despite severe cache churn.
- Decision plan: Make initial cache reuse the primary locality metric and expose
  retry work directly instead of hiding it inside a binary hit rate.
- Decision implementation: Sequence and data-plane messages now carry prefill
  attempt and preemption counters. Benchmark `local hit` includes only the
  first prefill per sequence. Added initial cached-token ratio, total prefill
  attempts, preemption count, and redundant prefill-token count to scenario
  JSON, rank statistics, and the horizontal summary table. Added an explicit
  `kv_ready` block lifecycle: allocation computes hashes privately, while the
  data plane publishes complete blocks to local/global prefix indexes only
  after successful model execution writes their KV data. Updated metric
  documentation accordingly.
- Decision result: Routing locality and scheduler churn are now independently
  measurable; retries can no longer improve the reported local-hit rate.

## 2026-07-16: Reserve Source Blocks Across Concurrent Transfer Plans

- Decision demand: A six-GPU transfer trial crashed because concurrent
  foreground rebalance plans selected the same source block; the first execute
  released it and the second execute raised `KeyError` while releasing it
  again. Ignoring the second release would still permit duplicate NCCL traffic
  and possible send/recv divergence.
- Decision plan: Give each pending transfer plan exclusive ownership of its
  source blocks before prepare begins, and reject stale source state before any
  NCCL operation starts.
- Decision implementation: The control plane now tracks in-flight
  `(source_rank, block_id)` reservations. Global rebalance planning excludes
  reserved move and copy candidates, plan enqueue atomically claims all source
  blocks, and every completion/failure/worker-down path releases the claims.
  A participating worker failure now aborts the whole transfer transaction
  immediately instead of allowing the remaining rank to complete a partial
  plan.
  Data-plane prepare verifies that every planned source block remains locally
  allocated. `release_blocks()` now reports an explicit stale-allocation error
  instead of leaking a set `KeyError`. Added concurrent-planning regression
  coverage.
- Decision result: Overlapping foreground plans cannot transfer or release the
  same physical source block, eliminating this crash and the associated NCCL
  deadlock risk.

## 2026-07-16: Resolve Models from Snapshot Metadata

- Decision demand: Offline benchmark execution passed a Hugging Face snapshot
  directory whose basename is a commit hash, causing model selection based on
  `Path(...).name` to reject a valid cached Qwen3 checkpoint.
- Decision plan: Identify local models from checkpoint metadata while retaining
  compatibility with repository IDs used by online execution.
- Decision implementation: Added a model-family resolver that reads a local
  `config.json` and recognizes its `architectures` or `model_type` fields. It
  falls back to an explicit `model_architecture` value and then the original
  model identifier. `ModelRunner` now selects Qwen3 or Llama through this
  resolver, with tests for both repository IDs and hash-named local snapshots.
- Decision result: `HF_HUB_OFFLINE=1` runs can use a cached snapshot path
  directly without renaming it or contacting Hugging Face.

## 2026-07-17: Route by Longest Contiguous Prefix and Incremental Capacity

- Decision demand: The locality benchmark showed that routing improved mean
  performance, but route/local hit rates remained low and LMPool still issued
  many low-value foreground transfers. Routing queried only the terminal full
  block hash and treated the entire prompt as new allocation even after a
  prefix hit.
- Decision plan: Make reusable KV length and incremental allocation demand the
  shared basis for routing, capacity checks, reservations, and metrics. Suppress
  repeated structural transfer failures without adding a workload-specific
  policy.
- Decision implementation: `ControlPlaneClient` already sends the cumulative
  full-block hash chain; `GlobalScheduler` now looks up every hash and retains
  each GPU's longest chain contiguous from block zero. Route scoring uses that
  block count, while admission checks and optimistic global reservations use
  `num_blocks - matched_prefix_blocks`. Route-cache validation follows the same
  chain semantics. The synthetic single-GPU hit measurement now marks KV ready
  and releases request references before the next lookup. Foreground transfer
  exposes its last failure reason to `Scheduler`, which applies bounded
  exponential cooldown for `no_plan`, `no_target_space`, and `stale_source`,
  resetting immediately after success.
- Decision result: Shared prefixes remain discoverable when request suffixes
  differ, cached blocks are no longer double-counted as required capacity, and
  locality traffic no longer creates sustained retries against unchanged
  transfer state. Unit and control-plane tests cover partial-chain hits,
  non-contiguous rejection, incremental reservation, ready-KV metrics, and
  structural-failure backoff.

## 2026-07-17: Preserve Prefix Chains and Equalize KV Capacity

- Decision demand: The locality comparison mixed two independent effects.
  Ordinary block-level LRU could evict an early ancestor before newer suffix
  blocks, leaving globally visible hashes that could not form a reusable prefix
  from block zero. Routing and transfer scenarios also accepted different KV
  block limits, so policy effects were not isolated under equal memory.
- Decision plan: Make prefix-chain validity a hard eviction constraint, retain
  recency only as the ordering policy among valid victims, and expose one
  canonical per-rank block budget for every benchmark scenario. Add diagnostics
  that separate workload potential, routing-time matches, worker reuse, stale
  routing decisions, and actual runtime capacity.
- Decision implementation: Added `parent_hash` and `prefix_depth` to local
  blocks and propagated parent metadata through worker block-state snapshots,
  the authoritative global page table, move/copy plans, and transfer-in block
  registration. Local reclamation and globally reported eviction candidates
  now contain only unreferenced KV-ready leaves; repeated eviction peels a chain
  from the deepest eligible leaf, with LRU ordering across independent leaves.
  All blocks touched by one completed request receive the same recency timestamp
  so per-block loop timing cannot make block zero look older than its suffix.
  Added `--kv-block-budget`, applied it to all five scenarios, rejected
  conflicting legacy routing/transfer budgets, and reported worker-resolved
  `max_cached_blocks` per rank. The benchmark now reports an unbounded workload
  theoretical prefix-hit upper bound, route matched-block ratio, and stale-route
  rate separately from initial worker local hits and cached-token ratio. Added
  chain and shared-ancestor regression tests plus a small-budget metric test.
- Decision result: Capacity comparisons now hold requested KV memory constant,
  and eviction cannot preserve unusable suffix hashes while discarding their
  required ancestors. The ranking layer remains leaf-LRU, providing a correct
  baseline for later leaf-LFU or TinyLFU admission-policy ablations without
  changing prefix-chain safety.

## 2026-07-17: Disambiguate Control-Plane and Data-Plane Prefix Metrics

- Decision demand: Benchmark labels `route`, `owner`, `local`, and
  `cached tokens` mixed decision-time and execution-time observations and did
  not state whether their denominator was requests, blocks, or tokens.
- Decision plan: Preserve JSON field compatibility while making every console,
  figure, and documentation label identify its plane and unit of aggregation.
- Decision implementation: Renamed visible metrics to `CP req hit`, `CP owner`,
  `CP blk match`, `CP stale`, `DP req hit`, and `DP tok reuse`; renamed the
  figure panel to `Prefix Reuse Metrics`; documented each existing JSON field,
  denominator, and the load-bypass case where CP owner selection can be lower
  than the control-plane request-hit rate.
- Decision result: A result now distinguishes routing knowledge from worker
  cache reality and binary request hits from the amount of prefill work
  actually avoided, without invalidating existing JSON consumers.

## 2026-07-17: Enforce Decode Page-Boundary Capacity Checks

- Decision demand: The first `load-skew` trial crashed in
  `BlockManager.append()` with an empty free-block deque during decode.
  `Scheduler.postprocess()` had already appended the sampled token, but
  `can_append()` checked the pre-append boundary condition and therefore
  approved a sequence that actually needed a new KV page.
- Decision plan: Align the capacity predicate with the Sequence lifecycle and
  make the low-level append primitive fail explicitly if a caller bypasses the
  predicate. Cover both one-sequence and same-batch multi-sequence boundaries.
- Decision implementation: Changed `can_append()` to require a free block when
  `num_tokens % block_size == 1`, matching `append()` and the fact that the new
  token is already present. Added a descriptive runtime error before accessing
  an empty deque. Added BlockManager tests for successful and blocked boundary
  growth and a Scheduler test where two sequences cross a boundary with only
  one free block; the second sequence now follows controlled preemption rather
  than crashing. Updated an old scheduler test that encoded the previous
  off-by-one behavior.
- Decision result: Decode page growth and its capacity check now use the same
  state transition. KV exhaustion is handled by the scheduler's normal
  reclaim/transfer/preemption policy and cannot surface as `IndexError`.

## 2026-07-17: Make Foreground Transfer Preserve a Usable Prefix Chain

- Decision demand: Existing locality and load-skew runs did not demonstrate
  transfer value. Foreground requests were attempted but had zero successful
  data movement, background transfer was explicitly disabled, and moving only
  a leaf could not create a prefix reusable from block zero at the target.
- Decision plan: Keep foreground and background semantics separate. Make a
  foreground capacity plan transfer the complete missing root-to-leaf fragment,
  release only cold leaf victims, and benchmark it with deterministic cache
  warm-up, source-side pressure, and reuse phases under an equal KV budget.
- Decision implementation: `GlobalBlockManager` now tracks pinned physical IDs
  and reconstructs root-to-leaf chains. `GlobalScheduler.plan_rebalance()`
  selects leaf victims by LRU, includes missing ancestors once per target,
  supports branches sharing one planned ancestor, and records exactly which
  leaves may be released. Data-plane prepare validates only release candidates
  for pinning; execute sends the complete fragment, retains copied ancestors,
  and releases selected leaves. For `memory-skew` only, `Scheduler` attempts
  this chain-preserving transfer before local cache reclamation and falls back
  to reclamation on failure. The benchmark now uses three phase barriers and
  reports sent, retained, released, and chain-plan counts plus reuse-phase
  request and token hit ratios. Added chain, shared-ancestor, phase-construction,
  and transfer-before-reclaim regression tests.
- Decision result: Foreground success now means an executable transfer both
  freed source capacity and installed a structurally reusable target prefix.
  Background transfer remains independently controlled and is absent when
  `--disable-background-copy` is set. The full CPU test suite passes; the next
  GPU experiment can directly validate capacity relief and reuse benefit with
  the new diagnostics instead of inferring them from aggregate throughput.

## 2026-07-17: Equalize Memory-Skew Placement Across Baselines

- Decision demand: The first three-phase memory-skew result assigned warm-up
  and pressure traffic to ranks 0, 2, and 4 for topology-aware scenarios, but
  assigned all of that traffic to rank 0 for `multi-gpu`, because the baseline
  intentionally had no `nvlink_topo` configuration. Its throughput, latency,
  and aggregate prefix metrics therefore described a different workload.
- Decision plan: Separate benchmark traffic placement from the topology exposed
  to an engine policy. Use the same source ranks for every multi-GPU scenario
  without enabling topology-aware routing or transfer in a baseline.
- Decision implementation: The benchmark now derives source ranks once from
  the command-line NVLink pairs and writes them to a benchmark-only placement
  field on all configurations. The three-phase runner resolves that explicit
  field before consulting engine topology. `single-gpu` remains fixed to rank
  0. Per-rank output now separates warm-up, pressure, and reuse submissions.
  Added a regression test for a six-rank topology-blind baseline using source
  ranks 0, 2, and 4.
- Decision result: All multi-GPU scenarios now execute the same warm-up and
  pressure placement, while only global-pool scenarios receive NVLink topology
  for policy decisions. Results produced before this fix remain useful for
  validating transfer mechanics, but not for baseline performance ranking.

## 2026-07-17: Release the Maximum Safe Prefix Suffix per Transfer

- Decision demand: A fair memory-skew run showed two successful foreground
  plans but no reuse benefit, while most plans returned `no_plan`. A complete
  linear chain consumed several target blocks but the planner credited and
  released only its leaf, making capacity relief much smaller than transfer
  cost and preventing plans for multi-block shortages.
- Decision plan: Compute source capacity relief from the complete prefix
  dependency graph. Release as many transferred or already-target-resident
  blocks as the current shortage requires, but retain any ancestor still needed
  by an untransferred branch. Add a value diagnostic that identifies whether
  transferred hashes belong to the warm-up hotspot.
- Decision implementation: `GlobalScheduler` now builds the source child graph
  from parent hashes, computes a deepest-first dependency-safe release order,
  and accepts a plan based on released capacity rather than sent-block count.
  Linear chains can release a multi-block suffix; shared ancestors are released
  only when all source children are also safe. Target-resident ancestors may be
  released without being sent again. Data-plane execute sends every transfer
  before releasing the union of source blocks, reports sent-retained-released
  counts without assuming release is a subset of send, and control-plane
  inflight ownership covers both sent and release-only blocks. The benchmark
  classifies transferred hashes against the common warm-up prefix and reports
  `hot sent` and `hot ratio`. Added tests for linear suffix release,
  target-resident ancestors, shared branches, and cumulative hash diagnostics.
- Decision result: One chain transfer can now relieve the actual multi-block
  shortage without breaking another prefix branch, reducing structural
  `no_plan` failures and exposing whether capacity relief preserved the KV used
  by the reuse phase. GPU performance impact remains to be measured with the
  updated memory-skew run.

## 2026-07-17: Make Foreground Transfer KV-Heat-Aware

- Decision demand: The fair three-phase memory-skew run proved that foreground
  transfer could send blocks and release source capacity, but `hot sent` stayed
  at zero and reuse did not improve. The worker owned real cache accesses, while
  the control plane replaced every evictable block timestamp with one snapshot
  time and had no access-frequency signal, so candidate value was effectively
  lost.
- Decision plan: Repair the existing state path instead of adding another
  scheduling layer. Keep per-block frequency and recency at the worker, publish
  them with each block-state snapshot, preserve them in the global block
  manager, and use one LFU-first policy consistently for local reclamation and
  foreground chain selection.
- Decision implementation: Added `access_count` to local blocks and increment
  it only on real cache hits; release updates recency without double-counting.
  Data-plane block-state reports now include each ready block's monotonic access
  time and frequency. `GlobalBlockManager.update_gpu_state()` preserves those
  values instead of assigning one synthetic timestamp, removes stale metadata,
  and carries frequency through move/copy records. Local reclamation orders
  dependency-safe leaves by frequency then recency. Foreground planning orders
  complete chains by `frequency * chain_length / missing_target_blocks`, uses
  recency as a tie-breaker, and sends source access counts so transferred blocks
  retain their heat at the target. Added regression tests for state propagation,
  LFU reclamation, and hot-chain selection.
- Decision result: The planner can now distinguish repeatedly reused warm-up KV
  from newer one-shot pressure KV. CPU regression tests validate the metadata
  and selection path; the next foreground-only GPU run must confirm non-zero
  hot-transfer ratio and improved reuse before background transfer is enabled.

## 2026-07-17: Include Decode Headroom in Foreground Transfer Demand

- Decision demand: The foreground-only memory-skew run transferred blocks but
  never selected the warm-up hotspot. Admission rejected requests using prompt
  blocks plus one decode-growth reserve, while foreground transfer calculated
  demand from prompt blocks alone. When the prompt fit but its decode reserve
  did not, transfer was skipped and local reclamation discarded reusable KV.
- Decision plan: Use one shortage definition throughout prefill admission and
  foreground transfer. Preserve the existing fallback: execute local cache
  reclamation only when transfer is disabled, below threshold, on cooldown, or
  fails to provide enough capacity.
- Decision implementation: `Scheduler.schedule()` now passes the complete
  admission deficit, including decode-growth headroom, to foreground transfer.
  The subsequent local reclaim remains after the transfer attempt. Added one
  regression test where the prompt fits but decode headroom is missing and a
  second test that records transfer-before-reclaim ordering on plan failure.
- Decision result: Scheduler tests confirm that the headroom-only shortage now
  requests one block of transfer, successful transfer preserves local cache,
  and failed transfer still admits the request through local reclamation. The
  updated memory-skew GPU benchmark must validate higher `hot sent` and reuse.

## 2026-07-17: Isolate Repeated Benchmark Rendezvous Stores

- Decision demand: A six-rank, three-repetition benchmark failed during the
  third `multi-gpu-kv-transfer` trial because rank 0 could not bind the selected
  TCPStore port. The previous free-port helper released its probe socket before
  workers started, leaving a race in which another local process could acquire
  the same port.
- Decision plan: Remove TCP port allocation from the single-node benchmark
  lifecycle without changing NCCL data transfer. Give every scenario trial a
  unique rendezvous resource and clean it after all workers exit.
- Decision implementation: Added `prepare_benchmark_rendezvous()`, which copies
  the scenario configuration and assigns a process- and UUID-specific `file://`
  rendezvous path when no explicit init method is supplied. `run_engine_scenario()`
  deletes that path after `engine.exit()`. Explicit rendezvous methods remain
  unchanged. Added tests for uniqueness and explicit-method preservation, and
  documented that FileStore handles startup while NCCL remains the transfer
  backend.
- Decision result: Repeated local trials no longer perform a vulnerable
  probe-close-rebind TCP sequence, so `EADDRINUSE` cannot arise from benchmark
  rendezvous allocation. Targeted benchmark and scheduler tests pass; the GPU
  benchmark should be rerun from the failed command.

## 2026-07-17: Route on Effective Rather Than Immediately Free Capacity

- Decision demand: The corrected memory-skew run achieved perfect reuse in
  LMPool but routed the entire reuse phase to only ranks 1, 3, and 5. Source
  ranks 0, 2, and 4 were idle because global routing required immediately free
  blocks, even though their one-shot pressure cache was locally reclaimable.
  Round-robin admitted the same requests by reclaiming that cache, so the
  control-plane approximation suppressed half of the available parallelism.
- Decision plan: Make global admission match Local Block Manager semantics.
  Compute effective capacity as current free blocks plus the maximum
  dependency-safe leaf-first reclamation, protect blocks matched by the incoming
  prefix, exclude pinned descendants, and prevent concurrent routes from
  promising the same capacity.
- Decision implementation: `GlobalBlockManager` now reconstructs reclaimable
  capacity from ready block hashes, parent links, and pinned IDs. It exposes
  effective-capacity checks and tracks optimistic block reservations by GPU and
  sequence. `GlobalScheduler` uses those checks for no-hit selection, prefix
  owners, load bypass, and capacity fallback, and returns free, reclaimable,
  effective, and `uses_reclaimable_capacity` diagnostics. The control plane
  reserves required new blocks by sequence; the target data plane releases the
  reservation only after first prefill writes KV and publishes a fresh block
  snapshot. Source blocks in an inflight transfer plan also block themselves
  and their ancestors from reclaimable-capacity accounting until the plan
  completes or aborts. Prefix blocks matched by a routed but uncommitted request
  receive the same temporary protection, preventing a concurrent no-hit route
  from reclaiming the first request's promised KV. The benchmark reports the
  resulting `CP reclaim` route rate and separates warm-up, pressure, and reuse
  mean/P90 TTFT and E2E, so pressure-tail latency cannot hide reuse-stage
  benefit. Added tests for chain-safe reclamation, pinned and protected
  descendants, stale snapshots, reservation overcommit, and load bypass to a
  free-zero rank.
- Decision result: CPU tests verify that an idle source with no immediate free
  blocks but enough reclaimable pressure cache is now a valid load-bypass
  target, while active prefix chains and concurrent reservations remain safe.
  The next six-GPU memory-skew run must confirm reuse traffic returns to source
  ranks and LMPool GPU utilization approaches the multi-GPU baseline.

## 2026-07-17: Unify Routing and Foreground Transfer Economics

- Decision demand: Effective-capacity routing restored traffic to all six
  ranks, but the memory-skew benchmark still trailed multi-GPU by 2.98% in
  throughput and 4.05% in mean E2E latency. Transfer raised reuse-phase token
  reuse from 85.40% to 89.82%, yet 33 transferred blocks cost more than the
  avoided prefill. The fixed load-bypass threshold treated reclaimable capacity
  as free and did not charge cold targets for missing-prefix recomputation.
- Decision plan: Compare every route in one token-equivalent cost domain and
  execute foreground transfer only when frequency-predicted saved prefill
  exceeds calibrated transfer cost by a safety margin. Keep effective capacity
  as an admission constraint, not as evidence that a route is cheap.
- Decision implementation: `GlobalScheduler` now computes per-candidate cost as
  token-aware queued work plus missing-prefix tokens times
  `route_prefill_cost_weight`, plus reclaimed blocks times block size and
  `route_reclaim_cost_weight`. Prefix-owner selection, load bypass, full-owner
  fallback, no-hit routing, and the route-cache fast path use this model; route
  metadata reports queue, prefill, reclaim, and total components. Foreground
  plans report estimated transfer cost, saved prefill, and benefit ratio, and
  return `low_benefit` below
  `foreground_transfer_min_benefit_ratio`. That reason uses structural-failure
  cooldown so the scheduler does not retry an uneconomic plan every cycle. The
  benchmark exposes all cost weights and prints a `low value` failure counter.
- Decision result: The complete CPU suite passes with `115 passed, 1 skipped`,
  and compile/diff validation is clean. Added regressions proving a moderately loaded owner
  retains a long prefix, a cold transfer is rejected, and a sufficiently hot
  transfer remains executable. The six-GPU memory-skew benchmark must now show
  fewer foreground transfers and `low value > 0`; throughput should approach
  multi-GPU while retaining reuse-phase benefit.

## 2026-07-17: Replace the Saturating Single-Prefix Memory-Skew Trace

- Decision demand: The post-cost-model run reduced transferred blocks from 33
  to 25 and rejected five low-value plans, but LMPool still trailed multi-GPU
  throughput by 4.17%. Routing-only and multi-GPU had exactly the same 90.63%
  reuse request hit and 85.40% token reuse. The policy changed, but the trace
  still could not expose its value.
- Decision plan: Check whether the baseline can learn the trace locally before
  changing the scheduler again. Construct a fair trace where all scenarios see
  identical requests, placement, KV budget, and phase barriers, but where
  preserving or routing each prefix remains useful beyond one cold request.
- Decision implementation: Replaced memory-skew's single hot prefix with an
  automatically sized set of up to 15 long hot prefixes. Warm-up repeats each
  group on a deterministic source rank, pressure uses unique prefixes at half
  the hot-prefix length, and reuse interleaves all hot groups. The automatic
  group count is odd to avoid alignment with even-sized round-robin GPU cycles;
  for the six-GPU, 128-request trace it chooses 15. Repeated warm-up hashes,
  rather than the intersection of the first two requests, now define hot
  transfer blocks. Added `--memory-skew-prefix-groups` and tests for phase
  construction, automatic sizing, and invalid values.
- Decision result: In the old one-prefix trace, round-robin needed only one miss
  per partner before its remaining requests hit locally, mathematically capping
  routing's opportunity at roughly three requests. In the new 15-prefix trace,
  the same 64-request reuse phase visits 30 distinct `(prefix, rank)` pairs;
  even after warm-up placement, idealized round-robin request reuse is about
  60.94%, leaving measurable room for routing and transfer. The repository's
  Qwen tokenizer yields eight blocks per hot prefix and four per pressure
  prefix; each source receives about 40 hot plus 44 pressure blocks against the
  common 64-block budget, so the trace creates real capacity pressure. The full
  CPU suite passes with `116 passed, 1 skipped`; the GPU run must validate
  actual values.

## 2026-07-17: Preserve Routed Prefix Promises Through Local Admission

- Decision demand: The 15-prefix memory-skew trace reported roughly 40% control-plane
  route hits but only 10% reuse-phase data-plane request hits. More than 73% of
  route hits were stale by prefill time, so routing selected valid owners but
  local admission reclaimed their promised KV while multiple routed requests
  waited in the worker queue.
- Decision plan: Carry the matched prefix identity with each routed request and
  make local reclamation honor all outstanding route promises. Keep the change
  within existing routing and reclamation boundaries instead of adding another
  score or cache policy.
- Decision implementation: `ControlPlaneClient.route_sequence()` now records
  the matched cumulative prefix hashes on `Sequence.routed_prefix_hashes`, and
  Sequence multiprocessing serialization preserves that field. Before local
  admission reclaims cache, `Scheduler` resolves the promised hashes of every
  waiting request to ready local block IDs and passes them to
  `BlockManager.reclaim_for_sequence()`. The current request's naturally
  matched chain and all queued route promises are protected until allocation;
  once allocated, normal block reference counts provide protection. Added
  serialization, block-manager, scheduler, and control-plane regressions.
- Decision result: Focused CPU tests verify that admitting one request cannot
  evict the prefix promised to the next queued request. The next GPU run should
  reduce `stale route` substantially and bring reuse request/token hit rates
  closer to control-plane route coverage.

## 2026-07-17: Allow Hybrid Transfer of Pinned Prefix Chains

- Decision demand: The same memory-skew run executed zero transfers and all 36
  foreground attempts failed as `no_plan`. The planner rejected an entire
  prefix chain whenever any ancestor was pinned, although completed KV prefix
  blocks are immutable and only the source release operation is unsafe for a
  pinned block.
- Decision plan: Permit pinned ancestors to be copied as destination
  dependencies while retaining the existing rule that pinned source blocks can
  never be released. Continue excluding blocks owned by another inflight plan.
- Decision implementation: Relaxed chain candidate validation in
  `GlobalScheduler._select_chain_move_candidates()` so pinned ancestors may be
  included in transfer payloads. `_dependency_safe_release_order()` remains
  authoritative for source reclamation and filters all pinned blocks, producing
  a hybrid plan that copies dependencies and moves only an unpinned leaf or
  suffix. Added a regression with a pinned root and evictable child.
- Decision result: Unit coverage verifies that the complete two-block chain is
  sent while only the unpinned leaf is released. The next memory-skew run should
  replace at least some `no_plan` failures with successful foreground transfers;
  profitability checks may still reject low-value chains.

## 2026-07-18: Make Route-Promise Protection Progress-Safe

- Decision demand: The first six-GPU rerun after adding waiting-prefix
  protection stopped making progress in the routing-only trial. With a submit
  window of 16 and a 64-block budget, queued route promises could collectively
  protect every cached block, leaving the FIFO head unable to reclaim its
  admission shortage.
- Decision plan: Treat queued route promises as priorities rather than permanent
  pins. Preserve all promises when capacity permits, but guarantee forward
  progress by allowing later FIFO promises to be reclaimed when they block the
  head request.
- Decision implementation: Local prefill admission now performs two-stage
  reclamation. The first pass protects matched blocks for every routed waiting
  request. If the head still cannot admit, a second pass retains the existing
  per-sequence protection for the head but removes additional tail protections.
  Added a scheduler regression where all cached blocks are promised to queued
  requests and verified that the head still reaches `RUNNING`. The end-to-end
  benchmark now emits one compact progress line every 30 seconds and reports
  each trial's elapsed time, distinguishing a long model run from a stalled
  admission loop without enabling verbose worker logs.
- Decision result: Route promises remain stable in the common case while an
  overcommitted waiting window can no longer create an admission livelock.
  Subsequent GPU validation should complete the routing trials; some stale tail
  hits remain acceptable under true overcommit and should be reported rather
  than hidden.

## 2026-07-18: Correct E2E Progress Timer Initialization

- Decision demand: The first run with compact progress reporting failed in the
  single-GPU scenario with `UnboundLocalError: last_progress_report`. A textual
  patch had initialized the timer in an unused independent-baseline helper
  rather than in `run_engine_scenario()`.
- Decision plan: Keep progress reporting but colocate all timer state in the
  function that owns the reporting loop.
- Decision implementation: Removed the misplaced initialization and initialized
  `last_progress_report` immediately after `run_engine_scenario()` records its
  `start_wall` value.
- Decision result: The progress branch now has initialized state for every E2E
  scenario, including the single-GPU baseline.

## 2026-07-18: Enforce the Physical KV Budget Used by Memory-Skew

- Decision demand: The first completed 15-prefix run requested 64 blocks per
  rank but every worker reported an actual capacity of 14. The benchmark's 5%
  GPU-memory fraction constrained ModelRunner before the configured block cap,
  turning the intended 64-block experiment into an unreported 14-block stress
  test. Data-plane reuse fell to zero in the baselines, all foreground transfer
  plans failed, and the resulting comparison did not represent the requested
  experiment.
- Decision plan: Give an explicit KV budget strict benchmark semantics and fail
  before request submission when workers cannot realize it. Raise the benchmark
  memory fraction enough for the requested 64-block cap while retaining that
  cap as the actual allocation bound.
- Decision implementation: Added `--gpu-memory-utilization` with a benchmark
  default of 0.20. Every scenario now waits for each worker's startup capacity
  report before submitting prompts. When `--kv-block-budget` is explicit, an
  actual capacity below it raises an actionable error containing requested and
  realized blocks plus the memory fraction. Per-rank capacity is retained in
  result diagnostics. Routed-prefix protection now also covers decode-growth
  reclamation, with the same progress-safe fallback used by prefill admission.
- Decision result: A formal `--kv-block-budget 64` run can no longer silently
  execute with 14 blocks. The next smoke run must either confirm 64 blocks on
  every rank or stop before workload execution; only the former is suitable for
  evaluating the 15-prefix trace.

## 2026-07-18: Admit Transfers by Measured Time and Batch KV Payloads

- Decision demand: The 15-prefix memory-skew run created real transfer opportunities:
  LMPool raised reuse-phase request hit rate from the multi-GPU baseline's 60.94%
  to 98.44%, but throughput did not improve and reuse P90 latency regressed. The
  existing planner summed historical access counts for every block in a prefix
  chain and compared token-equivalent values, while the data path issued
  `2 * layers * blocks` blocking NCCL operations. It therefore overestimated
  future reuse and underestimated actual transfer/interference cost.
- Decision plan: Make zero transfer a valid outcome unless expected wall-clock
  savings exceed measured end-to-end cost. Remove chain-length double counting,
  use conservative future demand, batch communication, and expose enough runtime
  measurements to calibrate the model rather than tuning thresholds blindly.
- Decision implementation: `GlobalScheduler.plan_rebalance()` now computes KV
  bytes from model shape and estimates transfer milliseconds from effective
  GiB/s, fixed protocol latency, and an inference-interference multiplier. It
  estimates saved prefill milliseconds from the least-frequent transferred
  chain block, subtracts the already observed access, applies a future-reuse
  discount, and admits only plans meeting the configured benefit ratio. The
  transfer primitive now packs all K/V blocks of one layer into one contiguous
  tensor, reducing blocking P2P calls from `2 * layers * blocks` to `layers`;
  destination writes use indexed scatter into physical blocks. Data-plane and
  E2E results now report actual bytes, source-side transfer time, effective
  GiB/s, estimated cost, and estimated saved prefill time. Added CPU pack/unpack
  correctness and conservative chain-demand tests, and exposed five calibration
  parameters in the benchmark CLI.
- Decision result: Static compilation and the complete CPU suite pass with
  `123 passed, 1 skipped`; the skipped case is the opt-in NCCL integration test.
  GPU validation should first calibrate effective bandwidth, then verify that
  transfer diagnostics show `estimated saved > estimated cost` for admitted
  plans. On natural workloads where this condition is absent, LMPool should
  execute zero transfers and converge toward routing-only performance rather
  than preserving cache at a net loss.

## 2026-07-18: Isolate Serving Time and Close the Transfer Cost Feedback Loop

- Decision demand: The calibrated memory-skew run showed useful routing tail
  gains but almost no incremental transfer gain. Two admitted plans estimated
  29.97 ms total transfer cost while source workers blocked for 337--374 ms;
  E2E effective bandwidth was only 2.0--2.27 GiB/s despite a warmed primitive
  reaching 10.9--18.7 GiB/s. Benchmark throughput also started before worker
  capacity reports, so it included model loading, warmup, NCCL initialization,
  and KV allocation. Transfer-only reused round-robin placement, which did not
  guarantee that a future request consumed a moved prefix.
- Decision plan: Fix measurement before changing policy. Move startup work
  outside serving metrics, initialize every configured P2P pair before ready,
  feed source-observed cost back to admission, and make memory-skew expose a
  deterministic cross-pair reuse opportunity shared by its baseline and
  transfer-only variants.
- Decision implementation: `run_engine_scenario()` now starts wall-clock
  throughput and GPU sampling only after all workers report realized KV
  capacity and workload metadata is prepared. Data-plane startup calls
  `prewarm_p2p_pairs()` for every normalized NVLink pair; the initial version
  used a validated CUDA marker before worker readiness. Source
  execute results return actual payload bytes and blocking time through
  `rebalance_done`. `GlobalScheduler.observe_transfer()` converts excess time
  over configured wire time into an EWMA, and subsequent plans use the maximum
  of static and observed cost. In memory-skew, warmup/pressure remain on source
  ranks while topology-blind and transfer-only reuse requests are placed on
  the corresponding NVLink partners. Added pair normalization, target mapping,
  online-cost, and optional NCCL prewarm coverage.
- Decision result: Static compilation and the complete CPU suite pass with
  `126 passed, 1 skipped`; the optional integration test now covers both P2P
  prewarm and batched KV validation. The next GPU run should report
  serving-only throughput, remove first-use communicator setup from transfer
  timing, and automatically increase `low_benefit` rejections if measured E2E
  overhead remains above the static estimate.

## 2026-07-18: Serialize Transfer Admission and Calibrate Each NVLink Pair

- Decision demand: The serving-only memory-skew result improved measurement
  validity, but transfer remained incremental: transfer-only exceeded the
  multi-GPU throughput by only about 2.6% in one run and slightly regressed
  mean TTFT. Two plans estimated roughly 30 ms while source workers accumulated
  277--299 ms of blocking time. Multiple first plans could be admitted before
  any completion updated the cost model, and one global EWMA could incorrectly
  mix NVLink pairs with different contention.
- Decision plan: Protect throughput and delay before increasing transfer
  frequency. Serialize foreground work per NVLink pair, calibrate and learn
  cost independently per pair, prewarm with representative KV bytes, split
  source/target timing, and batch only a complete proven-hot prefix chain rather
  than attaching unrelated cold blocks.
- Decision implementation: `control_plane_process()` now derives canonical
  transfer pairs, rejects overlapping plans as `pair_busy`, and releases the
  pair reservation on success, abort, failure, or worker loss. `GlobalScheduler`
  stores pair-specific transfer-overhead EWMA values and prices each target
  transfer separately. `prewarm_p2p_pairs()` sends configured block-shaped
  FP16 payloads for every model layer before worker readiness and reports the
  source observation to the control plane. Data-plane runtime statistics now
  separate source blocking time from target receive/register time. Prefix-chain
  selection stops once the real shortage is covered; the selected chain is
  packed per layer, but no colder chain is added merely to fill a batch.
  `GlobalBlockManager.lookup_prefix()` hides source locations marked
  transfer-inflight, preventing routing from selecting a block that a move plan
  is about to release; the destination becomes visible only through its committed
  worker block-state update.
- Decision result: The complete CPU suite passes with `128 passed, 1 skipped`;
  the skipped test requires real NCCL GPUs. The next GPU run should
  prioritize throughput, mean/P90 TTFT, and mean/P90 E2E. `pair_busy` and
  `low_benefit` are expected protective rejections; transfer is successful only
  when LMPool improves those serving metrics over routing-only, not when it
  merely increases the transfer count.

## 2026-07-18: Balance Prefix Locality with Decode Parallelism

- Decision demand: In the serving-only memory-skew run, routing reduced
  reuse-phase mean TTFT by 58.4% and P90 TTFT by 79.7%, but throughput was 2.5%
  below round-robin. Per-rank submissions were `[33, 11, 35, 6, 37, 6]`
  instead of the baseline's near-even distribution. All three foreground plans
  were correctly rejected as `low_benefit`, so transfer could not relieve the
  prefix-owner concentration.
- Decision plan: Preserve the existing transfer admission threshold. Account
  for expected decode work in routing load, cap owner/partner sequence skew,
  and make one explicit choice when an owner is overloaded: either spill the
  current request so it naturally seeds the partner, or keep it on the owner
  and create one cost-gated hot-prefix replica for future requests.
- Decision implementation: Route requests now carry `max_tokens`; optimistic
  reservations and worker block-state snapshots add configurable expected
  decode work. `GlobalScheduler` detects sequence-pressure skew only within the
  owner's NVLink pair and permits spill under a bounded extra-recomputation
  cost. When background copy is enabled and predicted repeated prefill savings
  cover pair-specific transfer cost, the scheduler instead annotates a
  `prefix_hit_replica_copy` decision and retains the current request locally.
  The control plane copies the ordered prefix chain only after the hotness and
  load-skew gates pass. A direct spill suppresses copy because that request will
  materialize the same KV on the partner. Benchmark JSON and diagnostics add
  `pair_spill_count` and `replica_copy_route_count`.
- Decision result: The complete CPU suite passes with `130 passed, 1 skipped`.
  The next GPU comparison must run with background copy enabled for LMPool and
  report throughput plus mean/P90 TTFT and E2E. A useful combined result should
  show a more even per-rank distribution than the previous 9.75x max/min skew,
  while `background_copy_success` remains much smaller than routed requests.

## 2026-07-18: Move Background Transfer Before Reuse Admission

- Decision demand: The `e2e_202607181801_memory_skew` run copied seven hot
  blocks in one successful plan, but LMPool remained 2.4% below routing-only
  throughput and changed routing-only P90 TTFT by only 0.2%. All copied blocks
  came from rank 2 to rank 3 after reuse routing had already started; the other
  two NVLink pairs copied nothing. The implementation counted route hits rather
  than completed block accesses, dropped candidates while a pair was busy, and
  assumed four future reuses even when only one or two requests remained.
- Decision plan: Turn background transfer into proactive cache placement. Use
  worker-owned access snapshots to identify maximal hot prefix chains, combine
  them with demand for requests already visible at ingress but not submitted,
  preserve candidates in one FIFO per NVLink pair, dispatch only at low queue
  pressure, and let a phase boundary wait for accepted work while charging that
  time to benchmark elapsed.
- Decision implementation: `GlobalBlockManager.get_hot_prefix_chains()` now
  reconstructs ordered parent chains and emits only deepest hot leaves, avoiding
  redundant ancestor plans. `control_plane_process()` maintains persistent
  candidate maps and per-pair queues, validates source residency and target
  space at dispatch, serializes each pair, retries queued work after every plan,
  and records queued/dispatched/completed/drop-reason lifecycle counters. The
  benefit model uses remaining prefix counts supplied by ingress, capped by
  `background_copy_expected_reuses`; without a forecast it uses discounted
  observed accesses. Route-triggered copy no longer keeps the current request
  on an overloaded owner. `ControlPlaneClient.flush_background_copies()` adds a
  synchronous placement boundary. The memory-skew benchmark computes hashes for
  the unsubmitted reuse phase, flushes after warm-up and pressure, includes the
  wait in serving time, and publishes planner lifecycle statistics in JSON and
  transfer diagnostics. Tests now cover maximal-chain discovery, access-count
  hotness, ordered transfer, and forecast-driven flush completion.
- Decision result: Static compilation succeeds and the complete CPU suite
  passes with `132 passed, 1 skipped`; the skipped test is the opt-in NCCL GPU
  integration case. GPU acceptance now requires all configured NVLink pairs to
  show proactive placement before reuse, `place done` to match dispatched plans,
  and LMPool to preserve routing TTFT gains without falling below routing-only
  throughput after placement time is included.

## 2026-07-18: Bound Placement Planning and Measure Real Compute Savings

- Decision demand: In `e2e_202607181932_memory_skew`, LMPool completed only
  two placement plans but queued 33,049 candidates and rejected 33,027 as
  `low_benefit`. Transfer-only improved reuse request hit from 76.6% to 87.0%
  but reduced throughput by 6.8%. Existing diagnostics counted whole prompts
  as prefill work and priced only source transfer time, so the planner both
  retried unchanged decisions and compared incomplete cost/benefit values.
- Decision plan: Bound control-plane work by the number of distinct candidate
  states, expose actual uncached model work and phase-boundary waiting, diagnose
  every NVLink pair independently, suppress immediate reciprocal placement,
  and make later admission learn complete dispatch-to-commit cost.
- Decision implementation: `control_plane_process()` now memoizes stable
  `low_benefit` and `no_target_space` results using prefix chain, source/destination,
  effective predicted reuse, and destination free capacity. Access-count or
  ingress-demand changes invalidate the memo only when they change effective
  predicted reuse. Identical
  snapshots increment `skipped_negative_cache` without requeueing; changed
  demand or capacity invalidates the memo. A canonical leaf/pair cooldown
  prevents an immediate reverse copy while page-table snapshots converge.
  Candidate lifecycle counters are maintained globally and per NVLink pair.
  Successful background plans report dispatch-to-commit elapsed time to
  `GlobalScheduler.observe_placement()`, and the larger of static, source-side,
  and full-placement pair costs controls later admission. Data-plane runtime
  reports now separate prompt, cached, and uncached prefill tokens and measure
  through sampled-token consumption rather than asynchronous kernel enqueue.
  The E2E benchmark records placement wait separately, preserves it in total
  elapsed, aggregates the new metrics across repetitions, and prints prefill
  compute and per-pair placement diagnostic tables.
- Decision result: Static compilation succeeds and the complete CPU suite
  passes with `134 passed, 1 skipped`; the skipped test requires opt-in NCCL
  GPUs. The next GPU acceptance run must
  show candidate `evaluated` near the number of unique prefix/pair states,
  explain every configured pair independently, and demonstrate that reduced
  uncached prefill work exceeds measured placement wait before transfer can be
  credited with throughput or latency improvement.

## 2026-07-18: Bind Proactive Placement to Handoff Traffic

- Decision demand: `e2e_202607182047_memory_skew` still showed only 9--14
  copied blocks, one or two completed background plans, and no evidence that a
  later request consumed those replicas. The baseline self-warmed the reuse
  targets, per-layer blocking P2P launches made serving transfer slower than
  the calibrated link, candidate discovery still depended on repeated state
  updates, and the fixed prefill-time prior could reject copies without using
  measured recomputation cost. Adding more policy branches would not solve
  these missing data-path and workload contracts.
- Decision plan: Make one transfer plan one NCCL payload, calibrate that exact
  protocol, learn destination prefill cost online, create a bounded route lease
  only after the replica commits, discover placement from ingress/route events
  rather than every block snapshot, and add a two-phase workload where reuse
  must cross an NVLink pair.
- Decision implementation: `kv_transfer.py` now packs every model layer, K/V
  tensor, and selected block into one contiguous payload and executes one
  blocking send/recv per plan. Every configured NVLink pair receives a dedicated
  NCCL process group at startup, P2P prewarm uses the same all-layer layout, and
  an already prepared plan skips the legacy block-ID negotiation round trips.
  Data-plane prefill completion reports uncached tokens and elapsed time, while
  `GlobalScheduler` maintains a discounted per-rank EWMA used by background
  admission. Successful forecast copies create per-prefix placement leases in
  the control process; matching routes consume roughly half of the forecast on
  the replica when its cost is no worse than the source, preserving pair-level
  decode parallelism. Block-state messages update authoritative state without
  scanning all candidates; ingress forecasts and threshold-crossing routes are
  the discovery events. The benchmark adds `session-handoff`, which warms
  independent prefixes only on source ranks, drains accepted placement, then
  continues the same sessions on NVLink partners. JSON and console diagnostics
  add `placement_lease_route_count` / `lease route`. Tests cover all-layer
  pack/unpack, online prefill cost, exact handoff trace construction, ingress
  negative-cache behavior, and routing through a committed replica lease.
  Data-plane workers wait on ingress and control queue connections together, so
  an idle request-queue wait cannot delay a transfer command by 50 ms; the
  control plane wakes the receiver before the sender. Placement-cost learning
  blends its first dispatch-to-commit sample with the calibrated prior through
  EWMA instead of replacing the prior with one cold-start outlier.
- Decision result: The opt-in two-GPU NCCL round-trip passes. The real 28-layer,
  two-block microbenchmark validates data with 3.986 ms mean latency, 5.149 ms
  p95, and 13.72 GiB/s effective bandwidth, improving mean latency by 20.5% and
  bandwidth by 25.9% over the previous 5.017 ms / 10.90 GiB/s measurement. In a
  fixed two-GPU `session-handoff` acceptance run, LMPool completes all four
  placement candidates, records two lease-routed requests, reaches the
  workload's 75% request-hit upper bound, and reduces mean TTFT/E2E by
  14.8%/13.3% versus topology-blind multi-GPU. The tiny four-token correctness
  run improves total throughput by 2.1%; paper throughput evaluation must use
  longer decode work so pair-level parallelism amortizes the placement boundary.
  Static compilation and the complete CPU suite pass (`140 passed, 1 skipped`);
  the skipped case is the separately passed opt-in NCCL integration test.

## 2026-07-18: Batch Handoff Placement and Preserve Replica Decode Batches

- Decision demand: In `e2e_202607182346_session_handoff`, LMPool reduced mean
  TTFT/E2E by 33.7%/27.7% versus the same-run multi-GPU baseline, but its
  throughput was statistically tied with routing-only and transfer-only and
  aggregate P90 barely changed. The control plane executed 32 independent
  prefix placement plans for 224 blocks, accumulated 376 ms of placement wait,
  and produced only 17 lease routes. Code inspection showed that every prefix
  candidate paid a separate prepare/execute transaction, while each committed
  lease assigned only half its forecast demand to the replica and re-ran a
  source/target cost comparison for every request. This fragmented the reuse
  decode batch back onto the warmup source.
- Decision plan: Keep the existing two-phase protocol and all-layer payload,
  but amortize it across independent prefix candidates on the same directed
  NVLink pair. Treat the drained source phase as historical work and bind the
  complete forecast reuse batch to the committed replica. Correct placement
  admission so repeated requests do not repeatedly claim the same cold-miss
  saving, and report reuse performance separately from warmup.
- Decision implementation: `control_plane_process()` now collects up to 16
  candidates per directed pair, deduplicates shared source blocks, applies a
  128-block default batch cap, and emits one copy transfer inside one
  prepare/execute plan. Candidate lifecycle counters remain candidate based;
  new `plans_dispatched` / `plans_completed` counters expose actual protocol
  transactions. Completion creates one lease per copied leaf with `remaining`
  equal to the ingress forecast, and lease routing no longer fragments that
  batch through per-request source/target cost checks; target capacity remains
  mandatory. Background admission estimates one avoidable target cold prefill
  per unique copied block when no eviction is predicted. The E2E benchmark now
  measures phase output tokens, elapsed time, and throughput, prints phase
  throughput beside TTFT/E2E, and writes an adjacent `_reuse_phase` figure with
  throughput, mean TTFT, mean E2E, and P90 E2E. Documentation distinguishes
  prefix candidates from pair-level plans and describes the forecast-bound
  lease contract.
- Decision result: Static compilation and the complete CPU suite pass (`141
  passed, 1 skipped`); the skipped test remains the opt-in NCCL integration
  case. A two-GPU end-to-end run combines four candidates and 12 blocks into
  one completed pair plan, routes all eight reuse requests through placement
  leases, and produces an exact 8/8 warmup/reuse rank split. Reuse request hit
  reaches 100%, compared with 50% for topology-blind multi-GPU, and the new
  phase table/figure are emitted successfully. The tiny four-token run is a
  protocol acceptance test rather than a throughput result: LMPool improves
  mean TTFT/E2E by 3.2% versus multi-GPU, while the 58 ms placement boundary is
  not amortized by its short decode. The six-GPU paper run should reduce 32
  candidate placements to approximately three pair-level plans and use the
  reuse-phase metrics to quantify amortized benefit.

## 2026-07-19: Remove Per-Token Control Sync and Expose Reuse Amortization

- Decision demand: The six-GPU `e2e_202607190037_session_handoff` result was
  initially described as worse than multi-GPU, but the JSON shows otherwise:
  LMPool reaches 578.38 tok/s versus 561.54 tok/s, reduces mean TTFT from
  2.304 s to 1.756 s, and reduces mean E2E from 6.973 s to 6.594 s. Its reuse
  phase reaches 649.75 tok/s versus 560.05 tok/s and reduces reuse mean TTFT
  from 2.281 s to 1.126 s. The aggregate throughput gain remains only 3.0%
  because an equal 64-request warm-up phase cannot benefit from a replica that
  does not exist yet. Code inspection also found that `Scheduler` published a
  full local block snapshot after individual sequence/token mutations even
  though `DataPlaneProcess` already published an authoritative snapshot after
  each model batch.
- Decision plan: Remove duplicate page-table traffic at its ownership boundary,
  test whether source/replica striping can preserve cache reuse while activating
  both sides of each NVLink pair, and make the benchmark expose how many requests
  pay cache construction versus consume the transferred KV. Do not claim a
  throughput win from a tiny protocol test when its output is too short to
  amortize placement and process overhead.
- Decision implementation: `Scheduler` now tracks `local_state_dirty` and keeps
  direct reporting enabled for standalone use. `DataPlaneProcess` disables
  per-sequence reporting, sends one authoritative block-state snapshot after a
  receive/model/transfer batch, and flushes a dirty idle state once. Placement
  leases now carry explicit source and replica quotas; adjacent prefixes start
  on opposite sides so a bounded submit window does not create all-replica then
  all-source waves. The E2E benchmark adds `--handoff-warmup-prompts`; its
  default `0` preserves the old 50/50 split, while `32` with 128 requests and
  32 prefix groups creates one 32-request cache-building round followed by 96
  reuse requests. Validation and documentation were updated for this phase
  contract.
- Decision result: The final complete CPU suite passes with `144 passed, 1
  skipped`; the skipped case is the opt-in NCCL integration test. A two-GPU,
  four-token acceptance run proves
  that the explicit stripe produces an exact 4/4 reuse distribution and 100%
  reuse request hits, but it does not improve throughput: LMPool reaches 17.13
  tok/s versus 19.14 tok/s for multi-GPU because transfer and process overhead
  dominate four output tokens. This is retained as a correctness result, not a
  paper performance result. The next six-GPU acceptance run must use 32 warm-up
  plus 96 reuse requests and report both aggregate and reuse-phase confidence
  intervals; a significant system claim requires LMPool to exceed the
  multi-GPU, routing-only, and transfer-only ablations under that contract.

## 2026-07-19: Make Every Benchmark Entry an Experimental Claim

- Decision demand: The benchmark directory exposed five Python files, but
  `benchmark_e2e.py` and `benchmark_kv_transfer.py` were only 12-line wrappers
  around implementations under obsolete names. `benchmark_kv_routing.py` used
  the E2E parser, fixed multi-GPU scenarios to two ranks, ignored repetitions
  and current budget/configuration helpers, and could silently diverge from the
  routing-only ablation used in the paper. Duplicate compatibility entries did
  not add evidence and made the artifact harder to audit.
- Decision plan: Keep exactly three executable claims: routing locality,
  isolated KV transport, and full-system composition. Put the complete E2E and
  transfer implementations under their canonical names, remove obsolete entry
  files and hidden capacity aliases, and give the routing experiment its own
  constrained argument surface and three-scenario execution flow.
- Decision implementation: The full shared-prefix implementation now lives in
  `benchmarks/benchmark_e2e.py`, whose default workload is the validated
  `session-handoff` comparison. The full NCCL microbenchmark now lives in
  `benchmarks/benchmark_kv_transfer.py`. The old
  `shared_prefix_benchmark.py` / `kv_transfer_benchmark.py` files and hidden
  `--routing-max-cached-blocks` / `--eviction-max-cached-blocks` options were
  removed. `benchmark_kv_routing.py` now independently parses only locality,
  topology, common-capacity, repetition, output, and routing-cost parameters;
  honors arbitrary `--world-size`; uses the same exact per-rank KV budget in
  all three scenarios; and explicitly disables foreground transfer,
  background placement, and cache-preserving transfer. Tests and English/
  Chinese documentation now import and advertise only the canonical entries.
- Decision result: All three scripts compile and expose independent `--help`
  output. Benchmark-focused tests pass (`26 passed, 1 skipped`), and the final
  repository-wide suite passes (`148 passed, 1 skipped`). The skipped case is
  the opt-in NCCL integration test exercised separately on GPU systems.

## 2026-07-19: Harden Control Concurrency and Atomic KV Transfer

- Decision demand: Local scheduler and block-manager mutations are owned by one
  data-plane event loop, so adding coarse locks to allocation, model forward,
  or decode would add overhead without fixing the actual distributed races.
  The real hazards were protocol boundaries: a move could release its source
  before every destination acknowledged durable placement; a recycled physical
  block id could satisfy a stale transfer plan; concurrent callers sharing one
  control response queue could consume one another's replies; stale snapshots
  could overwrite newer state; and a failed worker could remain routable.
- Decision plan: Preserve single-owner data-plane execution and protect only
  cross-process state transitions. Make transfer publication atomic and
  idempotent, attach identity/version metadata to reusable state, demultiplex
  concurrent RPC responses by request id, exclude unavailable ranks, replace
  queue-emptiness guesses with event-driven draining, and serialize only
  launcher lifecycle operations. Add fault-oriented tests for every boundary.
- Decision implementation: `BlockManager` now increments a generation whenever
  a physical block is allocated, validates hash/generation before transfer,
  locks prepared source blocks against reclamation, and keeps received target
  blocks hidden until publish. Foreground and background plans carry source
  generations. `DataPlaneProcess` implements bounded idempotent
  `prepare -> execute -> publish -> finalize/abort` state: execute copies into
  hidden targets, publish establishes every destination while retaining all
  sources, finalize then releases move sources, and abort releases reservations.
  `control_plane_process` waits for all target publication and finalization ACKs
  before reporting success. Control/worker epochs, monotonic
  snapshot versions, restart-time full-state requests, and unavailable-rank
  filtering prevent stale control and page-table state from being reused.
  `ControlPlaneClient` uses a short receive lock plus per-request response
  buffers for concurrent callers, while control restart converts an in-flight
  rebalance into a counted `control_restarted` failure instead of crashing the
  worker. The data-plane loop no longer calls unreliable `Queue.empty()`, and
  `LLMEngine` serializes `step`, process recovery, and idempotent shutdown.
- Decision result: The focused control, block-manager, scheduler, data-plane,
  and launcher safety suite passes (`108 passed`). No model-forward, decode, or
  CUDA tensor access acquired a new steady-state lock; added work is limited to
  integer generation/version updates and control-message transitions. The final
  repository-wide suite passes (`159 passed, 1 skipped`); the skipped case is
  the opt-in NCCL hardware integration test.

## 2026-07-19: Consolidate the Publishable Benchmark Artifact

- Decision demand: The repository had converged to three canonical benchmark
  entries, but the execution contract was still fragmented across long source
  comments and historical commands. The transfer microbenchmark could measure
  only one payload size per process invocation and could not export JSON or a
  figure. New machine-specific results also appeared as untracked source-tree
  files, while 151 historical artifacts needed to be preserved rather than
  silently deleted.
- Decision plan: Keep one executable per paper claim, add machine-readable
  transfer sweeps, define one topology-specific paper runbook with explicit
  primary and supplementary workloads, preserve old results, and prevent new
  generated artifacts from polluting Git status. Validate the benchmark CLI,
  export contract, plotting path, and complete repository tests.
- Decision implementation: `benchmark_kv_transfer.py` now accepts a stable
  `--block-counts` sweep, runs each payload with independent NCCL processes,
  prints one comparison table, and exports JSON plus a two-panel latency/
  bandwidth figure. NCCL initialization now binds the process group to the
  explicit CUDA device in the microbenchmark, hardware test, and `ModelRunner`,
  removing rank-to-device guessing from paper logs. `PAPER_RUNBOOK.md` defines
  the current six-GPU logical topology, environment capture, CPU/NCCL tests,
  routing locality, foreground memory-skew transfer, full session-handoff, and
  supplementary load-skew commands with fixed budgets and acceptance criteria.
  `benchmarks/results/README.md` defines the output layout, and `.gitignore`
  ignores future generated result files while retaining all existing history.
  Root and benchmark READMEs link to the canonical runbook; obsolete protocol
  wording in benchmark comments was removed.
- Decision result: Transfer benchmark parser/JSON/figure tests pass, all three
  canonical scripts expose current independent `--help` output, and the final
  repository-wide suite passes (`162 passed, 1 skipped`). The skipped case is
  the opt-in NCCL hardware round-trip that the paper runbook executes explicitly
  on one visible NVLink pair.

## 2026-07-19: Make Benchmark Titles Workload-Specific

- Decision demand: The shared E2E plotting helper labeled every experiment as
  `Shared Prefix Benchmark Summary`, even when the result measured routing,
  memory-skew transfer, load skew, or session handoff. This made distinct paper
  artifacts appear to represent the same workload.
- Decision plan: Keep metrics and scenario execution unchanged, but pass an
  explicit publication-facing title through terminal summaries and all figures.
  Resolve E2E titles from a strict workload mapping and give routing-only its
  own title.
- Decision implementation: `benchmark_e2e.py` now maps each supported workload
  to a specific title and applies it to the terminal table, overview figure,
  reuse-phase figure, and per-rank diagnostics. `benchmark_kv_routing.py` uses
  `KV-Aware Routing Benchmark Summary`, and the E2E CLI description no longer
  claims every workload is a shared-prefix benchmark. Parameterized tests cover
  every title and reject unknown workloads.
- Decision result: Paper outputs now identify the experiment they actually
  measure without changing configurations, scheduling behavior, or metric
  values.

## 2026-07-19: Rebuild the Paper Suite for Dual Qwen3 Model Scales

- Decision demand: The original benchmark configuration embedded the
  Qwen3-0.6B dimensions even when `--model-name-or-path` changed, so a nominal
  Qwen3-1.7B run would either fail weight loading or measure the wrong model.
  Model execution and KV allocation also inherited process-default float32,
  while transfer prewarm and cost accounting assumed two-byte values. Existing
  JSON retained only aggregate means, making run variance and the exact model
  contract impossible to audit.
- Decision plan: Resolve every structural model field and dtype from the local
  immutable snapshot before workers start, use that same dtype for weights, KV
  storage, prewarm payloads, and transfer-byte accounting, and fail on a
  conflicting microbenchmark geometry. Preserve exact invocation metadata,
  every repeated trial, sample standard deviation, and 95% Student-t confidence
  intervals. Provide one offline runner that executes the complete claim matrix
  for both 0.6B and 1.7B and calibrates E2E transfer bandwidth from every
  configured physical NVLink pair.
- Decision implementation: `benchmark_utils.py` now maps Qwen3/Llama
  `config.json` fields into the custom runtime config, normalizes
  float16/bfloat16/float32, computes model-specific KV bytes, assigns readable
  model labels, and records command, environment, Git state, arguments, model
  metadata, and resolved configuration. E2E and routing benchmarks construct
  every scenario from that resolved config. `ModelRunner` creates model weights
  and KV tensors in the selected dtype; NCCL prewarm uses the same dtype.
  `benchmark_kv_transfer.py` derives layers, KV heads, head dimension, and dtype
  from the model and rejects explicit conflicts. Repeated E2E artifacts retain
  raw trials, sample standard deviations, and 95% CI half-widths; overview and
  reuse-phase figures draw those intervals as error bars. JSON schema v2
  separates `metadata` from `results`. `run_paper_suite.sh` preflights two local
  snapshots, runs all three physical pairs for both models, uses the median
  4-block bandwidth in each model's cost model, and stores outputs under
  model-specific directories.
- Decision result: The actual local Qwen3-0.6B snapshot resolves to 28 layers,
  8 KV heads, head dimension 128, and bfloat16 consistently. Simulated 0.6B and
  1.7B snapshot configurations, metadata/JSON contracts, confidence intervals,
  model-runner dtype selection, figure export, and benchmark parsers pass the
  focused test suite; the repository-wide suite passes (`179 passed, 1
  skipped`), with only the opt-in NCCL hardware integration skipped.
  Qwen3-1.7B is not currently present in the local Hugging
  Face cache, so the runner fails before GPU work until `MODEL_17B` points to a
  complete local snapshot; it never substitutes 0.6B or downloads implicitly.

## 2026-07-19: Preserve BF16 Through Rotary Attention

- Decision demand: Enabling model-config BF16 for the dual-model paper suite
  caused the first Qwen3 warmup to fail at the attention output projection:
  RoPE promoted Q/K and the attention output to float32 while `o_proj` weights
  remained bfloat16.
- Decision plan: Fix the dtype promotion at its source in the shared rotary
  layer, retain float32 frequency construction for numerical accuracy, store
  the precomputed cache in the configured model dtype, and keep a boundary
  conversion for callers whose query dtype differs from the cache. Do not add
  a compensating cast at every output projection or change Triton kernels.
- Decision implementation: `RotaryEmbedding` now computes frequencies in
  float32, converts the completed cos/sin cache to `torch.get_default_dtype()`
  during model construction, and converts indexed cache values to the query
  dtype before applying RoPE. Added focused tests for configured BF16 cache
  creation and for BF16 query/key dtype preservation when a float32 cache is
  used. The tests index documents the new rotary regression coverage.
- Decision result: Focused tests pass (`10 passed`), and real CUDA warmups for
  both local Qwen3-0.6B and Qwen3-1.7B snapshots complete with model parameters,
  rotary cache, and KV cache all in `torch.bfloat16`. The repository-wide suite
  passes (`181 passed, 1 skipped`); the skipped test remains the opt-in NCCL
  hardware integration case.

## 2026-07-19: Make Paper and Report Evidence Directly Auditable

- Decision demand: The draft omitted the public repository URL and did not call
  the MLSys template hook that renders registered affiliations. The paper and
  report also emphasized a relative-gain summary without enough original-unit
  evidence, while the repository README used a light architecture image that
  was difficult to read in a dark presentation context.
- Decision plan: Restore publication metadata at the template-defined output
  points, retain the compact conclusion figure, add absolute and original
  benchmark views sourced from the archived five-trial JSON/artifacts, and
  generate light and dark architecture variants from one synchronized source.
- Decision implementation: The abstract now links the public GitHub repository
  and the title block calls `printAffiliationsAndNotice` after registering the
  HKUST affiliation. The report embeds both an absolute-metric aggregate and
  all four original routing/session-handoff benchmark summaries. The paper adds
  a four-panel absolute throughput/TTFT figure alongside the relative summary.
  The result generator exports both figures from the archived JSON. The
  architecture generator now renders a light paper image plus light and dark
  repository assets from shared geometry and labels; both README languages
  reference the dark asset.
- Decision result: Author metadata, source availability, relative claims,
  absolute values, and full benchmark panels are now independently visible.
  The paper and README architecture diagrams remain structurally identical
  while using backgrounds appropriate to their display contexts.

## 2026-07-19: Expose Routing, Prediction, and Packed-Transfer Decisions

- Decision demand: Routing, hotspot prediction, and transfer were described in
  prose but remained easy to conflate. In particular, an access threshold could
  be mistaken for immediate transfer admission, and the relationship between
  control metadata and the NCCL tensor payload was not visually explicit.
- Decision plan: Trace the implemented route, background placement, foreground
  shortage, and KV data paths; separate proposal from admission; and generate
  one synchronized flowchart in light and dark themes for the paper, report,
  and repository documentation.
- Decision implementation: Added a reproducible three-panel decision-flow
  generator. The routing panel shows contiguous prefix lookup, queue/prefill/
  reclaim cost, bounded owner bypass, and route reservation. The prediction
  panel shows complete-chain access counters, route hits, optional pending-
  ingress demand, conservative reuse estimation, pair batching, cooldown/idle/
  capacity checks, and the final benefit gate. The transfer panel shows actual
  foreground shortage or admitted background copy, source generation checks,
  prepare reservations, the contiguous `[layers, K/V, blocks, block size, KV
  heads, head dimension]` payload, pair-local NCCL, indexed unpack, publish, and
  copy/move finalize semantics. Paper, report, and both README languages now
  embed the corresponding generated figure and implementation explanation.
- Decision result: Documentation now makes clear that prediction only proposes
  a candidate, measured cost decides execution, metadata remains on the control
  protocol, and NCCL carries only packed K/V values. All three documents share
  one source of truth for the decision flow.

## 2026-07-19: Separate Decision Branches and Clarify Runtime Transfer Batches

- Decision demand: The combined three-column decision figure looked like three
  linear pipelines and hid rejection, deferral, spill, and rollback branches.
  The architecture figure also placed its NVLink annotation below the workers
  instead of visibly connecting two rank-local KV tensors. Documentation could
  be read as if the four-block microbenchmark fixed the online transfer batch.
- Decision plan: Render routing, hot-prefix prediction, and transfer as three
  independent branch diagrams; use one shape/arrow accent per diagram; simplify
  architecture labels with smaller explanatory text; use one neutral arrow
  color; connect the direct-pair data path at the Model Runner boundaries; and
  distinguish calibration samples from runtime plan sizing.
- Decision implementation: `generate_decision_flow.py` now emits three light
  paper/report images and three dark README images. Diamonds expose feasibility,
  owner-spill, hotness, replica need, benefit, prepare, and execute outcomes.
  `generate_architecture.py` now renders title/subtitle typography, neutral
  arrows, and a pair-local bidirectional arrow between rank 0 and rank 1 Model
  Runners. Paper, report, both README languages, and the paper runbook now state
  that foreground size equals actual shortage, one background candidate is
  capped at eight blocks by default, and same-pair candidates may coalesce up to
  the default 128-block plan cap. Four blocks remains a conservative measured
  initialization point, not an execution invariant.
- Decision result: Each policy can be read and cited independently, all failure
  paths are visible, the architecture distinguishes control edges from colored
  components, and benchmark calibration is no longer conflated with the actual
  transfer batch distribution.

## 2026-07-19: Document the Transactional KV-Block Lifecycle

- Decision demand: The documentation explained allocation, reclaim, and
  transfer separately, but it did not show when a physical KV block becomes
  reusable or routable, how a reference-zero cache entry differs from a free
  block, or what abort must restore after a failed transfer.
- Decision plan: Derive one lifecycle directly from Local Block Manager and
  data-plane transfer phases, separate the ordinary prefix-cache path from the
  cross-GPU transaction, and generate synchronized light and dark figures for
  the paper and repository documentation.
- Decision implementation: Added `generate_block_lifecycle.py`. The normal lane
  shows `Free`, `Allocated / Writing`, `Ready / In Use`, and `Cached /
  Reclaimable`, including partial release, prefix-hit reactivation, and
  dependency-safe reclaim. The transfer lane shows generation lock and target
  reservation, ModelRunner tensor movement, received-but-hidden state,
  publication, copy/move finalization, and abort. The paper and both README
  languages now embed the generated figure and state explicitly that routing
  observes only published ready blocks.
- Decision result: Block metadata ownership, physical tensor movement,
  visibility, cache retention, reclamation, and rollback are now represented by
  one implementation-aligned lifecycle rather than independent prose fragments.

## 2026-07-20: Separate and Visualize the Two Online Cost Models

- Decision demand: The report described routing as preserving "parallelism,"
  which could be confused with DP/TP/PP strategy rather than the actual problem
  of request skew and replica underutilization. Routing and transfer costs were
  also compressed into prose despite using different units and observations.
- Decision plan: Correct the terminology, derive both models directly from the
  current scheduler, and render routing and transfer as independent figures so
  each formula, gate, and online signal can be read without conflation.
- Decision implementation: Added `generate_cost_models.py`, producing light
  paper and dark README variants of two separate diagrams. The routing figure
  shows effective-capacity feasibility, token-aware queue work, missing prefill
  derived from contiguous prefix hits, reclaim pressure, minimum projected cost,
  and bounded spill. The transfer figure shows exact payload bytes, calibrated
  wire cost, pair/placement EWMAs, distinct foreground/background saved-work
  estimates, the benefit-ratio gate, and negative-cache rejection. Paper
  equations and both README languages now match these implementations. The
  report now describes request/load concentration instead of an ambiguous loss
  of parallelism.
- Decision result: Documentation now distinguishes unchanged data-parallel
  execution from load-skew control, and it presents locality routing and KV
  movement admission as two explicit, independently auditable cost models.

## 2026-07-20: Add Dataset-Level Prefix-Sharing Profiles

- Decision demand: Runtime request-hit and cached-token metrics could not show
  whether a weak result came from the trace itself or from finite capacity,
  placement, eviction, and transfer policy. The existing theoretical metric
  only reported whether a request shared at least one block and hid the amount
  of reusable prompt work.
- Decision plan: Profile every ordered prompt trace before worker launch using
  both request and token denominators, keep the calculation independent of the
  runtime KV budget, and expose the full counts in benchmark artifacts and the
  paper dataset table.
- Decision implementation: Added `profile_trace_prefix_sharing()` to compute
  the longest previously observed contiguous chain of complete cumulative
  prefix hashes for every request. It reports shareable requests, blocks and
  tokens, total prompt tokens, unique complete hashes, request prefix share,
  and token prefix share under unlimited-cache perfect placement. E2E and
  routing benchmarks now store the profile in `metadata.dataset_profile`, add
  explicit `trace_request_share_rate` and `trace_token_share_ratio` result
  fields, preserve `theoretical_prefix_hit_rate` as a compatibility alias, and
  print both rates next to runtime control/data-plane metrics. Added exact and
  partial-tail tests and synchronized the benchmark guide, paper, report, and
  both README languages with the four paper-workload profiles.
- Decision result: Dataset reuse potential can now be separated quantitatively
  from achieved `DP req hit` and `DP tok reuse`. The paper traces range from
  63.28--97.40% request sharing and 67.81--90.57% token sharing, with the
  locality trace at 91.67% and 86.20%, respectively.

## 2026-07-20: Make Workload Composition and Report Figure Scope Explicit

- Decision demand: Abbreviated dataset labels such as `hot+cold` and
  `15 hot + pressure` did not reveal request counts, prefix lengths, recurrence,
  or that every pressure prefix is unique. Report Figure 5 also appeared to be
  an unnamed single-workload result and its overlap with the following raw
  benchmark figures was unexplained.
- Decision plan: Replace every abbreviated prefix-group label with exact group
  and request counts, expose the three memory-skew phases, and document Figure
  5 panel by panel together with the purpose of the per-model raw figures.
- Decision implementation: Updated the benchmark guide, paper dataset table,
  report, and archived dataset profile to state the exact locality, load-skew,
  memory-skew, and session-handoff constructions. The paper table uses wrapping
  columns so the complete descriptions cannot overflow adjacent metrics. The
  report now identifies Figure 5(a--b) as locality/routing and Figure 5(c--d)
  as session handoff, and states that its throughput/TTFT overlap with the four
  raw benchmark figures is intentional: Figure 5 is a compact cross-model
  comparison, while the raw figures retain the wider metric and uncertainty
  set.
- Decision result: Prefix sharing percentages are now auditable from exact
  trace composition, and every report figure states both its workload and its
  relationship to the underlying benchmark artifacts.

## 2026-07-25: Replace Single-Point Transfer Pricing with Size-Aware Profiles

- Decision demand: The serving cost model priced every transfer with one
  effective-bandwidth scalar, historically selected from the four-block
  microbenchmark. Fixed NCCL and packing overhead makes effective bandwidth
  strongly payload-dependent, so that scalar overprices large plans and can
  underprice small plans. The pair-local EWMA also mixed observations from
  different plan sizes, allowing one cold small transfer to reject unrelated
  large transfers.
- Decision plan: Preserve the measured latency curve instead of collapsing it
  to one bandwidth. Calibrate every physical NVLink pair at powers-of-two block
  counts, map those files to the logical pairs used after CUDA remapping, use
  conservative piecewise-linear P95 latency as the offline prior, and restrict
  online corrections to the observed pair and block-size bucket.
- Decision implementation: `benchmark_kv_transfer.py` now reports mean, P50,
  and P95 for the paper sweep `1,2,4,8,16,32,64`.
  `build_transfer_profile.py` and `transfer_profile.py` create a versioned
  artifact containing physical provenance, logical-pair mapping, model KV
  geometry, raw samples, and monotonic decision points. `benchmark_e2e.py`
  loads this artifact through `--foreground-transfer-profile-json` and rejects
  pair or bytes-per-block mismatches before worker launch. `GlobalScheduler`
  interpolates within each pair curve, uses the final measured slope beyond
  the range, and maintains separate source-transfer and
  dispatch-to-publication residual EWMAs keyed by `(pair,
  next_power_of_two(block_count))`. The old bandwidth option remains only as a
  no-profile fallback. `run_paper_suite.sh` now constructs one profile per
  model and injects it into every E2E workload.
- Decision result: Transfer admission now prices the actual plan size and
  topology instead of treating four blocks as representative of all plans.
  Focused scheduler/profile coverage passes 40 tests and the complete CPU suite
  passes 188 tests with one hardware-gated NCCL skip. Coverage includes
  interpolation, pair isolation, size-bucket isolation, monotonic profile
  construction, and geometry/topology mismatch rejection. End-to-end
  performance claims require a fresh GPU benchmark run with the generated
  profile and are not inferred from these unit tests.

## 2026-07-25: Make Transfer Figures Reproducible from Result Artifacts

- Decision demand: Mean, P50, and P95 transfer latencies converge for warmed
  large payloads. Placing every value two points above its grouped bar caused
  the three labels to overlap, even though the underlying JSON and latency
  profile were correct.
- Decision plan: Treat figures as deterministic post-processing artifacts,
  preserve horizontal value labels, and make historical results redrawable
  without allocating GPUs or rerunning the microbenchmark.
- Decision implementation: `benchmark_kv_transfer.py` now uses a wider
  two-panel canvas, places Mean/P50/P95 labels on three vertical levels, and
  reserves explicit y-axis headroom. The new `--input-json` mode reads the
  model label and result rows from an existing artifact and writes only the
  requested PNG. A benchmark test covers this JSON-to-figure path.
- Decision result: Transfer plots can be corrected or restyled independently
  of measurement, while the JSON and E2E latency profile remain immutable.

## 2026-07-25: Remove Transfer-Sweep Port Races and Support Resumption

- Decision demand: The dual-model paper run aborted in the Qwen3-1.7B transfer
  sweep with `TCPStore EADDRINUSE`. The benchmark selected a nominally free TCP
  port, closed the probing socket, and only later spawned rank 0 to bind it.
  Another process could claim the port in that interval. The abort also forced
  users to choose between rerunning completed experiments and manually
  reconstructing the remaining matrix.
- Decision plan: Remove the probe-then-bind race instead of retrying arbitrary
  ports, fail fast when a worker exits, and make the paper runner reuse only
  structurally complete result artifacts.
- Decision implementation: Every transfer payload case now creates a unique
  `/tmp/lmpool-kv-transfer-<pid>-<uuid>.store` rendezvous and initializes NCCL
  through `file://`, then removes the store after both workers join. Queue
  polling catches only `queue.Empty`, stops as soon as a worker has a nonzero
  exit code, and includes both exit codes in validation failures. The known
  eager-init unbatched-P2P warning is disabled for this benchmark because each
  pair intentionally admits one transfer transaction at a time.
  `run_paper_suite.sh` adds `RESUME=1`; it accepts an existing JSON only when it
  is nonempty, contains both `metadata` and `results`, and has exactly the
  expected 7 transfer, 3 routing, or 5 E2E result rows. It rebuilds the transfer
  profile from the complete pair artifacts.
- Decision result: Five focused CPU tests pass, including unique rendezvous
  coverage. A real Qwen3-1.7B two-case sweep on physical GPUs 3/4 completed
  both one- and two-block cases with data validation passed, no TCP bind
  failure, and no eager-init serialization warning. The complete CPU suite
  passes 189 tests with one hardware-gated NCCL skip. The interrupted paper
  run was intended to continue in place without rerunning the complete
  Qwen3-0.6B matrix; the next decision records and corrects a schema-validation
  defect in that first resume implementation.

## 2026-07-25: Correct Resume Validation and Loaded Transfer Admission

- Decision demand: Resuming the interrupted paper suite unexpectedly reran and
  overwrote completed Qwen3-0.6B E2E experiments. The repeated memory-skew
  results also showed that size-aware admission accepted transfers whose
  measured target-side cost exceeded their predicted recomputation saving.
- Decision plan: Validate the actual JSON schemas emitted by every benchmark,
  then compare estimated transfer cost with the complete target-side serving
  transaction. Correct the calibration boundary without adding another
  scheduling policy.
- Decision implementation: `run_paper_suite.sh` now validates the transfer
  microbenchmark's result array and the routing/E2E benchmarks'
  scenario-keyed result object, while still requiring metadata and the exact
  expected result count. Resume additionally matches the model path,
  repetition count, workload, and E2E transfer-cost calibration so
  structurally complete artifacts from a different configuration are not
  silently mixed. The transfer profile remains an idle, pair- and size-specific P95
  data-path measurement. A size-independent additive residual fits both the
  short foreground and large handoff plans better than multiplying every
  payload by the short-plan slowdown. The final cold-start prior is therefore
  `40 ms + 1.2 * profile_P95(plan_bytes)`. The 40 ms term was selected manually
  as a conservative setting after earlier Qwen3-0.6B runs exposed a gap between
  an approximately 10.7 ms estimate for a seven-block foreground plan and an
  approximately 43--47 ms complete target-side transaction. It was not fitted
  by a standalone pilot or held-out cost study. The additive term prices
  scheduling, NCCL queuing, target publication, and block registration outside
  the idle microbenchmark boundary. Existing pair-and-size placement EWMAs
  remain a conservative online correction. A scheduler regression test checks
  the intended decision boundary: a low-reuse seven-block plan is rejected,
  while an amortized 64-block plan remains admissible under the measured
  latency-curve shape.
- Decision result: Resumption reuses matching transfer arrays and matching
  scenario-keyed routing/E2E artifacts, while rejecting old E2E files whose
  cost calibration differs. Cold, one-shot foreground plans are not admitted on an
  underpriced idle-link estimate, while session-handoff-scale transfers can
  still pass when their predicted prefill saving covers the loaded cost. The
  overwritten `20260725T031840Z` Qwen3-0.6B results are retained as diagnostic
  evidence rather than final post-fix evaluation. The complete CPU suite passes
  190 tests with one hardware-gated NCCL test skipped.

## 2026-07-25: Redraw Process Architecture and Decision Flows from Runtime Ownership

- Decision demand: The existing figures mixed process ownership with logical
  modules, routed request arrows through control-plane boxes, and used dark
  README variants that no longer matched the requested paper style. The
  diagrams also needed to make explicit that global state is metadata while
  physical KV moves only between data-plane Model Runners.
- Decision plan: Use one white-background, pastel, flat-vector visual system
  for the paper, READMEs, and report. Separate the architecture into the main,
  control-plane, and per-GPU data-plane processes; distinguish request/result,
  metadata/control, and KV-payload paths; and keep routing, background
  placement, and transactional transfer as three independent branching
  flowcharts.
- Decision implementation: Rewrote `generate_architecture.py` with explicit
  LLMEngine, Global Scheduler, Global Block Manager, Local Scheduler, Local
  Block Manager, Model Runner, and physical KV-cache modules. Request/result
  paths now run outside the control-plane modules, state/control paths are
  dashed, and the packed KV tensor connects the two Model Runners directly.
  Rewrote `generate_decision_flow.py` with concise English labels and actual
  feasibility, owner-spill, hotness, replica, capacity, cost, prepare, publish,
  and abort branches. Both generators emit synchronized PNG assets for
  README, paper, and report, plus vector PDF paper outputs. Updated both
  READMEs, the paper design section, the paper artifact guide, and the
  2026-07-20 report to describe the same ownership and flow semantics.
- Decision result: All current architecture and decision figures use a pure
  white background, soft academic colors, readable English labels, and no
  decorative texture or shadow. Visual inspection confirmed that arrows
  terminate at the intended modules, the NVLink payload is aligned between
  Model Runners, and every decision flow exposes its rejection or rollback
  path. This change is documentation-only and does not alter runtime behavior.

## 2026-07-25: Refine Diagram Geometry and Restore Dual Light/Dark Themes

- Decision demand: The first redraw still used opaque arrow-label backgrounds,
  compressed several arrowheads between tightly packed nodes, showed only two
  full worker ranks, and left the lifecycle and cost-model figures in the older
  visual style. README also needed a dark-background presentation while the
  paper and report required a white-background version.
- Decision plan: Keep one implementation-aligned diagram source per concept,
  but render synchronized light and dark variants. Use transparent labels,
  orthogonal connectors that terminate exactly at node borders, visible gaps
  before every arrowhead, a green KV payload path, and enough representative
  workers to communicate an N-rank deployment without duplicating every
  internal module.
- Decision implementation: `generate_architecture.py` now renders complete
  Rank 0 and Rank N-1 data planes plus compact Rank 1, Rank 2, intermediate,
  and Rank N-2 workers. LLMEngine request/result paths enter each full worker
  at its side midpoint, while the packed KV tensor connects Model Runners in
  green. `generate_decision_flow.py` uses compact process nodes, explicit
  feasibility/admission/failure diamonds, and only horizontal or vertical
  connectors. `generate_block_lifecycle.py` was rebuilt around separate local
  reuse and transactional transfer state machines, including hidden receive,
  publish, abort, copy, and move states. `generate_cost_models.py` was rebuilt
  as separate routing and transfer decision diagrams with size-aware profile,
  online EWMA, saved-work, and admission branches. All four generators emit
  light PNG/PDF paper assets and dark README assets; README links and color
  descriptions now select the dark variants.
- Decision result: Visual inspection of every light figure and representative
  dark variants confirmed readable arrowheads, border-aligned endpoints,
  transparent labels, balanced whitespace, and consistent academic colors.
  Paper/report figures remain white, README figures remain dark, and KV
  transfer is consistently green. This is a documentation-only change and
  does not alter routing, transfer, or block-management behavior.

## 2026-07-25: Center Architecture Content and Expose NVLink Rank Pairs

- Decision demand: The architecture still inherited a left-biased module
  helper, oversized process containers, labels that competed with box
  boundaries, and one-dimensional rank abbreviations. This made LLMEngine look
  off-center, allowed long local-module titles to approach their borders, and
  obscured that every intermediate rank has the same four-module process
  structure and belongs to a direct NVLink pair.
- Decision plan: Replace coordinate patching with a geometry-first layout.
  Center every title and subtitle on its owning rectangle, reserve whitespace
  between process containers for inter-process labels, make outer containers
  fit their contents, and represent the scalable worker topology with compact
  two-by-two rank cards connected in explicit pairs.
- Decision implementation: Rebuilt `generate_architecture.py` around centered
  process-band and module primitives. Main Process, Control Plane Process, and
  Data Plane Processes now use content-sized bounds. Full Rank 0 and Rank N-1
  workers and compact Rank 1, Rank 2, Rank 3, and Rank N-2 workers all use a
  two-by-two Scheduler, Block Manager, Model Runner, and KV Cache arrangement.
  Rank 0--1, Rank 2--3, and Rank N-2--N-1 Model Runners are joined by green
  bidirectional NVLink paths. The LLMEngine request/result path terminates at
  the Data Plane Processes side midpoint, while control/data labels sit beside
  their arrows in the gap between process borders. Long local-module titles
  use centered two-line text, and repeated labels that could enter adjacent
  rank cards were removed.
- Decision result: Light and dark renders now keep all module text centered and
  inside its owner, keep external labels out of module bounds, show scalable
  rank/process multiplicity, and expose three representative direct-NVLink
  pairs without a long payload line crossing unrelated ranks. The paper,
  report, and both READMEs describe the same paired topology. Runtime behavior
  is unchanged.

## 2026-07-25: Make Figure Edges Match Runtime Component Boundaries

- Decision demand: The architecture still hid the routing chain behind
  process-level arrows, abbreviated modules inside compact ranks, and did not
  label each pair-local payload edge as NVLink. The lifecycle figure mixed
  several unrelated state colors and started Copy/Move paths inside their
  decision diamond. The transfer cost figure merged saved-work inputs above
  the admission diamond without visibly reaching its top vertex.
- Decision plan: Draw communication between the concrete runtime owners,
  preserve only the two endpoint rank pairs plus an ellipsis for scalability,
  and make every connector terminate exactly at a module or decision border.
  Use one restrained state color family for the lifecycle rather than encoding
  undocumented semantics with color.
- Decision implementation: `generate_architecture.py` now labels workers as
  `Data Plane Process @ GPU Rank x`, expands every compact rank to the full
  Local Scheduler, Local Block Manager, Model Runner, and Physical KV Cache
  names, and shows Rank 0--1 and Rank N-2--N-1 Model Runner links with an
  explicit `NVLink` label. LLMEngine exchanges route query/target-rank data
  directly with Global Scheduler, dispatches the selected Sequence to a Local
  Scheduler, and the control plane exchanges transfer phases and versioned
  snapshots with concrete Local Block Managers. Process containers now use
  near-background fills so their modules remain visually distinct.
  `generate_block_lifecycle.py` applies one blue-green family to every state,
  decision, and edge, and starts Copy/Move at the left/right diamond vertices.
  `generate_cost_models.py` joins foreground and background saved-work inputs
  immediately above the admission decision and connects the merged edge to the
  diamond's top vertex.
- Decision result: The synchronized paper/report light figures and README dark
  figures expose routing, control, dispatch, and pair-local KV movement without
  crossing labels or entering node interiors. The lifecycle and transfer-cost
  decisions now have border-exact incoming and outgoing edges. These changes
  only affect documentation assets; runtime routing and transfer behavior are
  unchanged.

## 2026-07-25: Separate Process Boundaries from Runtime Module Colors

- Decision demand: The architecture process containers still reused the blue,
  purple, and green colors of their internal modules, compact ranks reduced
  module labels below the full-rank font size, and the Global Scheduler edge
  visually ended at a worker boundary rather than at the component that
  consumes routing and transfer decisions.
- Decision plan: Treat process boundaries as neutral grouping regions, retain
  semantic colors only for runtime modules, and keep compact ranks readable
  without expanding the overall topology. Route every control edge to the
  concrete component that owns the operation.
- Decision implementation: `generate_architecture.py` now uses neutral gray
  fills and borders for Main Process, Control Plane Process, Data Plane
  Processes, and individual rank containers. Process headings use a distinct
  deep blue in the light figure and a matching light blue in the dark figure.
  Compact Rank 1 and Rank N-2 cards were widened, moved toward the central
  ellipsis, and assigned the same per-module font sizes as full Rank 0 and
  Rank N-1. The dashed Global Scheduler edge now enters Rank 0's Local
  Scheduler directly and is labeled `routing target / transfer phases`;
  Local Block Manager snapshots remain connected to Global Block Manager.
- Decision result: Regenerated paper/report light assets and README dark assets
  keep process grouping visually separate from scheduler, manager, runner, and
  cache semantics. The scalable rank layout is denser without shrinking text,
  and both the request dispatch and control-plane routing path visibly
  terminate at Local Scheduler. This is a documentation-only change.

## 2026-07-25: Use Uniform Full-Rank Architecture Cards

- Decision demand: Mixed full and compact rank cards made paired NVLink edges
  slightly diagonal, reduced the visual weight of intermediate workers, and
  left excessive whitespace around the central ellipsis. Main Process and
  Control Plane Process also used narrower containers than Data Plane
  Processes, while arrow annotations were too small at paper scale.
- Decision plan: Give every explicitly named rank the same complete process
  representation, reserve the center only for an abstract multiplicity cue,
  and align every pair-local transfer edge on a shared horizontal axis. Use
  full-width process bands and one stronger annotation style throughout.
- Decision implementation: `generate_architecture.py` now draws Rank 0, Rank 1,
  Rank N-2, and Rank N-1 with identical-size two-by-two Local Scheduler, Local
  Block Manager, Model Runner, and Physical KV Cache layouts. Mirrored cards
  place paired Model Runners on the facing edges, making both labeled NVLink
  payload arrows exactly horizontal. Four small neutral rank pairs with green
  horizontal NVLink links surround the central ellipsis to represent the
  omitted instances. Main Process, Control Plane Process, and Data Plane
  Processes now share the same width. Arrow labels use larger bold text in
  both light and dark themes.
- Decision result: The regenerated paper/report and README figures present all
  visible workers at equal importance, distinguish omitted multiplicity
  without allocating another full card, and preserve horizontal transfer
  geometry. Updated paper and report captions describe the same representation.
  Runtime behavior remains unchanged.

## 2026-07-25: Expose All Scheduler and Block-Metadata Connections

- Decision demand: The architecture showed only representative control edges,
  which could be read as Global Scheduler controlling one worker and Global
  Block Manager receiving one worker snapshot. The scalable topology cue also
  needed six paired instances, larger labels, and connectors with simple
  orthogonal geometry.
- Decision plan: Draw separate scheduler fan-out and block-metadata fan-in
  buses. Every detailed Local Scheduler must receive a control branch from
  Global Scheduler, and every detailed Local Block Manager must publish a
  branch to Global Block Manager. Keep each bus segment straight and each
  local branch to at most one bend, retain horizontal green NVLink links, and
  enlarge labels without changing the process boundaries.
- Decision implementation: `generate_architecture.py` now draws a blue dashed
  routing/transfer-phase fan-out to all four detailed Local Schedulers and a
  purple dashed versioned-snapshot fan-in from all four detailed Local Block
  Managers. The central multiplicity cue contains six small NVLink-connected
  pairs, three above and three below the ellipsis. Process headings render
  above crossing lines, and all architecture annotations use the larger
  shared font scale. The paper caption, both READMEs, and the technical report
  describe the same component-level semantics.
- Decision result: Regenerated light and dark architecture assets make the
  global-to-local control relation and local-to-global metadata relation
  explicit for every detailed rank. Visual inspection confirms horizontal
  NVLink paths, border-terminated arrows, readable text, and no module-label
  overflow. Runtime behavior is unchanged.

## 2026-07-25: Rebase Paper Claims on the Latest Five-Trial Artifact

- Decision demand: The paper, READMEs, and report still quoted the
  `20260719T072508Z` batch after a new complete dual-model batch had been
  produced at `20260725T031840Z`. The old text also overstated routing latency
  gains and reduced the transfer microbenchmark to two calibration sizes.
- Decision plan: Recompute every stated number directly from the latest JSON,
  separate mechanism evidence from end-to-end evidence, retain negative
  workload results, and use figures that show both relative effects and
  absolute units with uncertainty.
- Decision implementation: Updated the routing and session-handoff tables,
  abstract, evaluation, conclusion, READMEs, paper guide, and report. Routing
  is now reported as a 48.6--50.2% reduction in uncached prefill tokens with a
  model-dependent TTFT result. Session handoff reports 4.4--7.4% throughput,
  32.4--40.4% mean-TTFT, 9.3--12.9% mean-E2E, and 7.4--7.6% P90-E2E
  improvements over round-robin multi-GPU. The transfer figure now plots the
  1/2/4/8/16/32/64-block mean profile with an observed min--max band over two
  models and three physical pairs. Absolute throughput and TTFT panels carry
  95% confidence intervals from five trials.
- Decision result: The public narrative now distinguishes three claims:
  routing improves cached-token locality, the NVLink microbenchmark validates
  and calibrates the packed data path, and session handoff demonstrates that
  reuse can amortize transfer end to end. Load skew and memory skew remain
  visible boundary results. The complete CPU suite passes with
  `190 passed, 1 skipped`; the skipped case is the opt-in CUDA/NCCL integration
  test. A local PDF build was not possible because neither `latexmk` nor
  `pdflatex` is installed.

## 2026-07-25: Rewrite the Paper as Continuous Academic Prose

- Decision demand: The draft contained emphasized principle labels, numbered
  contribution fragments, itemized metadata, and phase labels that read like
  generated notes rather than a finished systems paper.
- Decision plan: Preserve technically precise passages that already read
  naturally, but replace list-like body text with connected paragraphs,
  remove decorative bold and italic emphasis, and keep mathematical notation,
  tables, and escaped LaTeX characters intact.
- Decision implementation: Rewrote the abstract, principles, contributions,
  global-metadata description, transfer protocol, baselines, limitations, and
  related-work comparison in `example_paper.tex`. The routing discussion now
  refers to load balance rather than conflating request concentration with
  model/data parallelism. All `itemize`, body `textbf`, and body `emph`
  constructs were removed.
- Decision result: Static checks find no list environments or bold/italic body
  emphasis, no stale artifact identifiers or headline metrics, no trailing
  whitespace, and no `git diff --check` errors. Existing natural technical
  passages and equations were retained rather than rewritten for stylistic
  variation alone.

## 2026-07-25: Remove the Template Notice and Reflow Wide Equations

- Decision demand: The accepted-paper template printed an obsolete MLSys 2025
  proceedings and copyright sentence, and several routing and transfer
  equations exceeded a two-column paper's available width.
- Decision plan: Suppress only the template notice so author affiliations and
  correspondence remain visible. Reflow wide displays at semantic boundaries
  instead of shrinking their type, and introduce compact notation where a
  long textual condition or sample set caused the overflow.
- Decision implementation: `example_paper.tex` now clears
  `\Notice@String` after loading the accepted template. The NVLink-neighbor
  condition uses the defined set `\mathcal{N}_{\mathrm{NV}}(s)`, queue cost is
  split across weighted work terms, the measured transfer profile is named
  `\mathcal{P}_p`, and the three conservative transfer-cost candidates are
  placed on separate aligned lines.
- Decision result: The paper retains its affiliation footnote while omitting
  the obsolete proceedings notice. All display equations now fit the source's
  two-column layout without reduced math font size. Static LaTeX structure and
  whitespace checks pass; PDF compilation remains unavailable in the current
  environment because no TeX engine is installed.

## 2026-07-25: Publication-Level Language Revision

- Decision demand: The complete paper required a rigorous language revision
  for an MLSys submission, including grammar, sentence structure, formal
  register, article use, terminology consistency, and removal of contractions
  and noun possessives without changing the technical claims.
- Decision plan: Edit every prose section, caption, and explanatory table
  entry in place. Preserve the section structure, citations, references,
  LaTeX commands, mathematical expressions, experimental values, and explicit
  negative results. Prefer short, direct academic sentences over decorative
  vocabulary or list-like exposition.
- Decision implementation: Rewrote the abstract, introduction, motivation,
  system design, implementation, evaluation, limitations, related work,
  conclusion, figure captions, and descriptive table cells in
  `example_paper.tex`. The revision clarifies ownership boundaries, separates
  routing evidence from transfer evidence, removes method-name possessives and
  contractions, standardizes warm-up and component terminology, and preserves
  the distinction between positive session-handoff results and boundary
  results under memory and load skew.
- Decision result: The paper now uses consistent formal academic prose with no
  detected contractions, noun possessives, list environments, or newly added
  emphasis commands. LaTeX environment counts remain balanced, experimental
  values and citations are unchanged, and `git diff --check` passes. A visual
  PDF proof remains unavailable because the environment has no TeX engine.

## 2026-07-25: Final Systems-Paper Redline Audit

- Decision demand: The final draft required a high-tolerance audit against a
  systems-paper writing guide and the implementation. Only contradictions,
  ambiguous core terminology, severe language defects, and unsupported claims
  were eligible for correction.
- Decision plan: Compare the paper structure with the external guide, trace
  routing and transfer claims to the current engine code, verify the reported
  numbers against the latest five-trial artifacts, and check cited metadata
  against primary sources. Preserve established prose and negative results
  unless a concrete inconsistency requires a change.
- Decision implementation: Recast the abstract as five sentences within the
  recommended length, corrected ingress routing from pair-restricted to
  all-healthy-replica selection, and retained pair restriction for
  worker-originated rerouting and KV transfer. The routing formula now labels
  missing prefill work as a block-rounded estimate, the limitations state
  explicitly that scale-out beyond six GPUs was not evaluated, and the
  conclusion uses three sentences. The LMCache bibliography entry now matches
  the complete author list in the primary paper record.
- Decision result: The architecture narrative now agrees with the
  `requester_rank=-1` ingress path and pair-local transfer implementation.
  Routing, transfer, and session-handoff values remain consistent with
  `benchmarks/results/paper/20260725T031840Z`. No remaining contradiction or
  terminology defect that blocks interpretation was found by static review.

## 2026-07-25: Evidence-Based Evaluation Explanations

- Decision demand: The evaluation reported accurate means and improvements but
  did not consistently explain why one policy outperformed another. Each major
  conclusion required a compact LaTeX paragraph that connected the outcome to
  measured mechanism and per-rank evidence without inventing causality.
- Decision plan: Re-read the latest routing, transfer, session-handoff,
  memory-skew, and load-skew JSON artifacts for both models. Compare cached and
  uncached tokens, per-rank prefill and decode time, request distribution, GPU
  utilization, transfer counts, placement-lease routes, and confidence
  intervals before assigning any explanation.
- Decision implementation: Reorganized Q1--Q4 into titled `\paragraph{}`
  analyses. The routing section distinguishes token savings from latency
  payoff using summed per-rank execution time. The transfer microbenchmark
  explains batching through the sublinear latency increase when payload
  doubles. The handoff section separates the effects of transfer and routing,
  adds the omitted routing-only ablation to the results table, and identifies
  placement-lease routing and improved rank utilization as the measured source
  of the combined throughput gain. The boundary section explains why balanced
  memory-skew placement and naturally replicated load-skew prefixes leave
  little avoidable work.
- Decision result: Evaluation claims now state both the improvement and the
  measured reason for it. The text also records that transfer does not improve
  every latency metric: for Qwen3-0.6B session handoff, routing-only and full
  LMPool have overlapping mean-E2E confidence intervals. This qualification
  prevents the ablation result from being overstated.

## 2026-07-25: Figure and Table Caption Normalization

- Decision demand: Figure and table captions mixed noun phrases followed by
  periods with multi-sentence explanations, which violated the requested
  distinction between Title Case phrases and punctuated Sentence case prose.
- Decision plan: Classify every caption by function. Use short Title Case noun
  phrases for self-explanatory module and test tables, and use complete
  Sentence case statements when a figure or results table requires mechanism,
  panel, or statistical context.
- Decision implementation: Revised all fifteen captions in
  `example_paper.tex`. Figure captions now begin directly with the represented
  mechanism or metric and retain only information needed to interpret paths,
  admission decisions, panels, ranges, or confidence intervals. The module and
  test tables use unpunctuated Title Case phrases. Results-table captions use
  complete Sentence case statements. The definition of `Partial` was moved
  from the related-work table caption into the surrounding prose.
- Decision result: Every caption now follows one consistent grammatical form:
  an unpunctuated Title Case noun phrase or a complete, punctuated Sentence
  case statement. LaTeX commands, mathematical notation, metric units, and
  escaped special characters remain intact.

## 2026-07-25: MLSys Review and Claim-Scope Remediation

- Decision demand: The final paper required an independent MLSys-style review
  that separated correctness flaws from evidence limitations, followed by
  revisions for every weakness that could be repaired without fabricating new
  experiments.
- Decision plan: Evaluate community contribution, experimental rigor,
  baseline fairness, ablation coverage, and consistency between the
  introduction and evaluation. Preserve unresolved evidence gaps in the
  review, while narrowing the paper claims and making the experimental scope
  explicit.
- Decision implementation: Added
  `docs/reviews/review_20260725.md` with a 5/10 weak-reject/borderline
  assessment. The paper now identifies session handoff as a synthetic,
  forecast-assisted opportunity study, states that the benchmark supplies
  exact queued-request demand, and distinguishes controlled Mini-vLLM
  ablations from production-system comparisons. The limitations now cover the
  absence of a production trace and production baselines, the six-GPU and
  sub-2B-model scope, the 256-token block size, and the lack of held-out cost
  prediction and sensitivity results. The abstract, evaluation questions,
  dataset profile, captions, Q3 analysis, and conclusion use the same bounded
  claim.
- Decision result: The paper no longer implies that an online predictor was
  evaluated or that the internal baselines establish superiority over vLLM,
  SGLang, Mooncake, or LMCache. Missing trace, baseline, scaling, and
  cost-model experiments remain explicit review weaknesses rather than being
  hidden by prose.

## 2026-07-25: Transfer-Prior Provenance and Oral Presentation Package

- Decision demand: The paper incorrectly described the fixed 40 ms transfer
  residual as the result of a Qwen3-0.6B pilot, and an MLSys oral presentation
  required a concise slide deck, a timed script, and defensible answers to
  likely technical questions.
- Decision plan: Trace the value through benchmark defaults, retained paper
  commands, result metadata, and runtime admission code. Replace any fitted-
  pilot claim with the strongest statement supported by those records, expand
  the limitations and future-work section, and prepare presentation material
  that separates demonstrated results from synthetic assumptions and negative
  results.
- Decision implementation: The paper, READMEs, benchmark documentation, and
  review now identify 40 ms as a manually selected conservative cold-start
  prior for coordination, NCCL queuing, publication, and block registration
  outside the idle data-path profile. The paper records the missing held-out
  prediction and sensitivity studies and proposes online forecasting on public
  traces, production-engine comparisons, scaling studies, policy ablations,
  and replicated control-plane recovery. Added a 17-frame Beamer deck at
  `docs/slide/lmpool_oral.tex`, a 15-minute slide-by-slide script at
  `docs/script/lmpool_oral_script.md`, and a 35-question defense guide at
  `docs/query/lmpool_oral_qa.md`. The materials include the complete request
  lifecycle, routing and transfer decisions, transaction safety, evidence
  boundaries, and the exact role of the forecast-assisted handoff workload.
- Decision result: The current-facing documentation no longer presents the
  40 ms prior as independently fitted evidence. The oral package reports the
  positive routing, NVLink, and handoff results together with the memory-skew
  and load-skew boundary results. Static checks confirm balanced slide
  environments and existing image dependencies; PDF compilation remains
  unavailable because the workspace has no TeX engine.

## 2026-07-26: Oral QA, Timed Scripts, and Slide Alignment

- Decision demand: The oral materials needed precise follow-up answers for
  routing, versioning, publication, cost admission, placement leases, and
  workload semantics. The presentation also needed a realistic ten-minute
  script, a retained fifteen-minute version, stronger slide-to-script
  alignment, complete transfer-profile data, and a consistent single-author
  voice.
- Decision plan: Trace every technical answer to the current control-plane,
  block-manager, transfer, benchmark, and result code paths. Preserve the
  seventeen-slide narrative, write separate timed scripts with explicit
  pointer cues, and revise the slide deck only where the requested explanation
  or visual evidence was missing.
- Decision implementation: Expanded `docs/qa/qa_20260728.md` with the ingress
  rank sentinel, bounded-spill semantics, four versioning layers, the
  prepare--publish visibility boundary, pair-and-size EWMA admission, profile
  interpolation and tail extrapolation, placement leases, rejection and
  admission boundaries, model-size interpretation, warm-up phases, P2P
  ordering, publication semantics, and forecast limitations. Rewrote
  `docs/script/script_20260728.md` as a pointer-aligned ten-minute script and
  added `docs/script/script_20260728_15min.md` as the detailed fifteen-minute
  version. Updated `docs/slide/slide_20260728.tex` with a raised footer,
  complete route-cost symbols, the transfer cost-model and background
  placement flows, all seven measured block-count rows, and unambiguous bold
  performance ranges. Current-facing paper and presentation prose now uses
  `I` and `my` instead of `we` and `our`.
- Decision result: The short script contains about 859 spoken English words,
  leaving time for pointer movement and pauses, while the detailed script
  contains about 1,816 words. The current paper title remains
  `LMPool: Locality-Aware Routing and NVLink KV-Cache Transfer for Multi-GPU
  LLM Serving` because it names both design principles, the transport, and the
  evaluation setting without claiming production-system superiority. Static
  checks cover LaTeX environment balance, image references, pronoun
  consistency, and whitespace; PDF rendering remains unavailable because no
  TeX engine is installed.

## 2026-07-26: Bilingual QA and HKUST Presentation Branding

- Decision demand: The oral-defense guide needed separate English and Chinese
  files with matching technical coverage. The slide deck also needed an
  acknowledgment frame and one consistent HKUST blue across the theme and
  embedded blue diagrams.
- Decision plan: Preserve the complete Chinese guide as the `_zh` variant,
  translate all thirty-five questions and follow-ups into an English primary
  file, use the official HKUST Pantone 285 web value, and keep green reserved
  for NVLink transfer so that transport remains visually distinct.
- Decision implementation: Moved the Chinese guide to
  `docs/qa/qa_20260728_zh.md` and created the corresponding English guide at
  `docs/qa/qa_20260728.md`. Changed `LMBlue` to HKUST blue `#0074BC`, updated
  the slide-local architecture and KV lifecycle generators, regenerated their
  PNG and PDF outputs, and added an eighteenth acknowledgment frame. Both
  timed scripts now include that final frame without extending their stated
  durations.
- Decision result: English and Chinese guides each retain thirty-five numbered
  questions and the two requested follow-ups. The deck has eighteen balanced
  frames, the slide theme and blue figure elements use HKUST blue, and the
  green NVLink path remains distinguishable. Static and visual checks cover
  structure, generated assets, wording, and file references; PDF compilation
  remains unavailable because the workspace has no TeX engine.

## 2026-07-26: Decode TPOT, Repetition-Level Confidence Intervals, and Bright Visual Palette

- Decision demand: The benchmark column named TTPT needed to represent pure
  decode time per output token, but it was computed as E2E latency divided by
  output-token count and therefore included queueing and prefill. The role and
  calculation of 95% confidence intervals also needed to be explicit. Slides,
  benchmark plots, and system diagrams used palettes that appeared too dark
  for presentation.
- Decision plan: Timestamp first-token and completion events at their
  data-plane source, compute request TPOT from the decode interval, rename the
  result schema and plots, document the Student-\(t\) interval, and synchronize
  light and dark assets around a brighter technology-brand palette without
  sacrificing text contrast.
- Decision implementation: Data-plane workers now attach monotonic timestamps
  to first-token and completion messages. `LLMEngine` stores the timestamps
  for one consumer step and then clears completed entries, avoiding unbounded
  metadata growth without changing the public `step()` tuple.
  `benchmark_e2e.py` computes
  `TPOT=(completion-first_token)/(output_tokens-1)`, excluding single-token
  requests. Result fields, summary labels, confidence-interval fields, JSON
  metric definitions, tests, the paper, the runbook, and the READMEs now use
  TPOT. The documented 95% interval is
  \(t_{0.975,R-1}s/\sqrt{R}\) across complete repetitions. Slide colors were
  renamed to `LMYellow` where appropriate, and generated diagrams and
  benchmark/report plots now use Google Chat blue `#2684FC`, green `#00AC47`,
  yellow `#FBBC04`, and red `#EA4335`, with pale fills for paper figures and
  higher-contrast variants for dark assets.
- Decision result: Focused benchmark, engine-message, and E2E tests report 32
  passed tests, and the complete CPU suite reports 192 passed and one skipped.
  Regenerated architecture, routing, placement, transfer,
  lifecycle, cost-model, and report figures retain readable text and visibly
  brighter categorical accents. Existing E2E result JSON cannot reconstruct
  the request-level TPOT distribution. Paper E2E and routing workloads must
  therefore be rerun before TPOT values are cited or latency values are mixed
  with the new worker-event timing boundary. The paper-suite resume check now
  rejects legacy E2E artifacts that do not contain `mean_tpot_s`; transfer-only
  microbenchmark artifacts remain reusable because their metric boundary did
  not change.

## 2026-07-26: Neutral Process Labels in Architecture Figures

- Decision demand: Process names in the architecture figure should use neutral
  black text instead of blue so that color remains a module-type cue rather
  than a hierarchy cue.
- Decision plan: Change only process-band and rank-process title colors while
  preserving the brighter blue, green, yellow, and red module boundaries.
- Decision implementation: Updated both paper and slide architecture
  generators so light figures use `#202124` for Main Process, Control Plane
  Process, Data Plane Processes, and per-rank process titles. Dark figures use
  the corresponding high-contrast white `#F8F9FA`.
- Decision result: Process hierarchy is now visually neutral, while scheduler,
  block-manager, model-runner, cache, and NVLink colors remain unchanged.

## 2026-07-26: Paper Testbed NVLink Topology Clarification

- Decision demand: The paper named the selected GPU pairs as NV4 but did not
  distinguish the topology label from the NVLink generation or state whether
  reported KV bandwidth should be compared with a unidirectional or
  bidirectional hardware limit.
- Decision plan: Record the live topology, active link count, logical-to-
  physical rank mapping, directional peak, and the measurement boundary of the
  KV transfer microbenchmark in the evaluation section.
- Decision implementation: `nvidia-smi topo -m`, `nvidia-smi topo -p2p n`, and
  `nvidia-smi nvlink -s` confirm direct NV4 paths for physical pairs `(0,1)`,
  `(3,4)`, and `(5,6)`. The paper now explains that RTX 3090 uses third-
  generation NVLink, that NV4 denotes four bonded 14.0625 GB/s-per-direction
  links, and that the unidirectional KV workload must be compared with
  56.25 GB/s rather than the 112.5 GB/s bidirectional aggregate. The official
  NVIDIA GA102 architecture white paper is included as the hardware source.
- Decision result: The evaluation now reports the 64-block result as
  36.5 GiB/s, approximately 39.2 GB/s or 69.7% of the unidirectional peak, and
  identifies it as end-to-end effective KV bandwidth including gather, NCCL
  transfer, scatter, and synchronization rather than raw link bandwidth.

## 2026-07-26: Capacity-Offload Evidence and Calibrated Transaction Prior

- Decision demand: Copy-style session handoff proves cross-instance KV reuse
  but does not prove capacity offload because it retains every source block.
  The admission model also used a manually selected 40 ms fixed residual that
  online nonnegative EWMAs could only increase, so an overestimated cold prior
  could permanently reject useful plans.
- Decision plan: Keep session handoff as the copy/reuse experiment, add an
  independent capacity-offload workload whose success requires source release,
  and split transfer calibration into an idle payload profile and a serving
  transaction residual profile collected on a disjoint run.
- Decision implementation: Added the `capacity-offload` workload and
  `offload_verified`, which is true only when foreground move-style transfer
  releases source blocks. Summary output now reports copied and released
  blocks separately, and rank statistics record the free-block count after
  source release. The control plane exports bounded per-plan
  dispatch-to-publish observations containing pair, bytes, size bucket,
  measured data-path latency, elapsed latency, predicted latency, and
  nonnegative residual. `build_transfer_profile.py` can aggregate independent
  E2E calibration JSON files into pair-by-size-bucket P95 residual points. The
  scheduler computes
  `residual_P95 + interference * data_path_P95`; the scalar fixed latency now
  defaults to zero and is only a cold-start fallback. Once a complete placement
  sample exists, online observations can correct an uncalibrated scalar
  fallback in either direction. Added scenario filtering so calibration runs
  only `multi-gpu-lmpool`, and updated the paper runner to use a disjoint seed
  before formal evaluation. JSON and console diagnostics now report cost MAE,
  P95 absolute error, and underprediction rate.
- Decision result: The complete CPU test suite passes with 198 tests and one
  skip. Static Python compilation, shell syntax validation, and whitespace
  checks pass. GPU performance and offload effectiveness remain empirical
  acceptance conditions: the next capacity-offload result must show positive
  released blocks and foreground success before it can support an offloading
  claim. Historical `20260725T031840Z` results still use the manual 40 ms prior
  and must not be mixed with results from the calibrated policy.

## 2026-07-26: Long-Prefix Routing and Canonical Memory-Skew Evaluation

- Decision demand: The latest routing trace reduced uncached prefill work but
  left decode dominant, so routing did not improve throughput or latency.
  Capacity-offload admitted no transfer, while session handoff executed copy
  plans but derived most of its TTFT gain from routing. The paper workload
  therefore did not isolate either claimed performance mechanism.
- Decision plan: Make reusable prefill work the independent variable in the
  routing experiment, make memory skew the single canonical capacity-transfer
  workload, calibrate transaction cost at plan sizes that match foreground
  moves, and demote session handoff to optional mechanism evidence.
- Decision implementation: The paper runner now executes routing with the same
  192-request, 16-group trace at repeat factors 16, 32, and 48, fixes generation
  at eight tokens, raises model-length and equal KV-budget limits as needed,
  and records all three artifacts. `benchmark_kv_routing.py` validates the
  configured model and batch lengths against the tokenized trace. The canonical
  `memory-skew` trace now contains 64 warm-up requests over 16 hot prefixes,
  32 distinct pressure requests, and 64 reuse requests. New phase-size options
  are recorded in metadata and applied identically to all scenarios.
  `capacity-offload` remains only as a compatibility alias. Transaction
  calibration now collects disjoint 4/8/16/32/64-block-limited observations
  instead of applying one large handoff residual to small foreground plans.
  The main runner skips session handoff unless
  `RUN_HANDOFF_ABLATION=1`.
- Decision result: Qwen3 tokenization gives routing prompt lengths of 1,911,
  3,783, and 5,655 tokens for the 1x/2x/3x sweep. The trace request-share rate
  stays fixed at 91.67%, while reusable token volume increases. The revised
  memory-skew trace contains 160 requests and a 72.96% trace token-share ratio.
  Focused benchmark tests pass 37 cases, the complete CPU suite passes 200
  cases with one hardware-gated skip, and Python and shell syntax checks pass.
  GPU performance remains an explicit acceptance gate: memory skew must report
  positive source release and foreground success before the paper claims
  capacity-transfer benefit.

## 2026-07-26: Transfer-Specific Load and Memory Pressure Workloads

- Decision demand: The final paper matrix must give each mechanism a workload
  that actually activates it. Routing needs long reusable prefixes rather than
  relying on short generation alone. Load skew must exercise foreground and
  background transfer under a hot-owner burst. Memory skew must create enough
  source pressure to verify move-style capacity offload. Session handoff is
  uncommon and should not remain a primary performance claim. Every main
  experiment must still run on Qwen3-0.6B and Qwen3-1.7B.
- Decision plan: Keep the routing 1x/2x/3x prefix sweep, replace the old
  single-stage load trace with explicit source warm-up and burst reuse phases,
  strengthen memory pressure with longer prefixes and more one-shot requests,
  disable session handoff by default, and add an unambiguous per-model
  completion marker to the dual-model runner.
- Decision implementation: `load-skew` now warms six 5.7K-token prefix groups
  with 48 requests on the source sides of three NVLink pairs, exposes the
  remaining 144 requests as an ingress demand forecast, and submits reuse with
  window 64. Background copies may pack up to two 24-block chains per
  transaction, while foreground transfer remains gated by real block shortage
  and the calibrated cost model. `memory-skew` now uses eight long groups, 64
  warm-up requests, 64 distinct pressure requests, 64 reuse requests, repeat
  32, submit window 32, and a common 64-block budget; background copy is
  disabled so only source release counts as offload. E2E runtime limits are
  raised automatically when tokenized long prompts exceed the old 2,048-token
  default. The paper runner executes both local model snapshots and writes
  `SUITE_COMPLETE` only after each complete model matrix. Session handoff
  remains available through `RUN_HANDOFF_ABLATION=1` and is still used
  internally for disjoint transaction calibration, but is not a reported main
  workload.
- Decision result: Offline Qwen3 tokenization reports 96.88% request-level and
  96.56% token-level prefix sharing for revised load skew, and 62.50% and
  71.01% respectively for revised memory skew. The routing sweep reaches
  1,911, 3,783, and 5,655 maximum prompt tokens. Focused benchmark tests pass
  40 cases, the complete CPU suite passes 203 cases with one hardware-gated
  skip, and Python and shell syntax checks pass. GPU acceptance remains
  empirical: load skew must show completed background copies and foreground
  plans before claiming transfer relief; memory skew must additionally report
  positive source block release and `offload_verified=true`.

## 2026-07-26: Complete Small-Plan Transaction Calibration

- Decision demand: The idle NVLink data-path sweep measured one- and two-block
  payloads, but the complete dispatch-to-publish calibration started at four
  blocks. The residual interpolator therefore assigned the four-block
  transaction residual to every smaller plan. This conservative clamp can
  overestimate one-block background copies and two-block foreground moves and
  reject useful plans near the admission boundary.
- Decision plan: Use the same power-of-two size coverage for the data-path and
  complete-transaction profiles, and ensure that the control-plane candidate
  limit cannot silently enlarge a nominal one- or two-block calibration.
- Decision implementation: Changed
  `COST_CALIBRATION_BATCH_BLOCKS` to `1 2 4 8 16 32 64`. For each calibration,
  the per-prefix candidate limit is now `min(4, batch_limit)`, and resume
  validation checks both limits before reusing an artifact. Updated the
  automatic and manual commands in `PAPER_RUNBOOK.md`.
- Decision result: The next generated cost profile will contain measured
  transaction residuals for the smallest legal background and foreground plan
  sizes instead of extrapolating them from four blocks. Shell syntax and
  whitespace validation pass. GPU calibration must be rerun to populate the
  new points; benefit, topology, generation, and capacity gates still determine
  whether a transfer is admitted.

## 2026-07-26: Nonempty Long-Prompt Warm-Up and Worker Fail-Fast

- Decision demand: The revised load-skew trace raised `max_model_length` above
  4,096 tokens. The benchmark updated Scheduler key
  `max_num_batched_tokens` but left the legacy ModelRunner key
  `max_num_batch_tokens` at 4,096. Integer division then produced a zero-sized
  warm-up batch, and the launcher waited for worker capacity messages long
  after the worker had failed.
- Decision plan: Make one batch-token limit authoritative, construct warm-up
  sequences from the exact token budget without allowing an empty batch, and
  surface unexpected worker exits at the launcher pump boundary.
- Decision implementation: ModelRunner now normalizes the legacy and canonical
  batch-token keys, partitions the warm-up budget into nonempty sequences no
  longer than `max_model_length`, and rejects an empty prefill batch explicitly.
  The E2E benchmark synchronizes both aliases after resolving long-prompt
  limits. `LLMEngine.step()` now raises with failed rank and exit code as soon
  as a data-plane process exits. Unit tests cover a token budget below the
  sequence limit, a remainder batch, invalid limits, empty prefill, and launcher
  failure propagation.
- Decision result: The corrected load-skew configuration warms a nonempty
  5.7K-token-compatible batch rather than constructing zero sequences, and
  future initialization failures terminate the benchmark promptly instead of
  waiting for the capacity deadline. Focused tests pass 51 cases; the complete
  CPU suite passes 207 cases with one hardware-gated skip. Python compilation,
  shell syntax, and whitespace validation also pass.

## 2026-07-26: Stronger Locality, Load-Relief, and Capacity-Offload Traces

- Decision demand: The first revised load-skew run executed transfer but
  round-robin still reached a 97.92% reuse-phase request hit rate after only
  three partner-side cold prefills. Its aggregate throughput therefore stayed
  within 1% of the baseline. The memory-skew run verified source release, but
  equal 64-request warm-up, pressure, and reuse phases diluted the useful reuse
  interval. Routing also stopped at 3x and stored that point under the
  ambiguous filename `summary`.
- Decision plan: Strengthen the workload signal without changing the routing
  score, transfer admission, transaction protocol, or per-scenario fairness.
  Extend the default routing sweep to 5x, preserve 10x as an optional
  context-length sensitivity point, make every routing artifact name its
  multiplier, increase the number of load-skew prefixes that must cold-prefill
  on partners, and allocate a larger share of memory-skew requests to the
  post-pressure reuse phase.
- Decision implementation: The paper runner now produces
  `prefix_1x`, `prefix_3x`, and `prefix_5x` routing artifacts; setting
  `ROUTING_PREFIX_MULTIPLIERS="1 3 5 10"` adds the optional 10x point. Routing
  output remains fixed at 8 tokens. Load skew now uses 24 5.7K-token groups,
  48 warm-up requests, 144 reuse requests, 8 output tokens, and a common
  192-block per-rank budget. Group pairs are striped cyclically across source
  ranks so ordinary round-robin sends each reuse only to its owner or direct
  NVLink partner. Memory skew now uses 12 3.8K-token groups, 24 warm-up
  requests, 64 pressure requests, and 104 reuse requests with the same
  64-block budget used by all five scenarios. The runbook and benchmark
  examples use the same values, and tests cover the repeated group-to-pair
  mapping.
- Decision result: Offline tokenization gives maximum routing prompt lengths
  of 1,911, 5,655, 9,399, and 18,759 tokens at 1x, 3x, 5x, and 10x, all below
  the 40,960-token model limit. A load-skew source holds eight 22-block chains
  within its 192-block budget; the controlled round-robin reuse expectation
  falls from 141/144 hits (97.92%) to 132/144 (91.67%). A memory-skew source
  holds four 14-block hot chains within its 64-block budget before concurrent
  seven-block pressure requests create shortage. Focused benchmark tests pass
  38 cases; the complete CPU suite passes 208 cases with one hardware-gated
  skip. Python compilation, shell syntax, and whitespace validation pass.
  Fresh GPU results must verify the predicted transfer and offload behavior
  before these configurations are used for paper claims.

## 2026-07-26: Remove Session Handoff and Isolate Transaction Calibration

- Decision demand: Session handoff is no longer part of the paper claim or
  desired benchmark surface. The runner had stopped reporting it by default,
  but complete transfer-cost calibration still depended on the same workload
  name, prompt arguments, phase helpers, and artifact metadata. Deleting only
  the optional benchmark block would therefore have broken profile generation.
- Decision plan: Preserve the minimum two-phase source-build and partner-reuse
  trace required to observe complete dispatch-to-publish transactions, but
  classify it exclusively as internal transfer calibration. Remove every
  handoff workload option, result directory, runner switch, active benchmark
  document entry, and corresponding test name.
- Decision implementation: Replaced the internal workload with
  `transfer-calibration`, renamed its prefix/warm-up arguments and phase
  helpers, and changed calibration artifact metadata accordingly.
  `run_paper_suite.sh` now invokes only this calibration trace for the
  1/2/4/8/16/32/64-block residual matrix and no longer creates or executes a
  handoff ablation directory. The public serving matrix contains only routing,
  load skew, and memory skew. Benchmark tests now validate calibration trace
  construction under calibration-specific names. The active README files and
  paper runbook no longer present handoff as a current workload or result.
- Decision result: `benchmark_e2e.py --help` exposes no handoff workload or
  argument, focused benchmark tests pass 38 cases, and the complete CPU suite
  passes 208 cases with one hardware-gated skip. Python compilation, shell
  syntax, and whitespace validation pass. Existing historical result
  directories and prior decision/review records remain immutable archives;
  fresh paper data will use `workload=transfer-calibration` only in cost-profile
  artifacts and will not report its serving metrics. Manual runbook commands
  now also pass `GPU_SET` through `CUDA_VISIBLE_DEVICES`, so preflight and
  manual runs preserve the intended physical GPU to logical-rank mapping
  instead of depending on ambient shell state.

## 2026-07-26: Add a Reproducible Workload Preflight Suite

- Decision demand: The fixed preflight was documented as three manual commands.
  That left repeated environment setup, physical-to-logical GPU mapping, output
  isolation, and mechanism acceptance to the operator, even though none of
  those choices should vary between paper runs.
- Decision plan: Add one small runner for the minimum Qwen3-0.6B gate. Reuse an
  existing compatible transfer profile, execute one repetition of 5x routing,
  memory skew, and load skew, and reject the paper run if the expected
  mechanisms or all primary performance directions fail.
- Decision implementation: Added `run_preflight_suite.sh` with fixed current
  workload parameters, offline model and GPU validation, isolated environment
  capture, artifact-level `RESUME=1`, and JSON acceptance checks. Routing must
  reduce uncached prefill without transfer or rank concentration. Memory skew
  must complete a release-style foreground transfer. Load skew must complete
  background copy, route to a replica or placement lease, and exceed the
  measured round-robin reuse-hit rate. It does not require a foreground move:
  copied blocks are intentionally pinned by reuse requests. Each mechanism
  check also requires an improvement in throughput, mean TTFT, or P90 E2E
  latency against the corresponding baseline.
- Decision result: The paper runbook now has one preflight command and one
  explicit resume command. A successful run writes `PREFLIGHT_COMPLETE`; a
  failed mechanism check exits nonzero and prevents an accidental transition
  to the expensive dual-model suite. Resume validation checks the exact trace,
  capacity, output, and profile inputs before reusing an artifact. Shell syntax,
  all three JSON predicates, and the complete CPU suite were validated; the
  suite reports 208 passed tests and one hardware-gated skip.

## 2026-07-26: Make Foreground Admission Aware of Known Future Reuse

- Decision demand: The first automated preflight passed routing but rejected
  both transfer workloads. The load-skew artifact already contained 12
  successful background copies, 528 copied blocks, 72 placement-lease routes,
  and a 100% reuse-phase hit rate, so its failure came from an incorrect gate
  that also required foreground transfer. Memory skew was a real mechanism
  failure: both transfer scenarios reported zero successful foreground plans,
  while the rejection reasons were `low_benefit` and `no_plan`.
- Decision plan: Preserve the workload split. Use load skew to validate
  proactive background replication and load relief. Use memory skew to validate
  release-style foreground offload. At the memory-skew phase boundary, publish
  the benchmark ingress queue snapshot to foreground admission even when
  background copy is disabled, so the cost model can compare transfer cost
  against known reuse instead of a weak historical-access guess.
- Decision implementation: Added a synchronous
  `future_prefix_demands` control-plane message and a scheduler snapshot setter.
  `benchmark_e2e.py` now publishes exact future block-hash demand before it
  submits the memory-pressure phase. Foreground planning takes the larger of
  discounted historical reuse and the ingress forecast, records both estimates
  in the plan, and values saved prefill work with the observed destination-rank
  prefill cost. The preflight gate now checks background copy, replica or lease
  routing, measured reuse-hit improvement, and a primary performance benefit
  for load skew; foreground release remains mandatory only for memory skew.
- Decision result: A focused test reproduces the original `low_benefit`
  rejection and verifies that the same cold chain is admitted after its exact
  demand snapshot is installed. Control-plane tests cover publication with
  background copy disabled. The previous load-skew artifact passes the corrected
  acceptance predicate. The complete CPU suite passes 210 tests with one
  hardware-gated skip. A fresh GPU preflight is required to validate foreground
  source release because artifacts from the earlier code revision cannot prove
  the new path.

## 2026-07-27: Separate Pressure Sufficiency from Reuse Amortization

- Decision demand: The corrected preflight triggered all three mechanisms, but
  whole-trace memory-skew and load-skew improvements appeared small and the
  10-second goodput SLA admitted every multi-GPU request. Phase-level results
  showed that pressure was already sufficient: memory skew completed 12
  foreground moves and released 96 blocks, while load skew completed 12
  background plans and copied 528 blocks. The reuse phases contained the useful
  signal, which preparation work diluted in whole-trace averages.
- Decision plan: Do not add more cold pressure or weaken transfer admission.
  Keep 24 memory warm-up and 64 pressure requests, extend only the reuse horizon,
  and pre-register model-specific primary goodput SLAs. Report a fixed SLA
  sensitivity set from each run so the paper does not depend on one
  retrospectively selected threshold.
- Decision implementation: Increased memory-skew requests from 192 to 256,
  changing the final phase from 104 to 168 reuse requests while preserving all
  pressure and capacity parameters. The paper runner now uses a 3-second primary
  SLA for Qwen3-0.6B and a 5-second primary SLA for Qwen3-1.7B. Both runners pass
  `2/3/5/10`-second sensitivity thresholds. `benchmark_e2e.py` computes every
  threshold from the same completion timestamps and output-token counts, and
  stores per-threshold goodput means and 95% Student-t confidence intervals.
- Decision result: The revised 256-request memory trace contains 847,456 prompt
  tokens, a 70.31% request prefix-share rate, and a 76.12% token prefix-share
  ratio. SLA sensitivity adds no GPU work and does not affect routing or
  transfer decisions. Because the trace and primary SLA changed, the previous
  preflight remains a mechanism diagnostic but cannot authorize the revised
  formal suite; a fresh Qwen3-0.6B preflight is required.
