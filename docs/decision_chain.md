# LMPool Decision Chain

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
  alongside mean TTFT, mean TTPT, and mean E2E.
- Decision implementation: Extended `ScenarioResult` with `p90_ttft_s`,
  `p90_ttpt_s`, and `p90_e2e_s`. Reused the existing `_percentile()` helper for
  both P90 and P95 calculations. Updated the summary table and PNG figure to
  include `p90(e2e)`, and documented that TTFT/TTPT/E2E columns are
  mean/average values while P90/P95 are tail-latency metrics.
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
  subplot keeps TTFT, TTPT, mean E2E, and P90 E2E as separate visible series.
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
