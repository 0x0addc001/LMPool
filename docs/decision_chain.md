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
