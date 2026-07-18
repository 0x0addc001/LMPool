# Benchmarks

`benchmark_e2e.py` runs an online shared-prefix workload across the current LMPool engine stack. It compares five configurations in one horizontal summary table and can export the same results as JSON and a PNG figure.

Canonical entries:

- `benchmark_e2e.py`: full end-to-end comparison.
- `benchmark_kv_routing.py`: routing-focused comparison.
- `benchmark_kv_transfer.py`: direct KV transfer microbenchmark.

Compatibility entries `shared_prefix_benchmark.py` and `kv_transfer_benchmark.py` are kept for older commands.

## Scenarios

- `single-gpu`: one GPU, no global KV pool. Prefix-hit rate is measured with the local `BlockManager` as a single-card cache reference.
- `multi-gpu`: multiple GPUs with online round-robin request dispatch, no global KV sharing and no control-plane routing.
- `multi-gpu-kv-routing`: control-plane routing and global page-table lookup enabled, with foreground rebalance and background copy disabled. This is the routing-only baseline.
- `multi-gpu-kv-transfer`: global pool and foreground transfer enabled, with requests dispatched round-robin to isolate transfer behavior from cache-aware routing.
- `multi-gpu-lmpool`: full LMPool path with control-plane routing, global page table, data-plane workers, and rebalance support.

## Metrics

- `tput(tok/s)`: generated output tokens per second.
- `goodput`: generated output tokens per second for requests whose end-to-end latency is within `--goodput-e2e-sla-ms`.
- `ttft(ms)`: mean/average time from request submission to the first generated token event reported by the data-plane worker.
- `ttpt(ms)`: mean/average per-output-token latency proxy, computed as request E2E latency divided by the number of generated output tokens.
- `e2e(ms)`: mean/average end-to-end request latency.
- `p90(e2e)`: p90 end-to-end request latency.
- `p95(e2e)`: p95 end-to-end request latency.
- `gpu util`: mean GPU utilization sampled from `nvidia-smi`.
- `mem util`: mean GPU memory utilization sampled from `nvidia-smi`.
- `CP req hit` (`route_hit_rate` in JSON): fraction of routed requests for
  which the control plane found at least one contiguous reusable prefix block
  at decision time. Round-robin baselines report zero because they do not query
  the routing policy.
- `CP owner` (`routed_to_prefix_owner_rate`): fraction of all routed requests
  whose selected GPU was one of the GPUs owning the matched prefix. It can be
  lower than `CP req hit` when load balancing deliberately bypasses an owner.
- `DP req hit` (`prefix_hit_rate`): worker/data-plane prefix-cache hit rate on each request's initial
  prefill only. Retry hits after preemption are excluded, so cache churn cannot
  inflate this metric.
- `DP tok reuse` (`initial_cached_token_ratio`): fraction of all prompt tokens
  already cached on the initial prefill. Unlike binary `DP req hit`, this
  measures how much prefill work is
  actually avoided.
- `trace upper` (`theoretical_prefix_hit_rate`): capacity-unbounded prefix-hit upper bound implied by the
  workload trace; it is a reference, not an observed runtime hit rate.
- `CP blk match` (`route_matched_block_ratio`): fraction of complete prompt blocks that the control plane
  believed reusable on the selected worker at routing time.
- `CP reclaim` (`reclaimable_capacity_route_rate`): fraction of control-plane
  routes admitted using local dependency-safe cache reclamation in addition to
  immediately free blocks. This exposes whether routing restored parallelism
  without changing the per-rank KV block budget.
- `CP stale` (`stale_route_hit_rate`): fraction of control-plane request hits that reached the worker with zero
  initial cached tokens, indicating stale or structurally unusable page-table
  information.
- `attempts`: total prefill executions, including retries.
- `preempt`: number of live sequences preempted by the local scheduler.
- `redund tok`: prefill tokens reprocessed beyond the initial prompt work.
- `sent blocks` (`transfer_count`): number of KV blocks actually sent by the data plane.
- `source kept` (`transfer_copy_count`): sent blocks retained at the source. This
  includes shared ancestors copied by a chain-preserving foreground transfer
  and every block in background replication.
- `source freed` (`transfer_release_count`): source blocks actually released to
  relieve local capacity pressure.
- `chain plans` (`chain_transfer_count`): successful foreground plans that sent
  a usable root-to-leaf fragment and released selected leaves.
- `hot sent` / `hot ratio`: transferred blocks belonging to the common complete
  prefix learned during memory-skew warm-up, as a count and as a fraction of
  all sent blocks. A high release count with a low hot ratio relieves capacity
  but does not preserve the data reused in the final phase.
- `reuse req hit` / `reuse tok ratio`: request hit rate and cached-token ratio
  in the final reuse phase of `memory-skew` or `session-handoff`; other
  workloads report zero.
- `lease route`: requests routed to a committed replica through a forecast-bound
  placement lease. A completed handoff keeps the forecast reuse batch on the
  replica instead of splitting each prefix between source and target.
- `place cand` / `place done`: accepted and completed prefix candidates.
  `plan run` / `plan done` count actual pair-level protocol transactions; one
  plan can contain several candidates and one contiguous KV payload.
- `Transfer Workload Phase Latency`: request count, phase throughput, and
  mean/P90 TTFT and E2E
  for warm-up, optional pressure, and reuse separately. Use the reuse row to
  evaluate transfer benefit; aggregate P90 can be dominated by an earlier phase.
  Supplying `--output-figure` also writes an adjacent `_reuse_phase` figure.
- `fg ok` / `fg fail`: number of successful / failed foreground rebalance requests. Foreground rebalance is the current-request path that tries to free local KV blocks with move-style transfer.

Foreground transfer candidates use worker-reported KV heat. The control plane
orders complete prefix chains by access frequency and reuse delivered per
missing target block, then uses recency as a tie-breaker. Local cache pressure
uses the same LFU-first, LRU-second ordering. This keeps one-shot pressure data
from displacing a repeatedly reused prefix merely because it was accessed more
recently.
- `bg ok` / `bg fail`: number of successful / failed background speculative copy plans. Background copy is the non-blocking path that replicates hot prefix blocks to an NVLink peer for future requests.
- `pinned`: rebalance failures caused by source blocks still being referenced (`ref_count > 0`), which are safe copy candidates but not safe move/eviction victims.
- `no space`: rebalance failures caused by no NVLink target having enough free blocks.
- `no plan`: rebalance failures where the control plane could not build an executable plan.
- `bg space`: background copy failures caused by the target rank not having enough free blocks during prepare.

## Example

If physical GPU 0 and 2 are NVLink-connected, expose them as two logical devices and pass the logical pair `0,1`:

```bash
CUDA_VISIBLE_DEVICES=0,2 UV_CACHE_DIR=/tmp/uvcache uv run python benchmarks/benchmark_e2e.py \
  --num-prompts 32 \
  --prompt-repeat 10 \
  --max-tokens 16 \
  --temperature 0.6 \
  --ignore-eos \
  --seed 0 \
  --repetitions 3 \
  --locality-prefix-groups 16 \
  --output-json /tmp/shared_prefix_benchmark.json \
  --nvlink-pairs 0,1 \
  --kv-block-budget 64 \
  --gpu-memory-utilization 0.20 \
  --submit-window 4 \
  --goodput-e2e-sla-ms 120000 \
  --output-figure /tmp/shared_prefix_benchmark.png
```

The script prints `saved json: ...` and `saved figure: ...` after successful export. Parent directories are created automatically.

## Parameters

- `--num-prompts`: total number of requests in the benchmark.
- `--prompt-repeat`: lengthens the shared prefix by repeating a fixed instruction block; larger values make prefix reuse easier to observe.
- `--max-tokens`: maximum generated tokens per request.
- `--temperature`: sampling temperature.
- `--ignore-eos` / `--no-ignore-eos`: keep generating until `--max-tokens`
  (default), or allow EOS to end a request early. Keep the default for fair
  system comparisons with equal decode work.
- `--seed`: base random seed. Data-plane rank `r` uses `seed + r`.
- `--repetitions`: number of complete runs per scenario. Results are reported as
  means; JSON and the console variability table include throughput, goodput,
  TTFT, and E2E population standard deviations. Use at least `3` for paper
  results; the default `1` is intended for development runs.
- `--workload`: `locality`, `load-skew`, `memory-skew`, or `session-handoff`.
  `memory-skew` is a
  deterministic three-phase trace: hot-prefix warm-up on source ranks,
  unique-prefix pressure on those ranks, then hot-prefix reuse. For
  topology-blind and transfer-only scenarios, reuse is deterministically sent
  to the opposite side of each NVLink pair: the baseline recomputes there,
  while only a completed transfer can eliminate the first partner-side
  recomputation. Source ranks are derived once from the
  command-line NVLink pairs and applied identically to every multi-GPU
  scenario; topology-blind baselines receive only this workload placement, not
  topology-aware policy decisions.
  Per-rank JSON diagnostics expose `warmup_submitted`, `pressure_submitted`,
  and `reuse_submitted` so placement fairness can be checked directly.
  `session-handoff` uses equal warm-up and reuse phases: prefixes are built only
  on source ranks, then the same sessions continue on their NVLink partners.
  Use it to isolate copy benefit; use `memory-skew` for capacity pressure.
- `--locality-prefix-groups`: number of distinct long shared-prefix groups in
  the `locality` workload (default `16`). Requests are balanced across groups
  and deterministically shuffled with `--seed`, preventing prefix IDs from
  accidentally aligning with round-robin ranks. More groups expose redundant
  per-GPU caching without routing; keep the value no larger than
  `--num-prompts`.
- `--memory-skew-prefix-groups`: number of long hot prefixes preserved across
  the three memory-skew phases. `0` automatically chooses the largest odd value
  up to `15` that fits both warm-up and reuse. Each group is warmed repeatedly
  on one source rank; the reuse phase interleaves groups so round-robin cannot
  saturate its cache after one miss to a single global hotspot.
- `--handoff-prefix-groups`: independent session prefixes shared by the two
  `session-handoff` phases. `0` selects up to `32`; the maximum is half of
  `--num-prompts`. A value equal to half the requests removes destination
  self-warmup from the baseline.
- `--output-json`: write raw scenario results to JSON. Parent directories are created automatically, and the script prints `saved json: ...` on success.
- `--model-name-or-path`: model name or local model path. The default config targets `Qwen/Qwen3-0.6B`.
- `--nvlink-pairs`: logical NVLink pairs after `CUDA_VISIBLE_DEVICES` remapping, e.g. `0,1` or `0,1;2,3`. Quote values containing semicolons, e.g. `--nvlink-pairs "0,2;1,3;4,5;6,7"`. Pass an empty string to let the engine try `nvidia-smi topo -m` detection.
- `--world-size`: number of data-plane worker ranks for multi-GPU scenarios. The default is `2`; use `--world-size 8` for eight visible GPUs.
- `--kv-block-budget`: requested per-rank KV block count used by all five
  scenarios. When explicitly set, the benchmark verifies every worker realizes
  this capacity before submitting requests and fails instead of silently
  shrinking it. The hidden legacy options `--routing-max-cached-blocks` and
  `--eviction-max-cached-blocks` are accepted only when they resolve to the same
  value.
- `--gpu-memory-utilization`: fraction of currently free HBM used when deriving
  physical KV capacity, default `0.20`. Actual allocation remains capped by
  `--kv-block-budget`; increase this only when the requested budget cannot be
  realized.
- `--goodput-e2e-sla-ms`: end-to-end latency SLA for counting goodput tokens.
- `--skip-pool`: skip `multi-gpu-lmpool`.
- `--output-figure`: write the summary figure to PNG. Parent directories are created automatically, and the script prints `saved figure: ...` on success.
- `--submit-window`: maximum in-flight requests. Use `4` or `8` to let earlier requests populate the global page table before later routing decisions; use `0` or a negative value for burst submission of all requests.
- `--disable-background-copy`: disable background speculative copy-style transfer.
- `--background-copy-max-blocks`: maximum prefix blocks contributed by one
  candidate chain. Candidates for the same directed NVLink pair are internally
  coalesced into a bounded plan, amortizing protocol and payload setup.
- `--background-copy-cooldown-s`: cooldown before the same prefix can trigger another copy on the same source-target pair. Try `0.5` when evaluating background copy impact.
- `--background-copy-hot-threshold`: minimum worker-reported access count for every block in a maximal hot prefix chain. Larger values reduce speculative placement.
- `--background-copy-min-load-skew`: minimum owner-to-partner sequence-pressure gap for route-originated candidate discovery. Phase-boundary ingress forecasts use observed placement skew instead of this transient load gap.
- `--background-copy-expected-reuses`: conservative cap on predicted future reuse. The actual estimate comes from not-yet-submitted ingress prefix counts, or discounted worker access history when no forecast exists.
- `--route-decode-token-weight`: expected decode-work weight included in pending and worker load snapshots.
- `--route-owner-spill-sequence-skew`: sequence-pressure gap that permits direct spill to the owner's NVLink partner.
- `--route-owner-spill-max-extra-cost`: maximum extra token-equivalent recomputation cost accepted for direct pair spill.
- `--route-load-weight`: legacy tie-break weight for token-aware load in the prefix score.
- `--route-load-bypass-threshold`: minimum token-equivalent cost advantage required before a cold target may bypass a prefix owner.
- `--route-prefill-cost-weight`: cost assigned to each missing prefix token; the default `1.0` keeps it in the same units as queued tokens.
- `--route-reclaim-cost-weight`: extra cost per reclaimable block, expressed as a fraction of one block of prefill work.
- `--foreground-transfer-cost-weight`: multiplier applied to the time-domain transfer cost.
- `--foreground-transfer-min-benefit-ratio`: minimum predicted saved-prefill-ms / transfer-ms ratio required for foreground transfer.
- `--foreground-transfer-bandwidth-gib-s`: measured effective KV transfer bandwidth used by the admission model.
- `--foreground-transfer-fixed-latency-ms`: fixed per-plan protocol and coordination latency.
- `--foreground-transfer-interference-multiplier`: multiplier for packing, unpacking, and inference interference.
- `--foreground-prefill-token-time-ms`: estimated recomputation time per uncached prompt token.
- `--foreground-future-reuse-discount`: conservative discount from historical leaf-prefix accesses to future reuse.
- `--kv-transfer-prewarm-blocks`: representative KV blocks sent on every NVLink pair before serving; measured pair cost seeds admission and startup is excluded from serving metrics.
- `--route-cache-queue-slack`: token-equivalent cost slack allowed by the route-cache fast path.

Serving timing and GPU sampling start only after every worker reports its
realized KV capacity. Model loading, model warmup, process-group setup, P2P
communicator prewarm, and KV allocation are excluded. Workers return completed
source-transfer bytes and wall time to the control plane; a per-NVLink-pair EWMA
of observed excess latency can only raise subsequent admission cost above the
static model. Only one foreground plan may execute on a pair at a time.
Startup prewarm and serving transfer both use one all-layer contiguous payload.
Each configured pair uses a dedicated NCCL process group; prepared plans omit
block-ID negotiation, and idle workers wake directly for control-plane transfer
commands instead of waiting for the ingress queue timeout.
Completed uncached-prefill batches update a per-rank recomputation-cost EWMA.
A completed proactive replica creates a forecast-bound placement lease so the
known reuse batch consumes the copied KV instead of leaving an unused page-table
entry or being fragmented back across the warmup source.
The transfer diagnostics report `spill` for direct owner-to-partner routing,
plus candidate-level `place q` / `place cand` / `place done` and transaction-level
`plan run` / `plan done` for proactive placement lifecycle.
`copy route` is retained as a compatibility diagnostic but should remain zero:
routing no longer keeps a request on an overloaded owner while waiting for a
future copy. Actual replica completion remains visible through
`background_copy_success` and transferred block counts.

Routing cost-model defaults are set in `MODEL_CONFIG` inside `shared_prefix_benchmark.py`:

- `route_load_weight`: multiplier for token-aware load in the route score.
- `route_waiting_token_weight`: weight for queued prefill tokens.
- `route_running_token_weight`: weight for tokens already owned by running sequences.
- `route_running_sequence_weight`: fixed load weight per active running sequence.
- `route_load_bypass_threshold`: minimum total-cost improvement required to bypass locality.
- `route_prefill_cost_weight`: missing-prefix recomputation cost.
- `route_reclaim_cost_weight`: local cache-reclamation cost and future-miss risk.
- `foreground_transfer_cost_weight`: calibrated foreground transfer cost.
- `foreground_transfer_min_benefit_ratio`: required safety margin for preserving KV through transfer.

## Notes

- Prefix-hit rates depend on online timing and cache placement. With `--submit-window 0`, all requests are submitted in a burst before workers have finished prefill, so control-plane routing has less opportunity to use newly reported page-table state.
- Routing evaluates the full cumulative hash chain and counts only blocks that
  are contiguous from block zero on the same GPU. Capacity checks and
  optimistic reservations subtract those reusable blocks instead of charging
  the full prompt again.
- Complete prefix blocks remain cached after their active reference count reaches
  zero. They are reported as evictable global-page-table entries and reclaimed
  by a prefix-chain-aware leaf policy: an ancestor cannot be removed while a
  retained child depends on it, and eligible leaves use LFU-first/LRU-second
  order. Partial
  blocks are released immediately.
- Prefill admission reserves the next possible decode-growth block for every
  active sequence. If that headroom is unavailable, the scheduler drains
  running decode work instead of preempting it to admit another long prompt.
  In `memory-skew`, transfer scenarios treat both prompt allocation and this
  decode headroom as real admission demand. They attempt foreground transfer
  before local cache reclamation, so a prompt that already fits cannot silently
  discard the hot prefix merely because its future decode block is missing. A
  failed plan still falls back to local reclamation.
  Structural failures use exponential cooldown up to 30 seconds by default,
  so an unchanged `no_plan` or `no_target_space` state does not produce a tight
  loop of failed control-plane transactions.
- For publishable comparisons, keep `--ignore-eos`, set an explicit `--seed`,
  and use `--repetitions 3` or more. A repeated JSON result includes
  `repetitions`, `throughput_tok_s_std`, `goodput_tok_s_std`,
  `mean_ttft_s_std`, and `mean_e2e_s_std`.
- `multi-gpu` is an online round-robin baseline except for memory-skew's
  explicitly defined cross-pair reuse phase; it is not static offline sharding.
- `multi-gpu-kv-transfer` uses the same ingress placement as `multi-gpu` and
  enables foreground transfer. All scenarios retain the same block budget.
- Rebalance requests are based on the actual block shortage, not the full
  sequence block count. Foreground plans transfer every missing ancestor needed
  to make a selected leaf reusable at the target. They release the deepest
  dependency-safe suffix up to the requested shortage: a linear chain can free
  multiple blocks, while an ancestor with an untransferred branch remains at
  the source. Pinned blocks are never released.
- Every scenario trial constructs a new `LLMEngine`, worker set, local block
  managers, KV tensors, and control plane. `engine.exit()` joins or terminates
  workers, and workers destroy their NCCL process group. Each local trial uses
  a unique temporary FileStore rendezvous path, avoiding TCPStore port races
  during repeated multi-rank startup; NCCL remains the data-transfer backend.
  KV contents and page tables therefore do not carry into the next scenario.
  OS model-file cache, GPU temperature/power state, and general machine load can
  persist, so use repeated runs (and ideally randomized scenario order in final
  paper scripts) to control non-KV order effects.
- To evaluate background speculative copy itself, keep the common
  `--kv-block-budget` fixed and vary only workload pressure and background-copy
  parameters. A very small common budget is useful for failure analysis, but it
  often leaves too little target space for copy replication to improve hits.
  In `memory-skew`, the ingress forecasts hashes from the not-yet-submitted reuse
  phase after warm-up. The control plane drains accepted low-load placement
  plans before the next phase, and this drain time is included in benchmark
  elapsed time. `Prefill Compute Diagnostics` separates prompt, cached, and
  actually executed uncached tokens and reports the explicit placement wait.
  `Per-Pair Placement Diagnostics` reports queued, evaluated, dispatched, and
  completed candidates plus rejection and negative-cache counts for each
  NVLink pair. Repeated unchanged `low_benefit` and `no_target_space`
  decisions are memoized until demand, access counts, or target capacity
  changes, preventing block-state updates from creating a planner retry loop.
- For eight-GPU runs, pass both `CUDA_VISIBLE_DEVICES=...` and `--world-size 8`. NVLink pairs use the logical GPU indices after CUDA remapping.

## KV Transfer Microbenchmark

Use `benchmark_kv_transfer.py` to validate the data-plane transfer primitive directly:

```bash
CUDA_VISIBLE_DEVICES=0,2 UV_CACHE_DIR=/tmp/uvcache uv run python benchmarks/benchmark_kv_transfer.py \
  --num-layers 28 \
  --block-size 256 \
  --num-kv-heads 8 \
  --head-dim 128 \
  --num-transfer-blocks 2 \
  --iterations 5 \
  --warmup 1
```

It reports mean / p95 transfer latency, transferred bytes per iteration, effective GiB/s, and data validation status. This benchmark isolates transfer from routing and model execution.
