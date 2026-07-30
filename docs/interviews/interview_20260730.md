# LMPool CV & QA

## Resume Bullets

### One-Line Version

**LMPool | Multi-GPU LLM KV-Cache Scheduling System**: Built KV-locality-aware routing and transactional NVLink KV copy/move; on six RTX 3090 GPUs with Qwen3-0.6B/1.7B, 5x shared-prefix routing improved throughput by 11.8%/21.3% and reduced mean TTFT by 18.6%/25.2%.

### Three-Bullet Version

- Built a multi-GPU KV-cache coordination prototype in Python, PyTorch, and NCCL with separate control and data planes; used versioning, reservations, prepare--publish visibility, and rollback so incomplete replicas never entered routing decisions.
- Implemented locality- and load-aware KV routing over contiguous prefix reuse, token-equivalent queue work, capacity, and reclamation pressure. In five-trial internal ablations on six RTX 3090 GPUs, 5x shared prefixes improved throughput by 11.8%/21.3%, reduced mean TTFT by 18.6%/25.2%, and reduced uncached prefill tokens by 59.9%/61.0% for Qwen3-0.6B/1.7B.
- Implemented pair- and payload-aware NVLink KV transfer with packed all-layer K/V tensors, NCCL P2P copy/move, P95 profiles, and online residual EWMAs; validated 1--64-block transfers, 87 copied blocks and 95 lease routes after background placement, and safe release of 35 source blocks under 0.6B memory skew.

## STAR Summary

### Situation

Data-parallel LLM replicas maintain independent KV caches. Round-robin dispatch can send a request away from an already cached shared prefix and repeat prefill work. Strict prefix-owner routing improves locality but can overload one GPU. Unconditional KV movement is also unsafe and can cost more than recomputation.

### Task

Build a multi-GPU KV-cache prototype that routes requests to useful local prefixes when load permits, moves complete KV block chains only when the expected benefit covers the cost, and preserves consistency across independent worker processes.

### Action

I separated the system into a main process, an independent control plane, and one data plane per GPU. The Global Scheduler scores contiguous ready-prefix reuse, token-equivalent queue work, decode work, effective capacity, and reclamation pressure instead of maximizing cache hits alone. I implemented a four-phase transfer transaction: source validation and target reservation, packed NCCL P2P execution, destination publication, and copy/move finalization. A destination is invisible to routing until it publishes a complete hash chain and ready state. Transfer admission uses pair-specific P95 payload profiles plus conservative online source and placement residual EWMAs. I also created controlled long-prefix routing, load-skew, and memory-skew workloads and added unit and multi-process tests around block management, scheduling, the control plane, and KV transfer.

### Result

In five-trial internal ablations on six RTX 3090 GPUs, 5x shared-prefix routing improved throughput by 11.8%/21.3% and reduced mean TTFT by 18.6%/25.2% for Qwen3-0.6B/1.7B, while reducing uncached prefill by about 60%. The transfer path was byte-validated; load skew completed three background placements, copied 87 blocks, and routed 95 later requests through placement leases. Under 0.6B memory skew, move-style transfer safely released 35 source blocks. I retained the negative result that completed transfer does not automatically improve mean E2E latency or throughput.

## Interview Scripts

### 90-Second Project Overview

I built LMPool, a multi-GPU KV-cache scheduling system for LLM serving. In data-parallel deployment, each GPU owns an independent KV cache. Round-robin is balanced, but it may miss a shared prefix already cached on another GPU and recompute the entire prefix. Sending every request to the prefix owner has the opposite problem: it creates a hotspot.

I split the problem into locality and fluidity. For locality, the router considers contiguous reusable prefixes, queued tokens, decode work, capacity, and reclamation pressure. It sends a request to the GPU with the lowest projected completion cost rather than maximizing cache hits alone. For fluidity, the system can copy or move complete KV block chains over direct NVLink when KV placement no longer matches demand or capacity. A pair-specific, payload-aware cost model only admits a transfer when predicted saved prefill work covers its cost.

The implementation separates a control plane from per-GPU data planes. Physical blocks remain locally owned. Transfer is transactional: prepare reserves the target, execute sends one packed all-layer K/V tensor over NCCL, publish makes the new replica visible, and finalize either retains or releases the source. This prevents partially received state from being routed.

On six RTX 3090 GPUs, 5x shared-prefix routing improved throughput by 11.8% for Qwen3-0.6B and 21.3% for Qwen3-1.7B, with mean TTFT reductions of 18.6% and 25.2%. I also validated background placement, placement leases, and source release. Importantly, I do not claim that transfer always improves throughput or E2E latency, because the experiments show that decode contention can offset saved prefill time.

### Three-Minute STAR Walkthrough

The initial problem was that prefix caching works well inside one inference worker, but it becomes a placement problem across data-parallel GPUs. A request can have a reusable prefix on one GPU and still be dispatched to another GPU by round robin. If I routed every request to the prefix owner, I would trade recomputation for queueing and create a load hotspot. My task was to build a system that jointly considered request placement and KV placement without compromising worker-local memory ownership.

I first defined the ownership model. The Global Block Manager owns only versioned metadata, while each Local Block Manager exclusively owns physical allocation, reference counts, readiness, and KV tensors. Then I implemented the routing cost. It combines token-equivalent queued work, block-rounded missing prefill work, and reclamation pressure. A longer contiguous prefix directly reduces the missing-prefill term, so locality becomes saved computation instead of a separate cache-hit reward. I also allowed a bounded spill from an overloaded owner only when the extra recomputation stayed below a configured limit.

The hardest part was transfer correctness and admission. Copying a tensor is straightforward; ensuring that a received destination is safe to route to is not. I implemented prepare, execute, publish, and finalize phases. Prepare locks the source chain and reserves concrete target blocks. Execute sends one packed tensor containing all layers and both K and V values through a pair-local NCCL communicator. Publish records the complete hash chain and ready state. Only then can the control plane route to the destination. Finalize keeps the source for a copy or releases an unreferenced dependency-safe suffix for a move. Any failure releases the target reservation and retains the source. I used pair-specific 1--64-block P95 profiles and conservative online residual EWMAs to decide whether expected saved prefill time justified a plan.

The stable result was routing. With 5x shared prefixes, throughput improved by 11.8% and 21.3% on the two Qwen models, and mean TTFT fell by 18.6% and 25.2%. Transfer mechanisms also executed as designed: load skew copied 87 blocks and routed 95 later requests through leases, while 0.6B memory skew safely released 35 source blocks. However, the full system was not always faster in mean E2E because decode contention could offset the prefill savings. I reported that limitation directly. The main lesson was to separate a mechanism working correctly from a mechanism producing a general end-to-end gain, and to align the claim with the evidence.

## Common Follow-Ups

### Why not use routing alone?

Routing can only use KV state already present at a candidate GPU. When demand or capacity no longer matches placement, routing may concentrate work on the owner or be unable to release source capacity. Transfer provides controlled cache fluidity, but it does not replace routing.

### Why not always transfer?

Transfer has gather, NCCL, scatter, reservation, synchronization, and memory-interference costs. If future reuse is insufficient, recomputation is cheaper. LMPool therefore uses pair-specific P95 profiles and online residuals, and accepts a plan only when expected saved prefill work covers a conservative transfer estimate.

### How did you handle concurrency and correctness?

Each Local Block Manager is the sole mutator of its physical blocks. The control-plane event loop serializes global metadata mutations. Transactions use generation checks, reservations, a prepare--publish visibility boundary, idempotent phases, and abort rollback. A destination is never routable before publication.

### What was the hardest part?

The hardest part was not moving a tensor. It was deciding when movement was worthwhile and proving that it did not corrupt serving state. I separated data-path calibration, transfer admission, and publication semantics, then tested routing, background placement, and source release with separate controlled workloads.

### What would you do next?

I would evaluate online demand prediction on public or anonymized multi-turn traces, report precision, recall, lead time, and invalid-copy rate, compare against at least one production serving engine under the same model, KV budget, and arrival process, and ablate block size, KV geometry, GPU count, and the cost-model profile, EWMA, and safety ratio.
