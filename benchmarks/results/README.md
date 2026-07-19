# Benchmark Results

This directory is for generated benchmark artifacts and is ignored by Git.
Existing historical files are retained, but new paper runs should use:

```text
benchmarks/results/paper/<UTC run id>/
  environment/
  qwen3-0.6b/
    kv_transfer/
    routing/
    memory_skew/
    session_handoff/
    load_skew/
  qwen3-1.7b/
    kv_transfer/
    routing/
    memory_skew/
    session_handoff/
    load_skew/
```

Each JSON uses schema version 2 with separate `metadata` and `results`; repeated
scenarios retain their raw `trial_results`, sample standard deviation, and 95%
confidence interval. Keep JSON, PNG figures, console logs, GPU topology, Git
revision, and working-tree status together. See
[`benchmarks/PAPER_RUNBOOK.md`](../PAPER_RUNBOOK.md).
