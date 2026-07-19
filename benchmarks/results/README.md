# Benchmark Results

This directory is for generated benchmark artifacts and is ignored by Git.
Existing historical files are retained, but new paper runs should use:

```text
benchmarks/results/paper/<UTC run id>/
  environment/
  kv_transfer/
  routing/
  memory_skew/
  session_handoff/
  load_skew/
```

Each run should keep the JSON, PNG figures, console log, GPU topology, Git
revision, and working-tree status together. See
[`benchmarks/PAPER_RUNBOOK.md`](../PAPER_RUNBOOK.md).
