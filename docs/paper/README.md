# LMPool Paper

This directory contains the paper source synchronized with the current implementation and the experiment batch in:

```text
benchmarks/results/paper/20260727T231622Z/
```

## Files

- `example_paper.tex`: paper body, system design, implementation, evaluation, and limitations.
- `example_paper.bib`: verified bibliography used by the paper.
- `figures/fig_architecture.png` / `.pdf`: white-background architecture figure included by LaTeX and available as vector output.
- `figures/fig_routing_decision.png` / `.pdf`: KV-aware routing branches and outcomes.
- `figures/fig_hot_prefix_decision.png` / `.pdf`: background-placement prediction, capacity, and admission gates.
- `figures/fig_transfer_decision.png` / `.pdf`: foreground/background transfer admission, transaction, and rollback path.
- `figures/fig_kv_block_lifecycle.png` / `.pdf`: local and transactional cross-GPU KV-block lifecycle.
- `figures/fig_routing_cost_model.png` / `.pdf`: token-equivalent routing cost and bounded-spill model.
- `figures/fig_transfer_cost_model.png` / `.pdf`: measured transfer cost, saved-work, and admission model.
- `figures/fig_suite_routing.png` / `.pdf`: five-trial routing results across 1x, 3x, and 5x shared-prefix lengths.
- `figures/fig_suite_skew.png` / `.pdf`: load-skew and memory-skew mechanism and serving results.
- `figures/fig_suite_transfer_profile.png` / `.pdf`: pair-aggregated NVLink P95 transfer profiles.
- `figures/generate_suite_results.py`: reproducible aggregation and visualization source for the current paper suite.
- `figures/generate_architecture.py`: reproducible source for the light paper/report architecture and dark README variant.
- `figures/generate_decision_flow.py`: synchronized light paper/report and dark README decision-flow figures.
- `figures/generate_block_lifecycle.py`: synchronized light paper and dark README lifecycle figures.
- `figures/generate_cost_models.py`: synchronized light paper and dark README routing/transfer cost-model figures.
- `mlsys2025.sty` / `mlsys2025.bst`: retained paper template files.

## Build

Regenerate the architecture figure after changing its source:

```bash
MPLCONFIGDIR=/tmp/matplotlib UV_CACHE_DIR=/tmp/uvcache \
  uv run python docs/paper/figures/generate_architecture.py
```

Regenerate the synchronized decision-flow figures:

```bash
MPLCONFIGDIR=/tmp/matplotlib UV_CACHE_DIR=/tmp/uvcache \
  uv run python docs/paper/figures/generate_decision_flow.py
```

Regenerate the synchronized KV-block lifecycle figures:

```bash
MPLCONFIGDIR=/tmp/matplotlib UV_CACHE_DIR=/tmp/uvcache \
  uv run python docs/paper/figures/generate_block_lifecycle.py
```

Regenerate the separate synchronized cost-model figures:

```bash
MPLCONFIGDIR=/tmp/matplotlib UV_CACHE_DIR=/tmp/uvcache \
  uv run python docs/paper/figures/generate_cost_models.py
```

Regenerate the current suite figures directly from the archived paper JSON:

```bash
MPLCONFIGDIR=/tmp/matplotlib UV_CACHE_DIR=/tmp/uvcache \
  uv run python docs/paper/figures/generate_suite_results.py \
  --suite benchmarks/results/paper/20260727T231622Z
```

Build the paper on a machine with a TeX distribution:

```bash
cd docs/paper
latexmk -pdf example_paper.tex
```

Alternatively, run `pdflatex`, `bibtex`, then `pdflatex` twice. Generated PDF files are not kept when they cannot be rebuilt from the current source; this prevents a stale PDF from disagreeing with the paper text.

## Evidence Policy

The main paper claims use five-trial means from both Qwen3-0.6B and Qwen3-1.7B. The routing workload supports the primary end-to-end locality claim: at 5x shared-prefix length, it reduces uncached prefill by about 60\% and improves throughput and tail latency on both models. The load-skew and memory-skew workloads verify background placement, placement leases, and safe foreground source release, while also showing the boundary of the current scheduler: transfer does not uniformly improve mean E2E latency or throughput. The workload inputs are deterministic synthetic traces and are described as such; they are not presented as a production dataset.

The JSON metadata records a dirty worktree and a documentation-only revision transition during the long batch. The executable design used by the system experiments is consistent, so these artifacts are suitable for the current draft and internal comparison. A final archival submission should rerun the matrix from one clean, tagged revision.
