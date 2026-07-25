# LMPool Paper

This directory contains the paper source synchronized with the current implementation and the experiment batch in:

```text
benchmarks/results/paper/20260725T031840Z/
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
- `figures/fig_results_summary.png`: color summary of routing, transfer, and session-handoff results.
- `figures/fig_absolute_metrics.png`: absolute throughput and TTFT aggregates in original units.
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

Regenerate the color results figure directly from the archived paper JSON:

```bash
MPLCONFIGDIR=/tmp/matplotlib UV_CACHE_DIR=/tmp/uvcache \
  uv run python docs/reports/figures/generate_report_20260720.py \
  --output docs/paper/figures/fig_results_summary.png \
  --absolute-output docs/paper/figures/fig_absolute_metrics.png
```

Build the paper on a machine with a TeX distribution:

```bash
cd docs/paper
latexmk -pdf example_paper.tex
```

Alternatively, run `pdflatex`, `bibtex`, then `pdflatex` twice. Generated PDF files are not kept when they cannot be rebuilt from the current source; this prevents a stale PDF from disagreeing with the paper text.

## Evidence Policy

The main paper claims use five-trial means from both Qwen3-0.6B and Qwen3-1.7B. The routing workload supports a cached-token locality claim, while session handoff supports the end-to-end routing-plus-transfer claim. Load-skew and memory-skew results are retained as boundary results rather than omitted. The workload inputs are deterministic synthetic traces and are described as such; they are not presented as a production dataset.

The JSON metadata records a dirty worktree and a documentation-only revision transition during the long batch. The executable design used by the system experiments is consistent, so these artifacts are suitable for the current draft and internal comparison. A final archival submission should rerun the matrix from one clean, tagged revision.
