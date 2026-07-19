# LMPool Paper

This directory contains the paper source synchronized with the current implementation and the experiment batch in:

```text
benchmarks/results/paper/20260719T072508Z/
```

## Files

- `example_paper.tex`: paper body, system design, implementation, evaluation, and limitations.
- `example_paper.bib`: verified bibliography used by the paper.
- `figures/fig_architecture.png`: architecture figure included by LaTeX.
- `figures/fig_results_summary.png`: color summary of routing, transfer, and session-handoff results.
- `figures/fig_absolute_metrics.png`: absolute throughput and TTFT aggregates in original units.
- `figures/generate_architecture.py`: reproducible source for the light paper figure and dark root README figure.
- `mlsys2025.sty` / `mlsys2025.bst`: retained paper template files.

## Build

Regenerate the architecture figure after changing its source:

```bash
MPLCONFIGDIR=/tmp/matplotlib UV_CACHE_DIR=/tmp/uvcache \
  uv run python docs/paper/figures/generate_architecture.py
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

The main paper claims use five-trial means from both Qwen3-0.6B and Qwen3-1.7B. Routing and session-handoff results are reported as positive evidence. Load-skew and memory-skew results are retained as boundary results rather than omitted. The workload inputs are deterministic synthetic traces and are described as such; they are not presented as a production dataset.

The JSON metadata records a dirty worktree and a documentation-only revision transition during the long batch. The executable design used by the system experiments is consistent, so these artifacts are suitable for the current draft and internal comparison. A final archival submission should rerun the matrix from one clean, tagged revision.
