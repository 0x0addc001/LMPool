#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
SOURCE="$ROOT/docs/papers/paper"
TARGET="$ROOT/docs/papers/paper_zh"

mkdir -p "$TARGET/figures"
cp "$SOURCE/example_paper.bib" "$TARGET/references.bib"

for figure in \
  fig_architecture.png \
  fig_hot_prefix_decision.png \
  fig_kv_block_lifecycle.png \
  fig_routing_cost_model.png \
  fig_routing_decision.png \
  fig_suite_routing.png \
  fig_suite_skew.png \
  fig_suite_transfer_profile.png \
  fig_transfer_cost_model.png \
  fig_transfer_decision.png; do
  cp "$SOURCE/figures/$figure" "$TARGET/figures/$figure"
done
