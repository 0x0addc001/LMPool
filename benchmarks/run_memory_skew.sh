#!/usr/bin/env bash
set -euo pipefail

# Standalone proactive capacity-offload experiment.
# This script intentionally does not run the complete preflight or paper suite.
# It is useful for diagnosing whether memory-skew creates executable foreground
# transfer plans before committing a new workload to the formal suite. The
# trace never selects source or target worker ranks; every scenario dispatches
# through its normal round-robin or control-plane policy.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uvcache}"

GPU_SET="${GPU_SET:-0,1,3,4,5,6}"
WORLD_SIZE="${WORLD_SIZE:-6}"
NVLINK_PAIRS="${NVLINK_PAIRS:-0,1;2,3;4,5}"
MODEL="${MODEL:-${MODEL_06B:-}}"
TRANSFER_PROFILE="${TRANSFER_PROFILE:-}"
REPETITIONS="${REPETITIONS:-1}"
SEED="${SEED:-0}"
DTYPE="${DTYPE:-auto}"
GOODPUT_SLA_MS="${GOODPUT_SLA_MS:-3000}"
GOODPUT_SLA_SWEEP_MS="${GOODPUT_SLA_SWEEP_MS:-2000,3000,5000,10000}"
MEMORY_SKEW_PREFIX_GROUPS="${MEMORY_SKEW_PREFIX_GROUPS:-6}"
MEMORY_SKEW_WARMUP_PROMPTS="${MEMORY_SKEW_WARMUP_PROMPTS:-6}"
MEMORY_SKEW_PRESSURE_PROMPTS="${MEMORY_SKEW_PRESSURE_PROMPTS:-30}"
MEMORY_SKEW_TRIGGER_PROMPTS="${MEMORY_SKEW_TRIGGER_PROMPTS:-0}"
MEMORY_SKEW_PRESSURE_HOT_GROUPS="${MEMORY_SKEW_PRESSURE_HOT_GROUPS:-2}"
MEMORY_SKEW_PRESSURE_HOT_SHARE="${MEMORY_SKEW_PRESSURE_HOT_SHARE:-0.8}"
MEMORY_SKEW_ANCHOR_SHARE="${MEMORY_SKEW_ANCHOR_SHARE:-0.375}"
MEMORY_SKEW_REUSE_HOT_GROUPS="${MEMORY_SKEW_REUSE_HOT_GROUPS:-0}"
MEMORY_SKEW_REUSE_HOT_SHARE="${MEMORY_SKEW_REUSE_HOT_SHARE:-1.0}"
ROUTE_LOAD_BYPASS_THRESHOLD="${ROUTE_LOAD_BYPASS_THRESHOLD:-256}"
NUM_PROMPTS="${NUM_PROMPTS:-72}"
PROMPT_REPEAT="${PROMPT_REPEAT:-64}"
KV_BLOCK_BUDGET="${KV_BLOCK_BUDGET:-128}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-benchmarks/results/light/memory_skew/${RUN_ID}/qwen3-0.6b}"

HF_HUB="${HF_HOME:-${HOME}/.cache/huggingface}/hub"

find_snapshot() {
  local repository="$1"
  local config
  config="$(find "${HF_HUB}/${repository}/snapshots" -mindepth 2 -maxdepth 2 \
    -name config.json -print -quit 2>/dev/null || true)"
  if [[ -n "${config}" ]]; then
    dirname "${config}"
  fi
}

find_latest_profile() {
  find benchmarks/results/paper -path \
    '*/qwen3-0.6b/kv_transfer/latency_profile.json' -type f -print \
    2>/dev/null | sort | tail -n 1
}

MODEL="${MODEL:-$(find_snapshot models--Qwen--Qwen3-0.6B)}"
TRANSFER_PROFILE="${TRANSFER_PROFILE:-$(find_latest_profile)}"

if [[ -z "${MODEL}" || ! -f "${MODEL}/config.json" ]]; then
  echo "MODEL or MODEL_06B must identify a local Qwen3-0.6B snapshot" >&2
  exit 2
fi
if ! compgen -G "${MODEL}/*.safetensors" > /dev/null; then
  echo "model snapshot contains no safetensors weights: ${MODEL}" >&2
  exit 2
fi
if [[ -z "${TRANSFER_PROFILE}" || ! -f "${TRANSFER_PROFILE}" ]]; then
  echo "TRANSFER_PROFILE must identify a compatible latency_profile.json" >&2
  exit 2
fi
if ! command -v jq >/dev/null; then
  echo "jq is required for the post-run summary" >&2
  exit 2
fi

CUDA_VISIBLE_DEVICES="${GPU_SET}" uv run python -c '
import sys
import torch

expected = int(sys.argv[1])
actual = torch.cuda.device_count()
if actual != expected:
    raise SystemExit(
        f"WORLD_SIZE={expected}, but GPU_SET exposes {actual} CUDA devices"
    )
print(f"validated {actual} visible CUDA devices")
' "${WORLD_SIZE}"

mkdir -p "${OUT}"
printf '%s\n' "${MODEL}" > "${OUT}/model.txt"
printf '%s\n' "${TRANSFER_PROFILE}" > "${OUT}/transfer_profile.txt"
printf '%s\n' "${GPU_SET}" > "${OUT}/gpu_set.txt"
nvidia-smi -L > "${OUT}/gpus.txt"
nvidia-smi topo -m > "${OUT}/topology.txt"
git rev-parse HEAD > "${OUT}/git_revision.txt"
git status --short > "${OUT}/git_status.txt"

JSON="${OUT}/summary.json"
FIGURE="${OUT}/summary.png"

echo "[memory-skew] proactive capacity-offload run"
echo "[memory-skew] output: ${OUT}"
echo "[memory-skew] groups=${MEMORY_SKEW_PREFIX_GROUPS} warmup=${MEMORY_SKEW_WARMUP_PROMPTS} pressure=${MEMORY_SKEW_PRESSURE_PROMPTS} pressure_hot_groups=${MEMORY_SKEW_PRESSURE_HOT_GROUPS} reuse_hot_groups=${MEMORY_SKEW_REUSE_HOT_GROUPS} anchor_share=${MEMORY_SKEW_ANCHOR_SHARE} trigger=${MEMORY_SKEW_TRIGGER_PROMPTS} total=${NUM_PROMPTS} prompt_repeat=${PROMPT_REPEAT} kv_budget=${KV_BLOCK_BUDGET}"

CUDA_VISIBLE_DEVICES="${GPU_SET}" uv run python benchmarks/benchmark_e2e.py \
  --model-name-or-path "${MODEL}" \
  --dtype "${DTYPE}" \
  --world-size "${WORLD_SIZE}" \
  --workload memory-skew \
  --memory-skew-prefix-groups "${MEMORY_SKEW_PREFIX_GROUPS}" \
  --memory-skew-warmup-prompts "${MEMORY_SKEW_WARMUP_PROMPTS}" \
  --memory-skew-pressure-prompts "${MEMORY_SKEW_PRESSURE_PROMPTS}" \
  --memory-skew-pressure-hot-groups "${MEMORY_SKEW_PRESSURE_HOT_GROUPS}" \
  --memory-skew-pressure-hot-share "${MEMORY_SKEW_PRESSURE_HOT_SHARE}" \
  --memory-skew-anchor-share "${MEMORY_SKEW_ANCHOR_SHARE}" \
  --memory-skew-reuse-hot-groups "${MEMORY_SKEW_REUSE_HOT_GROUPS}" \
  --memory-skew-reuse-hot-share "${MEMORY_SKEW_REUSE_HOT_SHARE}" \
  --memory-skew-trigger-prompts "${MEMORY_SKEW_TRIGGER_PROMPTS}" \
  --num-prompts "${NUM_PROMPTS}" \
  --prompt-repeat "${PROMPT_REPEAT}" \
  --max-tokens 16 \
  --temperature 0.6 \
  --ignore-eos \
  --seed "${SEED}" \
  --repetitions "${REPETITIONS}" \
  --nvlink-pairs "${NVLINK_PAIRS}" \
  --submit-window 48 \
  --kv-block-budget "${KV_BLOCK_BUDGET}" \
  --gpu-memory-utilization 0.5 \
  --goodput-e2e-sla-ms "${GOODPUT_SLA_MS}" \
  --goodput-e2e-sla-sweep-ms "${GOODPUT_SLA_SWEEP_MS}" \
  --memory-skew-proactive-move \
  --background-transfer-mode move \
  --background-move-source-free-block-threshold 8 \
  --background-copy-max-blocks 64 \
  --background-copy-batch-max-blocks 64 \
  --background-copy-batch-max-candidates 1 \
  --background-copy-hot-threshold 1 \
  --foreground-transfer-min-benefit-ratio 1.1 \
  --foreground-transfer-profile-json "${TRANSFER_PROFILE}" \
  --foreground-transfer-fixed-latency-ms 0.0 \
  --foreground-transfer-interference-multiplier 1.2 \
  --route-load-bypass-threshold "${ROUTE_LOAD_BYPASS_THRESHOLD}" \
  --kv-transfer-prewarm-blocks 4 \
  --output-json "${JSON}" \
  --output-figure "${FIGURE}" \
  2>&1 | tee "${OUT}/run.log"

echo
echo "Memory-skew acceptance diagnostics"
jq -r '
  .results
  | to_entries[]
  | .key as $name
  | .value as $r
  | [
      $name,
      ("background_completed=" + (((($r.background_placement_stats.completed // 0)) | tostring))),
      ("released=" + ((($r.transfer_release_count // 0) | tostring))),
      ("verified=" + ((($r.offload_verified // false) | tostring))),
      ("reuse_tput=" + (((($r.phase_latency_stats.reuse.throughput_tok_s // 0) * 100) | round / 100) | tostring)),
      ("reuse_p90_e2e_ms=" + (((($r.phase_latency_stats.reuse.p90_e2e_s // 0) * 1000 * 100) | round / 100) | tostring))
    ] | join(" ")
' "${JSON}"

echo
echo "The workload performs one proactive move after pressure drains; foreground rebalance is disabled."
echo "Use benchmarks/run_preflight_suite.sh for the combined routing, memory-skew, and load-skew gate."
