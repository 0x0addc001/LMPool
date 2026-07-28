#!/usr/bin/env bash
set -euo pipefail

# Standalone load-skew experiment. It uses a small hot prefix set, one
# warm-up per prefix, and a long reuse burst so background replicas have
# available NVLink targets and enough future demand to relieve owner queues.
# The trace never names a source or target rank.

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
LOAD_SKEW_PREFIX_GROUPS="${LOAD_SKEW_PREFIX_GROUPS:-3}"
LOAD_SKEW_WARMUP_PROMPTS="${LOAD_SKEW_WARMUP_PROMPTS:-3}"
LOAD_SKEW_HOT_GROUPS="${LOAD_SKEW_HOT_GROUPS:-3}"
LOAD_SKEW_HOT_SHARE="${LOAD_SKEW_HOT_SHARE:-1.0}"
NUM_PROMPTS="${NUM_PROMPTS:-192}"
PROMPT_REPEAT="${PROMPT_REPEAT:-64}"
KV_BLOCK_BUDGET="${KV_BLOCK_BUDGET:-192}"
SUBMIT_WINDOW="${SUBMIT_WINDOW:-96}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-benchmarks/results/light/load_skew/${RUN_ID}/qwen3-0.6b}"

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
    raise SystemExit(f"WORLD_SIZE={expected}, but GPU_SET exposes {actual} CUDA devices")
print(f"validated {actual} visible CUDA devices")
' "${WORLD_SIZE}"

mkdir -p "${OUT}"
printf '%s\n' "${MODEL}" > "${OUT}/model.txt"
printf '%s\n' "${TRANSFER_PROFILE}" > "${OUT}/transfer_profile.txt"
printf '%s\n' "${GPU_SET}" > "${OUT}/gpu_set.txt"
nvidia-smi topo -m > "${OUT}/topology.txt"
git rev-parse HEAD > "${OUT}/git_revision.txt"
git status --short > "${OUT}/git_status.txt"

JSON="${OUT}/summary.json"
FIGURE="${OUT}/summary.png"

echo "[load-skew] proactive replica-relief run"
echo "[load-skew] groups=${LOAD_SKEW_PREFIX_GROUPS} warmup=${LOAD_SKEW_WARMUP_PROMPTS} burst=$((NUM_PROMPTS - LOAD_SKEW_WARMUP_PROMPTS)) repeat=${PROMPT_REPEAT} budget=${KV_BLOCK_BUDGET}"

CUDA_VISIBLE_DEVICES="${GPU_SET}" uv run python benchmarks/benchmark_e2e.py \
  --model-name-or-path "${MODEL}" \
  --dtype "${DTYPE}" \
  --world-size "${WORLD_SIZE}" \
  --workload load-skew \
  --load-skew-prefix-groups "${LOAD_SKEW_PREFIX_GROUPS}" \
  --load-skew-warmup-prompts "${LOAD_SKEW_WARMUP_PROMPTS}" \
  --load-skew-hot-groups "${LOAD_SKEW_HOT_GROUPS}" \
  --load-skew-hot-share "${LOAD_SKEW_HOT_SHARE}" \
  --num-prompts "${NUM_PROMPTS}" \
  --prompt-repeat "${PROMPT_REPEAT}" \
  --max-tokens 8 \
  --temperature 0.6 \
  --ignore-eos \
  --seed "${SEED}" \
  --repetitions "${REPETITIONS}" \
  --nvlink-pairs "${NVLINK_PAIRS}" \
  --submit-window "${SUBMIT_WINDOW}" \
  --kv-block-budget "${KV_BLOCK_BUDGET}" \
  --gpu-memory-utilization 0.7 \
  --goodput-e2e-sla-ms "${GOODPUT_SLA_MS}" \
  --goodput-e2e-sla-sweep-ms "${GOODPUT_SLA_SWEEP_MS}" \
  --background-copy-max-blocks 32 \
  --background-copy-batch-max-blocks 32 \
  --background-copy-batch-max-candidates 1 \
  --background-copy-hot-threshold 1 \
  --background-copy-min-load-skew 0 \
  --background-copy-expected-reuses 64 \
  --background-copy-cooldown-s 0.1 \
  --foreground-transfer-min-benefit-ratio 1.1 \
  --foreground-transfer-profile-json "${TRANSFER_PROFILE}" \
  --foreground-transfer-fixed-latency-ms 0.0 \
  --foreground-transfer-interference-multiplier 1.2 \
  --kv-transfer-prewarm-blocks 4 \
  --output-json "${JSON}" \
  --output-figure "${FIGURE}" \
  2>&1 | tee "${OUT}/run.log"

echo
echo "Load-skew acceptance diagnostics"
jq -r '
  .results | to_entries[] | .key as $name | .value as $r
  | [
      $name,
      ("copies=" + ((($r.transfer_copy_count // 0) | tostring))),
      ("background_completed=" + (((($r.background_placement_stats.completed // 0)) | tostring))),
      ("lease_routes=" + ((($r.placement_lease_route_count // 0) | tostring))),
      ("reuse_tput=" + (((($r.phase_latency_stats.reuse.throughput_tok_s // 0) * 100) | round / 100) | tostring)),
      ("reuse_ttft_ms=" + (((($r.phase_latency_stats.reuse.mean_ttft_s // 0) * 1000 * 100) | round / 100) | tostring)),
      ("reuse_p90_e2e_ms=" + (((($r.phase_latency_stats.reuse.p90_e2e_s // 0) * 1000 * 100) | round / 100) | tostring))
    ] | join(" ")
' "${JSON}"

echo
echo "Require LMPool copies, completed placement, and lease routes before comparing reuse throughput and latency."
