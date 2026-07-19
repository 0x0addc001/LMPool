#!/usr/bin/env bash
set -euo pipefail

# Run the complete paper experiment matrix for Qwen3-0.6B and Qwen3-1.7B.
# Both models must already exist locally; the suite deliberately runs offline.
# Override any default through environment variables, for example:
#   MODEL_17B=/models/Qwen3-1.7B REPETITIONS=3 bash benchmarks/run_paper_suite.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uvcache}"

GPU_SET="${GPU_SET:-0,1,3,4,5,6}"
WORLD_SIZE="${WORLD_SIZE:-6}"
NVLINK_PAIRS="${NVLINK_PAIRS:-0,1;2,3;4,5}"
TRANSFER_PAIRS="${TRANSFER_PAIRS:-0,1;3,4;5,6}"
REPETITIONS="${REPETITIONS:-5}"
SEED="${SEED:-0}"
DTYPE="${DTYPE:-auto}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-benchmarks/results/paper/${RUN_ID}}"
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

MODEL_06B="${MODEL_06B:-$(find_snapshot models--Qwen--Qwen3-0.6B)}"
MODEL_17B="${MODEL_17B:-$(find_snapshot models--Qwen--Qwen3-1.7B)}"

for model_var in MODEL_06B MODEL_17B; do
  model_path="${!model_var:-}"
  if [[ -z "${model_path}" || ! -f "${model_path}/config.json" ]]; then
    echo "${model_var} must point to a local Hugging Face snapshot containing config.json" >&2
    exit 2
  fi
  if ! compgen -G "${model_path}/*.safetensors" > /dev/null; then
    echo "${model_var} does not contain any .safetensors weight files: ${model_path}" >&2
    exit 2
  fi
done

CUDA_VISIBLE_DEVICES="${GPU_SET}" uv run python -c '
import sys, torch
expected = int(sys.argv[1])
actual = torch.cuda.device_count()
if actual != expected:
    raise SystemExit(f"WORLD_SIZE={expected}, but GPU_SET exposes {actual} CUDA devices")
print(f"validated {actual} visible CUDA devices")
' "${WORLD_SIZE}"

mkdir -p "${OUT}/environment"
nvidia-smi -L > "${OUT}/environment/gpus.txt"
nvidia-smi topo -m > "${OUT}/environment/topology.txt"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,power.limit \
  --format=csv > "${OUT}/environment/gpu_inventory.csv"
git rev-parse HEAD > "${OUT}/environment/git_revision.txt"
git status --short > "${OUT}/environment/git_status.txt"

run_model_suite() {
  local label="$1"
  local model="$2"
  local model_out="${OUT}/${label}"
  mkdir -p "${model_out}"/{kv_transfer,routing,memory_skew,session_handoff,load_skew}

  local pair
  IFS=';' read -r -a physical_pairs <<< "${TRANSFER_PAIRS}"
  for pair in "${physical_pairs[@]}"; do
    local pair_label="${pair/,/-}"
    CUDA_VISIBLE_DEVICES="${pair}" uv run python benchmarks/benchmark_kv_transfer.py \
      --model-name-or-path "${model}" \
      --dtype "${DTYPE}" \
      --block-size 256 \
      --block-counts 1,2,4,8 \
      --iterations 100 \
      --warmup 20 \
      --output-json "${model_out}/kv_transfer/pair_${pair_label}.json" \
      --output-figure "${model_out}/kv_transfer/pair_${pair_label}.png" \
      2>&1 | tee "${model_out}/kv_transfer/pair_${pair_label}.log"
  done

  local transfer_bandwidth
  transfer_bandwidth="$(uv run python -c '
import json, statistics, sys
values = []
for path in sys.argv[1:]:
    payload = json.load(open(path, encoding="utf-8"))
    values.extend(
        row["effective_bandwidth_gib_s"]
        for row in payload["results"]
        if row["num_transfer_blocks"] == 4
    )
if not values:
    raise SystemExit("missing 4-block transfer result")
print(f"{statistics.median(values):.6f}")
' "${model_out}"/kv_transfer/pair_*.json)"
  printf '%s\n' "${transfer_bandwidth}" > "${model_out}/kv_transfer/median_4_block_gib_s.txt"

  CUDA_VISIBLE_DEVICES="${GPU_SET}" uv run python benchmarks/benchmark_kv_routing.py \
    --model-name-or-path "${model}" \
    --dtype "${DTYPE}" \
    --world-size "${WORLD_SIZE}" \
    --num-prompts 192 \
    --prompt-repeat 16 \
    --max-tokens 64 \
    --temperature 0.6 \
    --ignore-eos \
    --seed "${SEED}" \
    --repetitions "${REPETITIONS}" \
    --locality-prefix-groups 16 \
    --nvlink-pairs "${NVLINK_PAIRS}" \
    --submit-window 16 \
    --kv-block-budget 64 \
    --gpu-memory-utilization 0.5 \
    --goodput-e2e-sla-ms 10000 \
    --output-json "${model_out}/routing/summary.json" \
    --output-figure "${model_out}/routing/summary.png" \
    2>&1 | tee "${model_out}/routing/run.log"

  CUDA_VISIBLE_DEVICES="${GPU_SET}" uv run python benchmarks/benchmark_e2e.py \
    --model-name-or-path "${model}" \
    --dtype "${DTYPE}" \
    --world-size "${WORLD_SIZE}" \
    --workload memory-skew \
    --memory-skew-prefix-groups 15 \
    --num-prompts 128 \
    --prompt-repeat 16 \
    --max-tokens 64 \
    --temperature 0.6 \
    --ignore-eos \
    --seed "${SEED}" \
    --repetitions "${REPETITIONS}" \
    --nvlink-pairs "${NVLINK_PAIRS}" \
    --submit-window 16 \
    --kv-block-budget 64 \
    --gpu-memory-utilization 0.5 \
    --goodput-e2e-sla-ms 10000 \
    --disable-background-copy \
    --foreground-transfer-min-benefit-ratio 1.1 \
    --foreground-transfer-bandwidth-gib-s "${transfer_bandwidth}" \
    --foreground-transfer-fixed-latency-ms 2.0 \
    --foreground-transfer-interference-multiplier 1.2 \
    --kv-transfer-prewarm-blocks 4 \
    --output-json "${model_out}/memory_skew/summary.json" \
    --output-figure "${model_out}/memory_skew/summary.png" \
    2>&1 | tee "${model_out}/memory_skew/run.log"

  CUDA_VISIBLE_DEVICES="${GPU_SET}" uv run python benchmarks/benchmark_e2e.py \
    --model-name-or-path "${model}" \
    --dtype "${DTYPE}" \
    --world-size "${WORLD_SIZE}" \
    --workload session-handoff \
    --handoff-prefix-groups 32 \
    --handoff-warmup-prompts 32 \
    --num-prompts 128 \
    --prompt-repeat 16 \
    --max-tokens 64 \
    --temperature 0.6 \
    --ignore-eos \
    --seed "${SEED}" \
    --repetitions "${REPETITIONS}" \
    --nvlink-pairs "${NVLINK_PAIRS}" \
    --submit-window 64 \
    --kv-block-budget 128 \
    --gpu-memory-utilization 0.5 \
    --goodput-e2e-sla-ms 10000 \
    --background-copy-max-blocks 8 \
    --background-copy-hot-threshold 1 \
    --background-copy-cooldown-s 0.1 \
    --background-copy-expected-reuses 4 \
    --foreground-transfer-min-benefit-ratio 1.1 \
    --foreground-transfer-bandwidth-gib-s "${transfer_bandwidth}" \
    --foreground-transfer-fixed-latency-ms 2.0 \
    --foreground-transfer-interference-multiplier 1.2 \
    --kv-transfer-prewarm-blocks 4 \
    --output-json "${model_out}/session_handoff/summary.json" \
    --output-figure "${model_out}/session_handoff/summary.png" \
    2>&1 | tee "${model_out}/session_handoff/run.log"

  CUDA_VISIBLE_DEVICES="${GPU_SET}" uv run python benchmarks/benchmark_e2e.py \
    --model-name-or-path "${model}" \
    --dtype "${DTYPE}" \
    --world-size "${WORLD_SIZE}" \
    --workload load-skew \
    --num-prompts 192 \
    --prompt-repeat 16 \
    --max-tokens 64 \
    --temperature 0.6 \
    --ignore-eos \
    --seed "${SEED}" \
    --repetitions "${REPETITIONS}" \
    --nvlink-pairs "${NVLINK_PAIRS}" \
    --submit-window 16 \
    --kv-block-budget 64 \
    --gpu-memory-utilization 0.5 \
    --goodput-e2e-sla-ms 10000 \
    --disable-background-copy \
    --foreground-transfer-bandwidth-gib-s "${transfer_bandwidth}" \
    --output-json "${model_out}/load_skew/summary.json" \
    --output-figure "${model_out}/load_skew/summary.png" \
    2>&1 | tee "${model_out}/load_skew/run.log"
}

run_model_suite "qwen3-0.6b" "${MODEL_06B}"
run_model_suite "qwen3-1.7b" "${MODEL_17B}"

echo "paper benchmark suite completed: ${OUT}"
