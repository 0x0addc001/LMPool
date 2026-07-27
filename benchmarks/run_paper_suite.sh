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
TRANSFER_FIXED_LATENCY_MS="${TRANSFER_FIXED_LATENCY_MS:-0.0}"
TRANSFER_INTERFERENCE_MULTIPLIER="${TRANSFER_INTERFERENCE_MULTIPLIER:-1.2}"
COST_CALIBRATION_REPETITIONS="${COST_CALIBRATION_REPETITIONS:-2}"
COST_CALIBRATION_BATCH_BLOCKS="${COST_CALIBRATION_BATCH_BLOCKS:-1 2 4 8 16 32 64}"
# The default sweep reaches 5x. Add 10 explicitly for an optional extreme
# context-length sensitivity run: ROUTING_PREFIX_MULTIPLIERS="1 3 5 10".
ROUTING_PREFIX_MULTIPLIERS="${ROUTING_PREFIX_MULTIPLIERS:-1 3 5}"
GOODPUT_SLA_MS_06B="${GOODPUT_SLA_MS_06B:-3000}"
GOODPUT_SLA_MS_17B="${GOODPUT_SLA_MS_17B:-5000}"
GOODPUT_SLA_SWEEP_MS="${GOODPUT_SLA_SWEEP_MS:-2000,3000,5000,10000}"
REPETITIONS="${REPETITIONS:-5}"
SEED="${SEED:-0}"
DTYPE="${DTYPE:-auto}"
RESUME="${RESUME:-0}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-benchmarks/results/paper/${RUN_ID}}"
HF_HUB="${HF_HOME:-${HOME}/.cache/huggingface}/hub"

artifact_complete() {
  local path="$1"
  local expected_results="$2"
  local expected_type="$3"
  local expected_model="$4"
  local expected_workload="${5:-}"
  local expected_interference_multiplier="${6:-}"
  local expected_fixed_latency_ms="${7:-}"
  local expected_repetitions="${8:-${REPETITIONS}}"
  [[ "${RESUME}" == "1" ]] \
    && [[ -s "${path}" ]] \
    && jq -e \
      --argjson expected_results "${expected_results}" \
      --arg expected_type "${expected_type}" \
      --arg expected_model "${expected_model}" \
      --arg expected_workload "${expected_workload}" \
      --arg expected_interference_multiplier "${expected_interference_multiplier}" \
      --arg expected_fixed_latency_ms "${expected_fixed_latency_ms}" \
      --argjson expected_repetitions "${expected_repetitions}" \
      '.metadata != null
       and (.results | type) == $expected_type
       and (.results | length == $expected_results)
       and .metadata.model.name_or_path == $expected_model
       and (
         if $expected_type == "object"
         then
           .metadata.arguments.repetitions == $expected_repetitions
           and all(.results[];
             . == null
             or (
               has("mean_tpot_s")
               and has("goodput_sla_sweep_tok_s")
             )
           )
         else true
         end
       )
       and (
         if $expected_workload != ""
         then .metadata.arguments.workload == $expected_workload
         else true
         end
       )
       and (
         if $expected_interference_multiplier != ""
         then
           .metadata.arguments.foreground_transfer_interference_multiplier
           == ($expected_interference_multiplier | tonumber)
         else true
         end
       )
       and (
         if $expected_fixed_latency_ms != ""
         then
           .metadata.arguments.foreground_transfer_fixed_latency_ms
           == ($expected_fixed_latency_ms | tonumber)
         else true
         end
       )' \
      "${path}" >/dev/null
}

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
  local goodput_sla_ms="$3"
  local model_out="${OUT}/${label}"
  mkdir -p "${model_out}"/{kv_transfer,routing,memory_skew,load_skew}
  rm -f "${model_out}/SUITE_COMPLETE"
  echo "[suite] starting ${label}: ${model}"

  local pair
  local -a transfer_profile_inputs=()
  IFS=';' read -r -a physical_pairs <<< "${TRANSFER_PAIRS}"
  for pair in "${physical_pairs[@]}"; do
    local pair_label="${pair/,/-}"
    local pair_json="${model_out}/kv_transfer/pair_${pair_label}.json"
    if artifact_complete "${pair_json}" 7 array "${model}"; then
      echo "[resume] reuse ${pair_json}"
    else
      CUDA_VISIBLE_DEVICES="${pair}" uv run python benchmarks/benchmark_kv_transfer.py \
        --model-name-or-path "${model}" \
        --dtype "${DTYPE}" \
        --block-size 256 \
        --block-counts 1,2,4,8,16,32,64 \
        --iterations 100 \
        --warmup 20 \
        --output-json "${pair_json}" \
        --output-figure "${model_out}/kv_transfer/pair_${pair_label}.png" \
        2>&1 | tee "${model_out}/kv_transfer/pair_${pair_label}.log"
    fi
    transfer_profile_inputs+=("${pair_json}")
  done

  local data_path_profile="${model_out}/kv_transfer/data_path_profile.json"
  uv run python benchmarks/build_transfer_profile.py \
    --inputs "${transfer_profile_inputs[@]}" \
    --logical-pairs "${NVLINK_PAIRS}" \
    --latency-metric p95_latency_ms \
    --output "${data_path_profile}"

  # Calibrate complete dispatch-to-publish transactions at power-of-two batch
  # limits. A large transaction cannot safely calibrate the 1--2-block
  # background and foreground moves produced by light pressure.
  local calibration_blocks
  local calibration_candidate_blocks
  local -a transaction_calibrations=()
  for calibration_blocks in ${COST_CALIBRATION_BATCH_BLOCKS}; do
    calibration_candidate_blocks=4
    if (( calibration_blocks < calibration_candidate_blocks )); then
      calibration_candidate_blocks="${calibration_blocks}"
    fi
    local transaction_calibration="${model_out}/kv_transfer/transaction_calibration_${calibration_blocks}.json"
    local calibration_complete=0
    if artifact_complete \
        "${transaction_calibration}" 5 object "${model}" transfer-calibration \
        "${TRANSFER_INTERFERENCE_MULTIPLIER}" 0.0 "${COST_CALIBRATION_REPETITIONS}" \
        && jq -e \
          --argjson expected_limit "${calibration_blocks}" \
          --argjson expected_candidate_limit "${calibration_candidate_blocks}" \
          '.metadata.arguments.background_copy_batch_max_blocks == $expected_limit
           and .metadata.arguments.background_copy_max_blocks == $expected_candidate_limit
           and (.results["multi-gpu-lmpool"].transfer_placement_observations | length > 0)' \
          "${transaction_calibration}" >/dev/null; then
      calibration_complete=1
    fi
    if [[ "${calibration_complete}" == "1" ]]; then
      echo "[resume] reuse ${transaction_calibration}"
    else
      CUDA_VISIBLE_DEVICES="${GPU_SET}" uv run python benchmarks/benchmark_e2e.py \
        --model-name-or-path "${model}" \
        --dtype "${DTYPE}" \
        --world-size "${WORLD_SIZE}" \
        --workload transfer-calibration \
        --scenarios multi-gpu-lmpool \
        --calibration-prefix-groups 66 \
        --calibration-warmup-prompts 66 \
        --num-prompts 132 \
        --prompt-repeat 8 \
        --max-tokens 8 \
        --temperature 0.6 \
        --ignore-eos \
        --seed "$((SEED + 1000 + calibration_blocks))" \
        --repetitions "${COST_CALIBRATION_REPETITIONS}" \
        --nvlink-pairs "${NVLINK_PAIRS}" \
        --submit-window 66 \
        --kv-block-budget 128 \
        --gpu-memory-utilization 0.5 \
        --goodput-e2e-sla-ms 10000 \
        --background-copy-max-blocks "${calibration_candidate_blocks}" \
        --background-copy-batch-max-blocks "${calibration_blocks}" \
        --background-copy-batch-max-candidates 32 \
        --background-copy-hot-threshold 1 \
        --background-copy-cooldown-s 0.1 \
        --background-copy-expected-reuses 4 \
        --foreground-transfer-min-benefit-ratio 0.0 \
        --foreground-transfer-profile-json "${data_path_profile}" \
        --foreground-transfer-fixed-latency-ms 0.0 \
        --foreground-transfer-interference-multiplier "${TRANSFER_INTERFERENCE_MULTIPLIER}" \
        --kv-transfer-prewarm-blocks 4 \
        --output-json "${transaction_calibration}" \
        --output-figure "${model_out}/kv_transfer/transaction_calibration_${calibration_blocks}.png" \
        2>&1 | tee "${model_out}/kv_transfer/transaction_calibration_${calibration_blocks}.log"
    fi
    transaction_calibrations+=("${transaction_calibration}")
  done

  local transfer_profile="${model_out}/kv_transfer/latency_profile.json"
  uv run python benchmarks/build_transfer_profile.py \
    --inputs "${transfer_profile_inputs[@]}" \
    --logical-pairs "${NVLINK_PAIRS}" \
    --latency-metric p95_latency_ms \
    --transaction-inputs "${transaction_calibrations[@]}" \
    --transaction-scenario multi-gpu-lmpool \
    --transaction-percentile 0.95 \
    --output "${transfer_profile}"

  local routing_multiplier
  for routing_multiplier in ${ROUTING_PREFIX_MULTIPLIERS}; do
    local routing_repeat=$((16 * routing_multiplier))
    local routing_max_model_length=$((2048 * routing_multiplier))
    local routing_max_batch_tokens="${routing_max_model_length}"
    if (( routing_max_batch_tokens < 4096 )); then
      routing_max_batch_tokens=4096
    fi
    local routing_stem="prefix_${routing_multiplier}x"
    local routing_json="${model_out}/routing/${routing_stem}.json"
    if artifact_complete "${routing_json}" 3 object "${model}" \
        && jq -e \
          --argjson expected_repeat "${routing_repeat}" \
          --argjson expected_max_tokens 8 \
          --argjson expected_max_length "${routing_max_model_length}" \
          --argjson expected_sla "${goodput_sla_ms}" \
          --arg expected_sweep "${GOODPUT_SLA_SWEEP_MS}" \
          '.metadata.arguments.prompt_repeat == $expected_repeat
           and .metadata.arguments.max_tokens == $expected_max_tokens
           and .metadata.arguments.max_model_length == $expected_max_length
           and .metadata.arguments.goodput_e2e_sla_ms
               == $expected_sla
           and .metadata.arguments.goodput_e2e_sla_sweep_ms
               == $expected_sweep' \
          "${routing_json}" >/dev/null; then
      echo "[resume] reuse ${routing_json}"
    else
      CUDA_VISIBLE_DEVICES="${GPU_SET}" uv run python benchmarks/benchmark_kv_routing.py \
        --model-name-or-path "${model}" \
        --dtype "${DTYPE}" \
        --world-size "${WORLD_SIZE}" \
        --num-prompts 192 \
        --prompt-repeat "${routing_repeat}" \
        --max-tokens 8 \
        --max-model-length "${routing_max_model_length}" \
        --max-num-batched-tokens "${routing_max_batch_tokens}" \
        --temperature 0.6 \
        --ignore-eos \
        --seed "${SEED}" \
        --repetitions "${REPETITIONS}" \
        --locality-prefix-groups 16 \
        --nvlink-pairs "${NVLINK_PAIRS}" \
        --submit-window 16 \
        --kv-block-budget 384 \
        --gpu-memory-utilization 0.7 \
        --goodput-e2e-sla-ms "${goodput_sla_ms}" \
        --goodput-e2e-sla-sweep-ms "${GOODPUT_SLA_SWEEP_MS}" \
        --output-json "${routing_json}" \
        --output-figure "${model_out}/routing/${routing_stem}.png" \
        2>&1 | tee "${model_out}/routing/${routing_stem}.log"
    fi
  done

  local memory_skew_json="${model_out}/memory_skew/summary.json"
  local memory_skew_complete=0
  if artifact_complete \
      "${memory_skew_json}" \
      5 object "${model}" memory-skew "${TRANSFER_INTERFERENCE_MULTIPLIER}" \
      "${TRANSFER_FIXED_LATENCY_MS}" \
      && jq -e \
        --argjson expected_sla "${goodput_sla_ms}" \
        --arg expected_sweep "${GOODPUT_SLA_SWEEP_MS}" \
        '.metadata.arguments.memory_skew_prefix_groups == 12
         and .metadata.arguments.memory_skew_warmup_prompts == 24
         and .metadata.arguments.memory_skew_pressure_prompts == 64
         and .metadata.arguments.num_prompts == 256
         and .metadata.arguments.prompt_repeat == 32
         and .metadata.arguments.max_tokens == 16
         and .metadata.arguments.submit_window == 32
         and .metadata.arguments.disable_background_copy == true
         and .metadata.arguments.goodput_e2e_sla_ms
             == $expected_sla
         and .metadata.arguments.goodput_e2e_sla_sweep_ms
             == $expected_sweep' \
        "${memory_skew_json}" >/dev/null; then
    memory_skew_complete=1
  fi
  if [[ "${memory_skew_complete}" == "1" ]]; then
    echo "[resume] reuse ${model_out}/memory_skew/summary.json"
  else
    CUDA_VISIBLE_DEVICES="${GPU_SET}" uv run python benchmarks/benchmark_e2e.py \
      --model-name-or-path "${model}" \
      --dtype "${DTYPE}" \
      --world-size "${WORLD_SIZE}" \
      --workload memory-skew \
      --memory-skew-prefix-groups 12 \
      --memory-skew-warmup-prompts 24 \
      --memory-skew-pressure-prompts 64 \
      --num-prompts 256 \
      --prompt-repeat 32 \
      --max-tokens 16 \
      --temperature 0.6 \
      --ignore-eos \
      --seed "${SEED}" \
      --repetitions "${REPETITIONS}" \
      --nvlink-pairs "${NVLINK_PAIRS}" \
      --submit-window 32 \
      --kv-block-budget 64 \
      --gpu-memory-utilization 0.5 \
      --goodput-e2e-sla-ms "${goodput_sla_ms}" \
      --goodput-e2e-sla-sweep-ms "${GOODPUT_SLA_SWEEP_MS}" \
      --disable-background-copy \
      --foreground-transfer-min-benefit-ratio 1.1 \
      --foreground-transfer-profile-json "${transfer_profile}" \
      --foreground-transfer-fixed-latency-ms "${TRANSFER_FIXED_LATENCY_MS}" \
      --foreground-transfer-interference-multiplier "${TRANSFER_INTERFERENCE_MULTIPLIER}" \
      --kv-transfer-prewarm-blocks 4 \
      --output-json "${memory_skew_json}" \
      --output-figure "${model_out}/memory_skew/summary.png" \
      2>&1 | tee "${model_out}/memory_skew/run.log"
  fi

  if artifact_complete \
    "${model_out}/load_skew/summary.json" \
    5 object "${model}" load-skew "${TRANSFER_INTERFERENCE_MULTIPLIER}" \
    "${TRANSFER_FIXED_LATENCY_MS}" \
    && jq -e \
      --argjson expected_sla "${goodput_sla_ms}" \
      --arg expected_sweep "${GOODPUT_SLA_SWEEP_MS}" \
      '.metadata.arguments.load_skew_prefix_groups == 24
       and .metadata.arguments.load_skew_warmup_prompts == 48
       and .metadata.arguments.num_prompts == 192
       and .metadata.arguments.prompt_repeat == 48
       and .metadata.arguments.max_tokens == 8
       and .metadata.arguments.submit_window == 64
       and .metadata.arguments.kv_block_budget == 192
       and .metadata.arguments.disable_background_copy == false
       and .metadata.arguments.goodput_e2e_sla_ms
           == $expected_sla
       and .metadata.arguments.goodput_e2e_sla_sweep_ms
           == $expected_sweep' \
      "${model_out}/load_skew/summary.json" >/dev/null; then
    echo "[resume] reuse ${model_out}/load_skew/summary.json"
  else
    CUDA_VISIBLE_DEVICES="${GPU_SET}" uv run python benchmarks/benchmark_e2e.py \
      --model-name-or-path "${model}" \
      --dtype "${DTYPE}" \
      --world-size "${WORLD_SIZE}" \
      --workload load-skew \
      --load-skew-prefix-groups 24 \
      --load-skew-warmup-prompts 48 \
      --num-prompts 192 \
      --prompt-repeat 48 \
      --max-tokens 8 \
      --temperature 0.6 \
      --ignore-eos \
      --seed "${SEED}" \
      --repetitions "${REPETITIONS}" \
      --nvlink-pairs "${NVLINK_PAIRS}" \
      --submit-window 64 \
      --kv-block-budget 192 \
      --gpu-memory-utilization 0.7 \
      --goodput-e2e-sla-ms "${goodput_sla_ms}" \
      --goodput-e2e-sla-sweep-ms "${GOODPUT_SLA_SWEEP_MS}" \
      --background-copy-max-blocks 24 \
      --background-copy-batch-max-blocks 48 \
      --background-copy-batch-max-candidates 4 \
      --background-copy-hot-threshold 2 \
      --background-copy-min-load-skew 2 \
      --background-copy-expected-reuses 8 \
      --background-copy-cooldown-s 0.1 \
      --foreground-transfer-min-benefit-ratio 1.1 \
      --foreground-transfer-profile-json "${transfer_profile}" \
      --foreground-transfer-fixed-latency-ms "${TRANSFER_FIXED_LATENCY_MS}" \
      --foreground-transfer-interference-multiplier "${TRANSFER_INTERFERENCE_MULTIPLIER}" \
      --kv-transfer-prewarm-blocks 4 \
      --output-json "${model_out}/load_skew/summary.json" \
      --output-figure "${model_out}/load_skew/summary.png" \
      2>&1 | tee "${model_out}/load_skew/run.log"
  fi

  printf '%s\n' "${model}" > "${model_out}/SUITE_COMPLETE"
  echo "[suite] completed ${label}"
}

run_model_suite "qwen3-0.6b" "${MODEL_06B}" "${GOODPUT_SLA_MS_06B}"
run_model_suite "qwen3-1.7b" "${MODEL_17B}" "${GOODPUT_SLA_MS_17B}"

echo "paper benchmark suite completed: ${OUT}"
