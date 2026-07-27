#!/usr/bin/env bash
set -euo pipefail

# Run the minimum Qwen3-0.6B workload gate before the dual-model paper suite.
# The preflight reuses an existing compatible transfer profile and does not
# regenerate microbenchmark or transaction-calibration artifacts.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uvcache}"

GPU_SET="${GPU_SET:-0,1,3,4,5,6}"
WORLD_SIZE="${WORLD_SIZE:-6}"
NVLINK_PAIRS="${NVLINK_PAIRS:-0,1;2,3;4,5}"
REPETITIONS="${REPETITIONS:-1}"
SEED="${SEED:-0}"
DTYPE="${DTYPE:-auto}"
GOODPUT_SLA_MS="${GOODPUT_SLA_MS:-3000}"
GOODPUT_SLA_SWEEP_MS="${GOODPUT_SLA_SWEEP_MS:-2000,3000,5000,10000}"
RESUME="${RESUME:-0}"
PREFLIGHT_ID="${PREFLIGHT_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-benchmarks/results/preflight/${PREFLIGHT_ID}/qwen3-0.6b}"
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
  if [[ ! -d benchmarks/results/paper ]]; then
    return 0
  fi
  find benchmarks/results/paper -path \
    '*/qwen3-0.6b/kv_transfer/latency_profile.json' -type f -print \
    2>/dev/null | sort | tail -n 1
}

MODEL="${MODEL:-${MODEL_06B:-$(find_snapshot models--Qwen--Qwen3-0.6B)}}"
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
  echo "TRANSFER_PROFILE must identify a compatible Qwen3-0.6B latency profile" >&2
  exit 2
fi
if ! command -v jq >/dev/null; then
  echo "jq is required for artifact validation" >&2
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

mkdir -p "${OUT}"/{environment,routing,memory_skew,load_skew}
nvidia-smi -L > "${OUT}/environment/gpus.txt"
nvidia-smi topo -m > "${OUT}/environment/topology.txt"
git rev-parse HEAD > "${OUT}/environment/git_revision.txt"
git status --short > "${OUT}/environment/git_status.txt"
printf '%s\n' "${TRANSFER_PROFILE}" > "${OUT}/environment/transfer_profile.txt"

artifact_complete() {
  local path="$1"
  local expected_results="$2"
  local expected_workload="${3:-}"
  [[ "${RESUME}" == "1" ]] \
    && [[ -s "${path}" ]] \
    && jq -e \
      --argjson expected_results "${expected_results}" \
      --arg expected_model "${MODEL}" \
      --arg expected_workload "${expected_workload}" \
      --argjson expected_repetitions "${REPETITIONS}" \
      '.metadata != null
       and (.results | type) == "object"
       and (.results | length == $expected_results)
       and .metadata.model.name_or_path == $expected_model
       and .metadata.arguments.repetitions == $expected_repetitions
       and (
         if $expected_workload == ""
         then true
         else .metadata.arguments.workload == $expected_workload
         end
       )
       and all(.results[];
         has("mean_tpot_s") and has("goodput_sla_sweep_tok_s")
       )' \
      "${path}" >/dev/null
}

routing_json="${OUT}/routing/prefix_5x.json"
routing_complete=0
if artifact_complete "${routing_json}" 3 \
    && jq -e \
      --argjson expected_sla "${GOODPUT_SLA_MS}" \
      --arg expected_sweep "${GOODPUT_SLA_SWEEP_MS}" \
      '.metadata.arguments.num_prompts == 192
       and .metadata.arguments.prompt_repeat == 80
       and .metadata.arguments.max_tokens == 8
       and .metadata.arguments.max_model_length == 10240
       and .metadata.arguments.locality_prefix_groups == 16
       and .metadata.arguments.kv_block_budget == 384
       and .metadata.arguments.goodput_e2e_sla_ms == $expected_sla
       and .metadata.arguments.goodput_e2e_sla_sweep_ms == $expected_sweep' \
      "${routing_json}" >/dev/null; then
  routing_complete=1
fi
if [[ "${routing_complete}" == "1" ]]; then
  echo "[resume] reuse ${routing_json}"
else
  echo "[preflight] routing 5x"
  CUDA_VISIBLE_DEVICES="${GPU_SET}" uv run python benchmarks/benchmark_kv_routing.py \
    --model-name-or-path "${MODEL}" \
    --dtype "${DTYPE}" \
    --world-size "${WORLD_SIZE}" \
    --num-prompts 192 \
    --prompt-repeat 80 \
    --max-tokens 8 \
    --max-model-length 10240 \
    --max-num-batched-tokens 10240 \
    --temperature 0.6 \
    --ignore-eos \
    --seed "${SEED}" \
    --repetitions "${REPETITIONS}" \
    --locality-prefix-groups 16 \
    --nvlink-pairs "${NVLINK_PAIRS}" \
    --submit-window 16 \
    --kv-block-budget 384 \
    --gpu-memory-utilization 0.7 \
    --goodput-e2e-sla-ms "${GOODPUT_SLA_MS}" \
    --goodput-e2e-sla-sweep-ms "${GOODPUT_SLA_SWEEP_MS}" \
    --output-json "${routing_json}" \
    --output-figure "${OUT}/routing/prefix_5x.png" \
    2>&1 | tee "${OUT}/routing/prefix_5x.log"
fi

memory_json="${OUT}/memory_skew/summary.json"
memory_complete=0
if artifact_complete "${memory_json}" 5 memory-skew \
    && jq -e \
      --arg expected_profile "${TRANSFER_PROFILE}" \
      --argjson expected_sla "${GOODPUT_SLA_MS}" \
      --arg expected_sweep "${GOODPUT_SLA_SWEEP_MS}" \
      '.metadata.arguments.memory_skew_prefix_groups == 12
       and .metadata.arguments.memory_skew_warmup_prompts == 24
       and .metadata.arguments.memory_skew_pressure_prompts == 64
       and .metadata.arguments.num_prompts == 256
       and .metadata.arguments.prompt_repeat == 32
       and .metadata.arguments.max_tokens == 16
       and .metadata.arguments.submit_window == 32
       and .metadata.arguments.kv_block_budget == 64
       and .metadata.arguments.disable_background_copy == true
       and .metadata.arguments.goodput_e2e_sla_ms == $expected_sla
       and .metadata.arguments.goodput_e2e_sla_sweep_ms == $expected_sweep
       and .metadata.arguments.foreground_transfer_profile_json
           == $expected_profile' \
      "${memory_json}" >/dev/null; then
  memory_complete=1
fi
if [[ "${memory_complete}" == "1" ]]; then
  echo "[resume] reuse ${memory_json}"
else
  echo "[preflight] memory skew"
  CUDA_VISIBLE_DEVICES="${GPU_SET}" uv run python benchmarks/benchmark_e2e.py \
    --model-name-or-path "${MODEL}" \
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
    --goodput-e2e-sla-ms "${GOODPUT_SLA_MS}" \
    --goodput-e2e-sla-sweep-ms "${GOODPUT_SLA_SWEEP_MS}" \
    --disable-background-copy \
    --foreground-transfer-min-benefit-ratio 1.1 \
    --foreground-transfer-profile-json "${TRANSFER_PROFILE}" \
    --foreground-transfer-fixed-latency-ms 0.0 \
    --foreground-transfer-interference-multiplier 1.2 \
    --kv-transfer-prewarm-blocks 4 \
    --output-json "${memory_json}" \
    --output-figure "${OUT}/memory_skew/summary.png" \
    2>&1 | tee "${OUT}/memory_skew/run.log"
fi

load_json="${OUT}/load_skew/summary.json"
load_complete=0
if artifact_complete "${load_json}" 5 load-skew \
    && jq -e \
      --arg expected_profile "${TRANSFER_PROFILE}" \
      --argjson expected_sla "${GOODPUT_SLA_MS}" \
      --arg expected_sweep "${GOODPUT_SLA_SWEEP_MS}" \
      '.metadata.arguments.load_skew_prefix_groups == 24
       and .metadata.arguments.load_skew_warmup_prompts == 48
       and .metadata.arguments.num_prompts == 192
       and .metadata.arguments.prompt_repeat == 48
       and .metadata.arguments.max_tokens == 8
       and .metadata.arguments.submit_window == 64
       and .metadata.arguments.kv_block_budget == 192
       and .metadata.arguments.disable_background_copy == false
       and .metadata.arguments.goodput_e2e_sla_ms == $expected_sla
       and .metadata.arguments.goodput_e2e_sla_sweep_ms == $expected_sweep
       and .metadata.arguments.foreground_transfer_profile_json
           == $expected_profile' \
      "${load_json}" >/dev/null; then
  load_complete=1
fi
if [[ "${load_complete}" == "1" ]]; then
  echo "[resume] reuse ${load_json}"
else
  echo "[preflight] load skew"
  CUDA_VISIBLE_DEVICES="${GPU_SET}" uv run python benchmarks/benchmark_e2e.py \
    --model-name-or-path "${MODEL}" \
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
    --goodput-e2e-sla-ms "${GOODPUT_SLA_MS}" \
    --goodput-e2e-sla-sweep-ms "${GOODPUT_SLA_SWEEP_MS}" \
    --background-copy-max-blocks 24 \
    --background-copy-batch-max-blocks 48 \
    --background-copy-batch-max-candidates 4 \
    --background-copy-hot-threshold 2 \
    --background-copy-min-load-skew 2 \
    --background-copy-expected-reuses 8 \
    --background-copy-cooldown-s 0.1 \
    --foreground-transfer-min-benefit-ratio 1.1 \
    --foreground-transfer-profile-json "${TRANSFER_PROFILE}" \
    --foreground-transfer-fixed-latency-ms 0.0 \
    --foreground-transfer-interference-multiplier 1.2 \
    --kv-transfer-prewarm-blocks 4 \
    --output-json "${load_json}" \
    --output-figure "${OUT}/load_skew/summary.png" \
    2>&1 | tee "${OUT}/load_skew/run.log"
fi

failures=0

check_result() {
  local name="$1"
  local path="$2"
  local expression="$3"
  if jq -e "${expression}" "${path}" >/dev/null; then
    echo "[pass] ${name}"
  else
    echo "[fail] ${name}: ${path}" >&2
    failures=$((failures + 1))
  fi
}

benefit_filter='
  def benefit($candidate; $baseline):
    ($candidate.throughput_tok_s > $baseline.throughput_tok_s)
    or ($candidate.mean_ttft_s < $baseline.mean_ttft_s)
    or ($candidate.p90_e2e_s < $baseline.p90_e2e_s);'

check_result "routing locality and load balance" "${routing_json}" "
  ${benefit_filter}
  .results as \$r
  | \$r[\"multi-gpu\"] as \$baseline
  | \$r[\"multi-gpu-kv-routing\"] as \$candidate
  | ((\$candidate.transfer_count // 0) == 0)
    and ((\$candidate.transfer_copy_count // 0) == 0)
    and (\$candidate.prefill_uncached_tokens < \$baseline.prefill_uncached_tokens)
    and (
      [\$candidate.rank_stats[].submitted] as \$submitted
      | ((\$submitted | add) > 0)
        and ((\$submitted | max) / (\$submitted | add) <= 0.35)
    )
    and benefit(\$candidate; \$baseline)"

check_result "memory-skew foreground offload" "${memory_json}" "
  ${benefit_filter}
  def offloaded:
    (.offload_verified == true)
    and ((.rebalance_success // 0) > 0)
    and ((.transfer_release_count // 0) > 0);
  .results as \$r
  | (
      (\$r[\"multi-gpu-kv-transfer\"] | offloaded)
      and benefit(\$r[\"multi-gpu-kv-transfer\"]; \$r[\"multi-gpu\"])
    )
    or (
      (\$r[\"multi-gpu-lmpool\"] | offloaded)
      and benefit(\$r[\"multi-gpu-lmpool\"]; \$r[\"multi-gpu-kv-routing\"])
    )"

check_result "load-skew background relief" "${load_json}" "
  ${benefit_filter}
  def copied:
    ((.background_copy_success // 0) > 0)
    and ((.transfer_copy_count // 0) > 0);
  .results as \$r
    | (
      (\$r[\"multi-gpu-kv-transfer\"] | copied)
      or (\$r[\"multi-gpu-lmpool\"] | copied)
    )
    and (
      ((\$r[\"multi-gpu-lmpool\"].placement_lease_route_count // 0) > 0)
      or ((\$r[\"multi-gpu-lmpool\"].replica_copy_route_count // 0) > 0)
    )
    and (
      (\$r[\"multi-gpu-lmpool\"].reuse_phase_request_hit_rate // 0)
      > (\$r[\"multi-gpu\"].reuse_phase_request_hit_rate // 0)
    )
    and (
      benefit(\$r[\"multi-gpu-kv-transfer\"]; \$r[\"multi-gpu\"])
      or benefit(\$r[\"multi-gpu-lmpool\"]; \$r[\"multi-gpu-kv-routing\"])
    )"

echo
jq -r '
  .results as $r
  | "routing: uncached prefill \($r["multi-gpu"].prefill_uncached_tokens)"
    + " -> \($r["multi-gpu-kv-routing"].prefill_uncached_tokens), "
    + "throughput \($r["multi-gpu"].throughput_tok_s | tostring)"
    + " -> \($r["multi-gpu-kv-routing"].throughput_tok_s | tostring), "
    + "goodput \($r["multi-gpu"].goodput_tok_s | tostring)"
    + " -> \($r["multi-gpu-kv-routing"].goodput_tok_s | tostring), "
    + "TTFT \($r["multi-gpu"].mean_ttft_s | tostring)"
    + " -> \($r["multi-gpu-kv-routing"].mean_ttft_s | tostring)"' \
  "${routing_json}"
jq -r '
  .results as $r
  | "memory: transfer-only fg=\($r["multi-gpu-kv-transfer"].rebalance_success), "
    + "released=\($r["multi-gpu-kv-transfer"].transfer_release_count), "
    + "verified=\($r["multi-gpu-kv-transfer"].offload_verified), "
    + "goodput=\($r["multi-gpu-kv-transfer"].goodput_tok_s); "
    + "LMPool fg=\($r["multi-gpu-lmpool"].rebalance_success), "
    + "released=\($r["multi-gpu-lmpool"].transfer_release_count), "
    + "verified=\($r["multi-gpu-lmpool"].offload_verified), "
    + "goodput=\($r["multi-gpu-lmpool"].goodput_tok_s)"' \
  "${memory_json}"
jq -r '
  .results as $r
  | "load: transfer-only bg=\($r["multi-gpu-kv-transfer"].background_copy_success), "
    + "fg=\($r["multi-gpu-kv-transfer"].rebalance_success); "
    + "LMPool bg=\($r["multi-gpu-lmpool"].background_copy_success), "
    + "fg=\($r["multi-gpu-lmpool"].rebalance_success), "
    + "lease=\($r["multi-gpu-lmpool"].placement_lease_route_count), "
    + "replica=\($r["multi-gpu-lmpool"].replica_copy_route_count), "
    + "reuse hit=\($r["multi-gpu-lmpool"].reuse_phase_request_hit_rate), "
    + "goodput=\($r["multi-gpu-lmpool"].goodput_tok_s)"' \
  "${load_json}"

echo
echo "Preflight artifacts: ${OUT}"
echo "Transfer profile: ${TRANSFER_PROFILE}"
if (( failures > 0 )); then
  echo "preflight failed ${failures} acceptance check(s); do not start the paper suite" >&2
  exit 1
fi

printf '%s\n' "${MODEL}" > "${OUT}/PREFLIGHT_COMPLETE"
echo "preflight passed"
