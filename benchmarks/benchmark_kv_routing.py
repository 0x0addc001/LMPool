"""Routing-only locality benchmark.

This entry isolates the cache-locality contribution of global routing. It runs
three configurations with the same prompt trace and per-rank KV budget:

* single-gpu: local-cache reference;
* multi-gpu: topology-blind round-robin baseline;
* multi-gpu-kv-routing: global page-table lookup and routing, with every
  transfer path disabled.

Example:
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES=0,1,3,4,5,6 UV_CACHE_DIR=/tmp/uvcache \
  uv run python benchmarks/benchmark_kv_routing.py \
    --model-name-or-path /path/to/Qwen3-0.6B \
    --world-size 6 \
    --num-prompts 128 \
    --prompt-repeat 16 \
    --max-tokens 64 \
    --locality-prefix-groups 16 \
    --nvlink-pairs "0,1;2,3;4,5" \
    --submit-window 16 \
    --kv-block-budget 128 \
    --gpu-memory-utilization 0.5 \
    --repetitions 5 \
    --output-json benchmarks/results/routing.json \
    --output-figure benchmarks/results/routing.png
"""

import argparse
from dataclasses import asdict

import torch
from transformers import AutoTokenizer

try:
    from .benchmark_e2e import (
        MODEL_CONFIG,
        SamplingParams,
        build_prompts,
        make_config,
        measure_single_gpu_prefix_hit_rate,
        parse_pairs,
        print_summary_table,
        run_repeated_engine_scenario,
        save_summary_figure,
        save_summary_json,
    )
except ImportError:
    from benchmark_e2e import (
        MODEL_CONFIG,
        SamplingParams,
        build_prompts,
        make_config,
        measure_single_gpu_prefix_hit_rate,
        parse_pairs,
        print_summary_table,
        run_repeated_engine_scenario,
        save_summary_figure,
        save_summary_json,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cache-aware routing locality benchmark"
    )
    parser.add_argument("--num-prompts", type=int, default=128)
    parser.add_argument("--prompt-repeat", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument(
        "--ignore-eos", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--locality-prefix-groups", type=int, default=16)
    parser.add_argument(
        "--model-name-or-path",
        default=MODEL_CONFIG["model_name_or_path"],
    )
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--nvlink-pairs", default="0,1")
    parser.add_argument(
        "--kv-block-budget", type=int, default=MODEL_CONFIG["max_cached_blocks"]
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=MODEL_CONFIG["gpu_memory_utilization"],
    )
    parser.add_argument("--submit-window", type=int, default=16)
    parser.add_argument("--goodput-e2e-sla-ms", type=float, default=10000.0)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-figure", default="")
    parser.add_argument(
        "--route-load-weight",
        type=float,
        default=MODEL_CONFIG["route_load_weight"],
    )
    parser.add_argument(
        "--route-decode-token-weight",
        type=float,
        default=MODEL_CONFIG["route_decode_token_weight"],
    )
    parser.add_argument(
        "--route-owner-spill-sequence-skew",
        type=float,
        default=MODEL_CONFIG["route_owner_spill_sequence_skew"],
    )
    parser.add_argument(
        "--route-owner-spill-max-extra-cost",
        type=float,
        default=MODEL_CONFIG["route_owner_spill_max_extra_cost"],
    )
    parser.add_argument(
        "--route-load-bypass-threshold",
        type=float,
        default=MODEL_CONFIG["route_load_bypass_threshold"],
    )
    parser.add_argument(
        "--route-prefill-cost-weight",
        type=float,
        default=MODEL_CONFIG["route_prefill_cost_weight"],
    )
    parser.add_argument(
        "--route-reclaim-cost-weight",
        type=float,
        default=MODEL_CONFIG["route_reclaim_cost_weight"],
    )
    parser.add_argument(
        "--route-cache-queue-slack",
        type=float,
        default=MODEL_CONFIG["route_cache_queue_slack"],
    )
    return parser.parse_args()


def _validate_args(args) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")
    if args.world_size < 2:
        raise SystemExit("--world-size must be >= 2 for a routing comparison")
    if args.world_size > torch.cuda.device_count():
        raise SystemExit(
            f"--world-size {args.world_size} exceeds visible CUDA devices "
            f"{torch.cuda.device_count()}"
        )
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be >= 1")
    if args.kv_block_budget < 1:
        raise SystemExit("--kv-block-budget must be >= 1")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        raise SystemExit("--gpu-memory-utilization must be in (0, 1]")
    if not 1 <= args.locality_prefix_groups <= args.num_prompts:
        raise SystemExit(
            "--locality-prefix-groups must be between 1 and --num-prompts"
        )


def _configure(config: dict, args) -> dict:
    config["model_name_or_path"] = args.model_name_or_path
    config["max_cached_blocks"] = args.kv_block_budget
    config["gpu_memory_utilization"] = args.gpu_memory_utilization
    config["require_exact_kv_block_budget"] = True
    config["random_seed"] = args.seed
    return config


def _configure_routing(config: dict, args) -> dict:
    _configure(config, args)
    config["enable_foreground_rebalance"] = False
    config["enable_background_copy"] = False
    config["preserve_cache_via_transfer"] = False
    config["route_load_weight"] = args.route_load_weight
    config["route_decode_token_weight"] = args.route_decode_token_weight
    config["route_owner_spill_sequence_skew"] = (
        args.route_owner_spill_sequence_skew
    )
    config["route_owner_spill_max_extra_cost"] = (
        args.route_owner_spill_max_extra_cost
    )
    config["route_load_bypass_threshold"] = args.route_load_bypass_threshold
    config["route_prefill_cost_weight"] = args.route_prefill_cost_weight
    config["route_reclaim_cost_weight"] = args.route_reclaim_cost_weight
    config["route_cache_queue_slack"] = args.route_cache_queue_slack
    return config


def main():
    args = parse_args()
    _validate_args(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    prompts = build_prompts(
        tokenizer,
        args.num_prompts,
        args.prompt_repeat,
        workload="locality",
        locality_prefix_groups=args.locality_prefix_groups,
        seed=args.seed,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        ignore_eos=args.ignore_eos,
        max_model_length=MODEL_CONFIG["max_model_length"],
    )
    common = {
        "prompts": prompts,
        "sampling_params": sampling_params,
        "tokenizer": tokenizer,
        "goodput_e2e_sla_s": args.goodput_e2e_sla_ms / 1000.0,
        "submit_window": args.submit_window,
        "workload": "locality",
    }

    single_config = _configure(make_config(1, False, None), args)
    single = run_repeated_engine_scenario(
        args.repetitions,
        name="single-gpu",
        config=single_config,
        route_mode="round_robin",
        **common,
    )

    multi_config = _configure(make_config(args.world_size, False, None), args)
    multi = run_repeated_engine_scenario(
        args.repetitions,
        name="multi-gpu",
        config=multi_config,
        route_mode="round_robin",
        **common,
    )

    pairs = parse_pairs(args.nvlink_pairs) if args.nvlink_pairs else None
    routing_config = _configure_routing(
        make_config(args.world_size, True, pairs), args
    )
    routing = run_repeated_engine_scenario(
        args.repetitions,
        name="multi-gpu-kv-routing",
        config=routing_config,
        route_mode="control_plane",
        **common,
    )

    theoretical_hit = measure_single_gpu_prefix_hit_rate(
        tokenizer,
        prompts,
        block_size=single_config["block_size"],
        max_cached_blocks=single_config["max_cached_blocks"],
    )
    results = [single, multi, routing]
    for result in results:
        result.theoretical_prefix_hit_rate = theoretical_hit

    print_summary_table(results)
    if args.output_figure:
        save_summary_figure(results, args.output_figure)
    if args.output_json:
        save_summary_json(
            {result.name: asdict(result) for result in results},
            args.output_json,
        )


if __name__ == "__main__":
    main()
