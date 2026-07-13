"""
KV routing benchmark.

Runs only the scenarios needed to isolate cache-aware routing:
single-gpu, multi-gpu round-robin, and multi-gpu-kv-routing.
"""

from dataclasses import asdict

import torch
from transformers import AutoTokenizer

from shared_prefix_benchmark import (
    MODEL_CONFIG,
    SamplingParams,
    build_prompts,
    make_config,
    measure_single_gpu_prefix_hit_rate,
    parse_args,
    parse_pairs,
    print_summary_table,
    run_engine_scenario,
    save_summary_figure,
    save_summary_json,
)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")

    model_name = args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompts = build_prompts(tokenizer, args.num_prompts, args.prompt_repeat)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_model_length=MODEL_CONFIG["max_model_length"],
    )
    goodput_e2e_sla_s = args.goodput_e2e_sla_ms / 1000.0

    baseline_config = make_config(1, False, None)
    baseline_config["model_name_or_path"] = model_name
    baseline = run_engine_scenario(
        "single-gpu",
        baseline_config,
        prompts,
        sampling_params,
        tokenizer,
        goodput_e2e_sla_s=goodput_e2e_sla_s,
        submit_window=args.submit_window,
    )
    baseline.prefix_hit_rate = measure_single_gpu_prefix_hit_rate(
        tokenizer,
        prompts,
        block_size=baseline_config["block_size"],
        max_cached_blocks=baseline_config["max_cached_blocks"],
    )

    multi_gpu_config = make_config(2, False, None)
    multi_gpu_config["model_name_or_path"] = model_name
    multi_gpu = run_engine_scenario(
        "multi-gpu",
        multi_gpu_config,
        prompts,
        sampling_params,
        tokenizer,
        route_mode="round_robin",
        goodput_e2e_sla_s=goodput_e2e_sla_s,
        submit_window=args.submit_window,
    )

    routing_config = make_config(2, True, parse_pairs(args.nvlink_pairs) if args.nvlink_pairs else None)
    routing_config["model_name_or_path"] = model_name
    routing_config["max_cached_blocks"] = args.routing_max_cached_blocks
    kv_routing = run_engine_scenario(
        "multi-gpu-kv-routing",
        routing_config,
        prompts,
        sampling_params,
        tokenizer,
        route_mode="control_plane",
        goodput_e2e_sla_s=goodput_e2e_sla_s,
        submit_window=args.submit_window,
    )

    results = [baseline, multi_gpu, kv_routing]
    print_summary_table(results)
    if args.output_figure:
        save_summary_figure(results, args.output_figure)
    if args.output_json:
        save_summary_json(
            {
                "single-gpu": asdict(baseline),
                "multi-gpu": asdict(multi_gpu),
                "multi-gpu-kv-routing": asdict(kv_routing),
            },
            args.output_json,
        )


if __name__ == "__main__":
    main()
