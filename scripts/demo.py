#!/usr/bin/env python3
"""Interactive demo for qualitative inspection of speculative decoding."""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_configs import get_model_pair, list_model_pairs
from src.model_loader import load_models
from src.speculative_decoder import SpeculativeDecoder
from src.baseline_decoder import BaselineDecoder
from src.utils import set_seed, gpu_memory_stats


def main():
    parser = argparse.ArgumentParser(
        description="Interactive speculative decoding demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available model pairs:\n{list_model_pairs()}",
    )
    parser.add_argument("--model-pair", default=None, help="Preset model pair: llama, qwen")
    parser.add_argument("--target-model", default=None)
    parser.add_argument("--draft-model", default=None)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Resolve model IDs
    if args.target_model and args.draft_model:
        target_id = args.target_model
        draft_id = args.draft_model
    elif args.model_pair:
        pair = get_model_pair(args.model_pair)
        target_id = args.target_model or pair.target_id
        draft_id = args.draft_model or pair.draft_id
    else:
        pair = get_model_pair("qwen")
        target_id = pair.target_id
        draft_id = pair.draft_id

    print("Loading models...")
    target_model, draft_model, tokenizer = load_models(target_id, draft_id)

    mem = gpu_memory_stats()
    if mem:
        print(f"GPU memory: {mem['allocated_gb']:.1f} GB\n")

    speculative = SpeculativeDecoder(
        target_model, draft_model, tokenizer,
        K=args.K, temperature=args.temperature, top_p=args.top_p,
    )
    baseline = BaselineDecoder(
        target_model, tokenizer,
        temperature=args.temperature, top_p=args.top_p,
    )

    print("=" * 60)
    print("Speculative Decoding Demo")
    print(f"Target: {target_id}")
    print(f"Draft:  {draft_id}")
    print(f"K={args.K}, temp={args.temperature}, top_p={args.top_p}")
    print("=" * 60)
    print("Enter a prompt (or 'quit' to exit):\n")

    while True:
        try:
            prompt = input("> ")
        except (EOFError, KeyboardInterrupt):
            break

        if prompt.strip().lower() in ("quit", "exit", "q"):
            break

        if not prompt.strip():
            continue

        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        # Speculative
        set_seed(args.seed)
        spec_result = speculative.generate(input_ids, max_new_tokens=args.max_new_tokens)

        # Baseline
        set_seed(args.seed)
        base_result = baseline.generate(input_ids, max_new_tokens=args.max_new_tokens)

        speedup = (
            spec_result.tokens_per_second / base_result.tokens_per_second
            if base_result.tokens_per_second > 0 else 0.0
        )

        print(f"\n--- Speculative Output ---")
        print(spec_result.text)
        print(f"\n--- Stats ---")
        print(f"  Speculative: {spec_result.tokens_per_second:.1f} tok/s | "
              f"Acceptance: {spec_result.acceptance_rate:.1%} | "
              f"{spec_result.tokens_per_target_call:.1f} tok/target call")
        print(f"  Baseline:    {base_result.tokens_per_second:.1f} tok/s")
        print(f"  Speedup:     {speedup:.2f}x")
        print()


if __name__ == "__main__":
    main()
