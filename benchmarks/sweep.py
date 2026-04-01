"""Hyperparameter sweep for speculative decoding.

Sweeps over K (draft length), temperature, and top-p to find the
optimal operating point and characterize the tradeoff space.
"""

import json
import os
from datetime import datetime
from itertools import product

import pandas as pd
from tqdm import tqdm

from benchmarks.prompts import get_sweep_subset
from src.baseline_decoder import BaselineDecoder
from src.speculative_decoder import SpeculativeDecoder
from src.utils import set_seed

DEFAULT_K_VALUES = [1, 2, 3, 4, 5, 7, 10, 15]
DEFAULT_TEMPERATURES = [0.0, 0.3, 0.6, 0.8, 1.0]
DEFAULT_TOP_P_VALUES = [0.9, 0.95, 1.0]


def run_sweep(
    target_model,
    draft_model,
    tokenizer,
    K_values: list[int] | None = None,
    temperatures: list[float] | None = None,
    top_p_values: list[float] | None = None,
    n_repeats: int = 2,
    seed: int = 42,
    output_dir: str = "results",
) -> pd.DataFrame:
    """Run hyperparameter sweep and return results as DataFrame.

    Args:
        target_model: The large target model.
        draft_model: The small draft model.
        tokenizer: Shared tokenizer.
        K_values: List of K values to sweep. Defaults to [1,2,3,4,5,7,10,15].
        temperatures: List of temperatures. Defaults to [0.0,0.3,0.6,0.8,1.0].
        top_p_values: List of top-p values. Defaults to [0.9,0.95,1.0].
        n_repeats: Number of repeats per configuration.
        seed: Random seed.
        output_dir: Directory to save results.

    Returns:
        DataFrame with columns: K, temperature, top_p, acceptance_rate,
        tokens_per_sec, speedup, tokens_per_target_call.
    """
    if K_values is None:
        K_values = DEFAULT_K_VALUES
    if temperatures is None:
        temperatures = DEFAULT_TEMPERATURES
    if top_p_values is None:
        top_p_values = DEFAULT_TOP_P_VALUES

    prompts = get_sweep_subset()
    configs = list(product(K_values, temperatures, top_p_values))

    print(f"Sweep: {len(configs)} configs x {len(prompts)} prompts x {n_repeats} repeats")
    print(f"Total runs: {len(configs) * len(prompts) * n_repeats * 2} (spec + baseline)")

    rows = []

    for K, temp, top_p in tqdm(configs, desc="Sweep configs"):
        speculative = SpeculativeDecoder(
            target_model, draft_model, tokenizer,
            K=K, temperature=temp, top_p=top_p,
        )
        baseline = BaselineDecoder(
            target_model, tokenizer,
            temperature=temp, top_p=top_p,
        )

        for prompt in prompts:
            for i in range(n_repeats):
                set_seed(seed + i)
                input_ids = tokenizer.encode(prompt.prompt_text, return_tensors="pt")

                spec_result = speculative.generate(
                    input_ids, max_new_tokens=prompt.max_new_tokens
                )

                set_seed(seed + i)
                base_result = baseline.generate(
                    input_ids, max_new_tokens=prompt.max_new_tokens
                )

                speedup = (
                    spec_result.tokens_per_second / base_result.tokens_per_second
                    if base_result.tokens_per_second > 0
                    else 0.0
                )

                rows.append({
                    "K": K,
                    "temperature": temp,
                    "top_p": top_p,
                    "prompt_name": prompt.name,
                    "category": prompt.category,
                    "repeat": i,
                    "acceptance_rate": spec_result.acceptance_rate,
                    "spec_tokens_per_sec": spec_result.tokens_per_second,
                    "base_tokens_per_sec": base_result.tokens_per_second,
                    "speedup": speedup,
                    "tokens_per_target_call": spec_result.tokens_per_target_call,
                    "n_generated_tokens": spec_result.n_generated_tokens,
                })

    df = pd.DataFrame(rows)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"sweep_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSweep results saved to {csv_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SWEEP SUMMARY (averaged across prompts and repeats)")
    print("=" * 60)
    summary = df.groupby(["K", "temperature"]).agg(
        acceptance_rate=("acceptance_rate", "mean"),
        speedup=("speedup", "mean"),
    ).round(3)
    print(summary.to_string())

    return df
