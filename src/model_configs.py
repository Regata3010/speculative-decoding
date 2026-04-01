"""Pre-configured model pairs for benchmarking.

Each pair consists of a target model and a draft model that share the
same tokenizer family. Adding a new pair is as simple as adding an entry.
"""

from dataclasses import dataclass


@dataclass
class ModelPair:
    name: str
    target_id: str
    draft_id: str
    description: str


MODEL_PAIRS = {
    "llama": ModelPair(
        name="llama",
        target_id="meta-llama/Llama-3.1-8B",
        draft_id="meta-llama/Llama-3.2-1B",
        description="Llama 3.1 8B + Llama 3.2 1B (same tokenizer, gated)",
    ),
    "qwen": ModelPair(
        name="qwen",
        target_id="Qwen/Qwen2.5-7B",
        draft_id="Qwen/Qwen2.5-0.5B",
        description="Qwen 2.5 7B + Qwen 2.5 0.5B (same tokenizer, open access)",
    ),
    "qwen-1.5b": ModelPair(
        name="qwen-1.5b",
        target_id="Qwen/Qwen2.5-7B",
        draft_id="Qwen/Qwen2.5-1.5B",
        description="Qwen 2.5 7B + Qwen 2.5 1.5B (stronger draft, 4.7:1 ratio)",
    ),
    "llama-70b": ModelPair(
        name="llama-70b",
        target_id="meta-llama/Llama-3.1-70B",
        draft_id="meta-llama/Llama-3.1-8B",
        description="Llama 3.1 70B + 8B (large target, where spec decoding shines)",
    ),
    "qwen-72b": ModelPair(
        name="qwen-72b",
        target_id="Qwen/Qwen2.5-72B",
        draft_id="Qwen/Qwen2.5-7B",
        description="Qwen 2.5 72B + 7B (large target, open access, no gating)",
    ),
}


def get_model_pair(name: str) -> ModelPair:
    if name not in MODEL_PAIRS:
        available = ", ".join(MODEL_PAIRS.keys())
        raise ValueError(f"Unknown model pair '{name}'. Available: {available}")
    return MODEL_PAIRS[name]


def list_model_pairs() -> str:
    lines = []
    for name, pair in MODEL_PAIRS.items():
        lines.append(f"  {name:10s} {pair.description}")
    return "\n".join(lines)
