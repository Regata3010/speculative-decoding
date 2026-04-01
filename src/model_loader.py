"""Load draft and target models with shared tokenizer."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEFAULT_TARGET_MODEL = "meta-llama/Llama-3.1-8B"
DEFAULT_DRAFT_MODEL = "meta-llama/Llama-3.2-1B"


def load_models(
    target_model_id: str = DEFAULT_TARGET_MODEL,
    draft_model_id: str = DEFAULT_DRAFT_MODEL,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    compile_models: bool = False,
    compile_draft_only: bool = False,
    quantize_target: str | None = None,
) -> tuple[AutoModelForCausalLM, AutoModelForCausalLM, AutoTokenizer]:
    """Load target model, draft model, and shared tokenizer.

    Args:
        target_model_id: HuggingFace model ID for the target (large) model.
        draft_model_id: HuggingFace model ID for the draft (small) model.
        device: Device to load models onto. Auto-detected if None.
        dtype: Data type for model weights. Auto-detected if None.
        compile_models: If True, torch.compile() both models.
        compile_draft_only: If True, only torch.compile() the draft model.
        quantize_target: If "4bit" or "8bit", quantize the target model
            to fit large models (70B+) on a single GPU.

    Returns:
        (target_model, draft_model, tokenizer)
    """
    if device is None:
        from src.utils import get_device
        device = get_device()

    if dtype is None:
        from src.utils import get_dtype
        dtype = get_dtype(device)

    # Load tokenizer from target model (draft shares the same one)
    tokenizer = AutoTokenizer.from_pretrained(target_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build target model kwargs
    target_kwargs = {"device_map": "auto" if device.type == "cuda" else None}

    if quantize_target == "4bit":
        print(f"  Quantizing target to 4-bit (NF4)...")
        target_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
        )
    elif quantize_target == "8bit":
        print(f"  Quantizing target to 8-bit...")
        target_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        target_kwargs["dtype"] = dtype

    # Load target model
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_id, **target_kwargs
    )
    if device.type != "cuda" and quantize_target is None:
        target_model = target_model.to(device)
    target_model.eval()

    # Load draft model (always in fp16/bf16 — small enough)
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_model_id,
        dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
    )
    if device.type != "cuda":
        draft_model = draft_model.to(device)
    draft_model.eval()

    # torch.compile() — CUDA graphs eliminate kernel launch overhead
    if device.type == "cuda":
        if compile_models:
            print("Compiling both models with torch.compile (reduce-overhead)...")
            target_model = torch.compile(target_model, mode="reduce-overhead")
            draft_model = torch.compile(draft_model, mode="reduce-overhead")
        elif compile_draft_only:
            print("Compiling draft model with torch.compile (reduce-overhead)...")
            draft_model = torch.compile(draft_model, mode="reduce-overhead")

    return target_model, draft_model, tokenizer
