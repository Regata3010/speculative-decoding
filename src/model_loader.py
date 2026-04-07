"""Load draft and target models with shared tokenizer."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEFAULT_TARGET_MODEL = "meta-llama/Llama-3.1-8B"
DEFAULT_DRAFT_MODEL = "meta-llama/Llama-3.2-1B"


def _get_quant_config(precision: str, compute_dtype: torch.dtype) -> BitsAndBytesConfig | None:
    """Return BitsAndBytesConfig for a given precision string."""
    if precision == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
        )
    elif precision == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def load_models(
    target_model_id: str = DEFAULT_TARGET_MODEL,
    draft_model_id: str = DEFAULT_DRAFT_MODEL,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    compile_models: bool = False,
    compile_draft_only: bool = False,
    quantize_target: str | None = None,
    quantize_draft: str | None = None,
) -> tuple[AutoModelForCausalLM, AutoModelForCausalLM, AutoTokenizer]:
    """Load target model, draft model, and shared tokenizer.

    Args:
        target_model_id: HuggingFace model ID for the target (large) model.
        draft_model_id: HuggingFace model ID for the draft (small) model.
        device: Device to load models onto. Auto-detected if None.
        dtype: Data type for model weights. Auto-detected if None.
        compile_models: If True, torch.compile() both models.
        compile_draft_only: If True, only torch.compile() the draft model.
        quantize_target: "4bit", "8bit", or None for fp16/bf16.
        quantize_draft: "4bit", "8bit", or None for fp16/bf16.

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

    # Load target model
    target_kwargs = {"device_map": "auto" if device.type == "cuda" else None}
    target_quant = _get_quant_config(quantize_target, dtype) if quantize_target else None
    if target_quant:
        target_kwargs["quantization_config"] = target_quant
        print(f"  Target: {quantize_target} quantization")
    else:
        target_kwargs["dtype"] = dtype
        print(f"  Target: {dtype}")

    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_id, **target_kwargs
    )
    if device.type != "cuda" and quantize_target is None:
        target_model = target_model.to(device)
    target_model.eval()

    # Load draft model
    draft_kwargs = {"device_map": "auto" if device.type == "cuda" else None}
    draft_quant = _get_quant_config(quantize_draft, dtype) if quantize_draft else None
    if draft_quant:
        draft_kwargs["quantization_config"] = draft_quant
        print(f"  Draft:  {quantize_draft} quantization")
    else:
        draft_kwargs["dtype"] = dtype
        print(f"  Draft:  {dtype}")

    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_model_id, **draft_kwargs
    )
    if device.type != "cuda" and quantize_draft is None:
        draft_model = draft_model.to(device)
    draft_model.eval()

    # torch.compile()
    if device.type == "cuda":
        if compile_models:
            print("  Compiling both models...")
            target_model = torch.compile(target_model, mode="reduce-overhead")
            draft_model = torch.compile(draft_model, mode="reduce-overhead")
        elif compile_draft_only:
            print("  Compiling draft model...")
            draft_model = torch.compile(draft_model, mode="reduce-overhead")

    return target_model, draft_model, tokenizer
