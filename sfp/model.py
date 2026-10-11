"""
model.py — Model loading + LoRA setup + checkpointing.

~100 lines. Uses transformers + peft. No imports from other project files.
"""

from __future__ import annotations

import os

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


def load_model(
    name: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    lora_rank: int = 64,
    lora_alpha: int = 128,
    target_modules: list[str] | None = None,
    device: str = "auto",
    full_finetune: bool = False,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load a HuggingFace model + tokenizer, optionally applying LoRA.

    Steps:
      1. Load the base causal-LM and its tokenizer from the HF hub (or local path).
      2. Ensure the tokenizer has a pad token (defaults to eos_token if missing).
      3. If full_finetune is False (default), wrap with a LoRA adapter via peft.
      4. Place the model on the requested device.

    Args:
        name: HF model name or path.
        lora_rank: LoRA rank (r). Ignored when full_finetune=True.
        lora_alpha: LoRA alpha scaling. Ignored when full_finetune=True.
        target_modules: Which modules to apply LoRA to. Defaults to ["q_proj", "v_proj"].
        device: Device string ("auto", "cpu", "cuda", "mps").
        full_finetune: If True, train all parameters (no LoRA adapter).

    Returns:
        (model, tokenizer) tuple.
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    # --- Load base model ---
    # For "auto" we rely on accelerate's device_map when CUDA is available;
    # otherwise fall back to CPU (MPS + device_map="auto" causes Peft issues).
    if device == "auto" and torch.cuda.is_available():
        base_model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    elif device == "auto":
        device = "cpu"
        base_model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=torch.bfloat16,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=torch.bfloat16,
        )

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(name)

    # Pad token fallback — many decoder-only models ship without one.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        base_model.config.pad_token_id = tokenizer.pad_token_id

    if full_finetune:
        # Full fine-tuning: all parameters trainable, no LoRA adapter.
        model = base_model
        if device != "auto":
            model = model.to(device)
        return model, tokenizer

    # --- Apply LoRA adapter ---
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)

    # Move to explicit device when not using "auto" device_map.
    if device != "auto":
        model = model.to(device)

    return model, tokenizer


def save_checkpoint(model: PreTrainedModel, path: str, task_id: int) -> None:
    """Save LoRA adapter weights + config to ``path/task_{task_id}/``.

    Only the LoRA parameters are persisted (not the full base model), keeping
    checkpoints small and fast to write.

    Args:
        model: A PeftModel whose adapter weights will be saved.
        path: Root checkpoint directory.
        task_id: Numeric task identifier used to namespace the subdirectory.
    """
    save_dir = os.path.join(path, f"task_{task_id}")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)


def load_checkpoint(model: PreTrainedModel, path: str, task_id: int) -> PreTrainedModel:
    """Load LoRA adapter weights from ``path/task_{task_id}/``.

    Merges a previously saved adapter into *model* (which should be a base
    model **without** an active adapter).

    Args:
        model: The base HuggingFace model to attach the adapter to.
        path: Root checkpoint directory (same one passed to :func:`save_checkpoint`).
        task_id: Numeric task identifier.

    Returns:
        The model with the loaded LoRA adapter attached.
    """
    load_dir = os.path.join(path, f"task_{task_id}")
    return PeftModel.from_pretrained(model, load_dir)
