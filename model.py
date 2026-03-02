"""
model.py — Model loading + LoRA setup + checkpointing.

~100 lines. Uses transformers + peft. No imports from other project files.
"""

from __future__ import annotations
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
import torch
import os


def load_model(
    name: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    lora_rank: int = 64,
    lora_alpha: int = 128,
    target_modules: list[str] | None = None,
    device: str = "auto",
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load a HuggingFace model + tokenizer and apply LoRA.

    Args:
        name: HF model name or path.
        lora_rank: LoRA rank (r).
        lora_alpha: LoRA alpha scaling.
        target_modules: Which modules to apply LoRA to. Defaults to ["q_proj", "v_proj"].
        device: Device string ("auto", "cpu", "cuda", "mps").

    Returns:
        (model, tokenizer) tuple.
    """
    raise NotImplementedError


def save_checkpoint(model: PreTrainedModel, path: str, task_id: int) -> None:
    """Save LoRA adapter weights + config to path/task_{task_id}/."""
    raise NotImplementedError


def load_checkpoint(model: PreTrainedModel, path: str, task_id: int) -> PreTrainedModel:
    """Load LoRA adapter weights from path/task_{task_id}/."""
    raise NotImplementedError
