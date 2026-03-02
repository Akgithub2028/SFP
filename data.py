"""
data.py — Dataset loading + memory buffers.

~150 lines. Uses HF datasets + torch. No imports from other project files.

Task data sources:
  - math: GSM8K
  - code: MBPP
  - ifeval: instruction-following constraints
  - safety: refusal alignment
  - domain: PubMed / legal
"""

from __future__ import annotations
import torch
import json
import hashlib
import random
from pathlib import Path


def load_task(
    task_name: str,
    split: str = "train",
    max_samples: int | None = None,
) -> list[dict]:
    """Load a task dataset.

    Args:
        task_name: One of "math", "code", "ifeval", "safety", "domain".
        split: "train" or "test".
        max_samples: Cap the number of examples (for fast mode).

    Returns:
        List of {"input": str, "output": str, "task_id": str}.
    """
    raise NotImplementedError


def get_batch(
    dataset: list[dict],
    batch_size: int,
    tokenizer,
    max_length: int = 512,
    device: str = "auto",
) -> dict:
    """Sample a random batch from dataset, tokenize, return tensors.

    Returns:
        {"input_ids": Tensor, "attention_mask": Tensor, "labels": Tensor}
    """
    raise NotImplementedError


def create_memory_buffer(
    dataset: list[dict],
    size: int = 128,
    seed: int = 42,
) -> list[dict]:
    """Uniform random sample from dataset. Returns a frozen list."""
    raise NotImplementedError


def save_memory(buffer: list[dict], path: str) -> None:
    """Save memory buffer as JSONL with SHA256 hash for reproducibility."""
    raise NotImplementedError


def load_memory(path: str) -> list[dict]:
    """Load memory buffer from JSONL file."""
    raise NotImplementedError
