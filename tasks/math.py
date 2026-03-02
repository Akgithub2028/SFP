"""
tasks/math.py — Math reasoning evaluation (GSM8K).

~80 lines. Standalone — no imports from other project files.
"""

from __future__ import annotations

import re

import torch
from datasets import load_dataset
from tqdm import tqdm


def _extract_number(text: str) -> str | None:
    """Extract the last number from text (ignoring commas)."""
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if not numbers:
        return None
    return numbers[-1].replace(",", "")


def _extract_gold(answer_text: str) -> str:
    """Extract gold answer from GSM8K (appears after ####)."""
    parts = answer_text.split("####")
    if len(parts) < 2:
        return ""
    return parts[-1].strip().replace(",", "")


@torch.no_grad()
def evaluate(model, tokenizer, mode: str = "fast") -> dict[str, float]:
    """Evaluate math reasoning.

    fast: GSM8K test subset (200 examples), exact-match on final number.
    full: all GSM8K test examples.

    Returns: {"math_accuracy": float}
    """
    ds = load_dataset("openai/gsm8k", "main", split="test")
    n = 200 if mode == "fast" else len(ds)
    ds = ds.select(range(n))

    correct = 0
    total = 0

    for ex in tqdm(ds, desc="math", total=n):
        question = ex["question"]
        gold = _extract_gold(ex["answer"])

        prompt = (
            f"Solve the following math problem step by step.\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_ids = outputs[0, inputs["input_ids"].shape[1] :]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        pred = _extract_number(gen_text)
        if pred is not None and pred == gold:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    return {"math_accuracy": accuracy}
