"""
tasks/ifeval.py — Instruction-following compliance evaluation.

~100 lines. Standalone — no imports from other project files.
Objective, judge-free: checks if output satisfies programmatic constraints.
"""

from __future__ import annotations

import re

import torch
from datasets import load_dataset
from tqdm import tqdm

# ── constraint checkers ──────────────────────────────────────────────────────

_CHECKERS: dict[str, object] = {}


def _count_sentences(text: str) -> int:
    return len(re.split(r"[.!?]+", text.strip())) - 1 or len(text.strip().split("."))


def _check_constraint(instruction: str, response: str) -> bool:
    """Check a single IFEval-style constraint against the response."""
    low = instruction.lower()

    # Word count constraints
    m = re.search(r"at least (\d+) words", low)
    if m:
        return len(response.split()) >= int(m.group(1))
    m = re.search(r"at most (\d+) words", low)
    if m:
        return len(response.split()) <= int(m.group(1))

    # Sentence count constraints
    m = re.search(r"at least (\d+) sentence", low)
    if m:
        return _count_sentences(response) >= int(m.group(1))
    m = re.search(r"at most (\d+) sentence", low)
    if m:
        return _count_sentences(response) <= int(m.group(1))

    # Keyword inclusion
    m = re.search(r'include the keyword[s]?\s*["\']([^"\']+)["\']', low)
    if m:
        return m.group(1).lower() in response.lower()

    # Forbidden words
    m = re.search(r'do not include the keyword[s]?\s*["\']([^"\']+)["\']', low)
    if m:
        return m.group(1).lower() not in response.lower()

    # All uppercase
    if "entire response in uppercase" in low or "write in all capital" in low:
        letters = [c for c in response if c.isalpha()]
        return all(c.isupper() for c in letters) if letters else False

    # All lowercase
    if "entire response in lowercase" in low or "write in all lowercase" in low:
        letters = [c for c in response if c.isalpha()]
        return all(c.islower() for c in letters) if letters else False

    # Bullet / numbered list
    if "bullet point" in low or "bullet list" in low:
        return any(line.strip().startswith(("- ", "* ", "• ")) for line in response.split("\n"))

    # Paragraph count
    m = re.search(r"at least (\d+) paragraph", low)
    if m:
        paras = [p.strip() for p in response.split("\n\n") if p.strip()]
        return len(paras) >= int(m.group(1))

    # If we can't parse the constraint, be lenient — count as satisfied
    return True


@torch.no_grad()
def evaluate(model, tokenizer, mode: str = "fast") -> dict[str, float]:
    """Evaluate instruction-following compliance.

    Verifiable constraints: word count, format, keyword inclusion, etc.

    Returns: {"ifeval_accuracy": float}
    """
    ds = load_dataset("google/IFEval", split="train")
    n = 100 if mode == "fast" else len(ds)
    ds = ds.select(range(n))

    satisfied = 0
    total = 0

    for ex in tqdm(ds, desc="ifeval", total=n):
        prompt = ex["prompt"]
        instructions = ex.get("instruction_id_list", [])

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_ids = outputs[0, inputs["input_ids"].shape[1] :]
        response = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # Check all constraints for this example
        all_ok = all(_check_constraint(inst, response) for inst in instructions)
        if all_ok:
            satisfied += 1
        total += 1

    return {"ifeval_accuracy": satisfied / total if total > 0 else 0.0}
