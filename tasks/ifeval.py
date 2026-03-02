"""
tasks/ifeval.py — Instruction-following compliance evaluation.

~80 lines. Standalone — no imports from other project files.
Objective, judge-free: checks if output satisfies programmatic constraints.
"""

from __future__ import annotations


def evaluate(model, tokenizer, mode: str = "fast") -> dict[str, float]:
    """Evaluate instruction-following compliance.

    Verifiable constraints: word count, format, keyword inclusion, etc.

    Returns: {"ifeval_accuracy": float}
    """
    raise NotImplementedError
