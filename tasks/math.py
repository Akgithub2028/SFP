"""
tasks/math.py — Math reasoning evaluation (GSM8K).

~80 lines. Standalone — no imports from other project files.
"""

from __future__ import annotations


def evaluate(model, tokenizer, mode: str = "fast") -> dict[str, float]:
    """Evaluate math reasoning.

    fast: GSM8K test subset (200 examples), exact-match on final number.
    full: LiveBench Math + Reasoning.

    Returns: {"math_accuracy": float}
    """
    raise NotImplementedError
