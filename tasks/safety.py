"""
tasks/safety.py — Safety / refusal behavior evaluation.

~60 lines. Standalone — no imports from other project files.
"""

from __future__ import annotations


def evaluate(model, tokenizer, mode: str = "fast") -> dict[str, float]:
    """Evaluate safety behavior.

    Returns: {
        "refusal_rate": float,       # correct refusals on harmful prompts
        "over_refusal_rate": float,   # false refusals on benign prompts
    }
    """
    raise NotImplementedError
