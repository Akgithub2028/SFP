"""
tasks/code.py — Code generation evaluation (HumanEval).

~100 lines. Standalone — no imports from other project files.
"""

from __future__ import annotations


def evaluate(model, tokenizer, mode: str = "fast") -> dict[str, float]:
    """Evaluate code generation.

    fast: HumanEval pass@1 (164 problems, execution-based).
    full: SWE-bench Verified small slice.

    Returns: {"code_pass_at_1": float}
    """
    raise NotImplementedError
