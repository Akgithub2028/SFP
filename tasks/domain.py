"""
tasks/domain.py — Domain shift evaluation (perplexity + QA).

~60 lines. Standalone — no imports from other project files.
"""

from __future__ import annotations


def evaluate(model, tokenizer, mode: str = "fast") -> dict[str, float]:
    """Evaluate domain adaptation.

    Returns: {
        "domain_ppl": float,         # held-out perplexity
        "domain_qa_accuracy": float,  # domain QA exact-match
    }
    """
    raise NotImplementedError
