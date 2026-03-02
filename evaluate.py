"""
evaluate.py — Evaluation orchestrator + forgetting metrics.

~200 lines. Imports from tasks/*.py. CLI entry point.

Usage:
    python evaluate.py --checkpoint out/task_2/
    python evaluate.py --run out/my_run/ --all
"""

from __future__ import annotations


def evaluate_all(
    model, tokenizer,
    task_names: list[str],
    mode: str = "fast",
) -> dict[str, dict[str, float]]:
    """Run all task evals.

    Returns: {task_name: {metric_name: value}}
    """
    raise NotImplementedError


def compute_forgetting(results_history: list[dict]) -> dict[str, float]:
    """F_{i,t} = best_i - current_i for each task."""
    raise NotImplementedError


def compute_bwt(results_history: list[dict]) -> float:
    """Backward transfer: mean forgetting across old tasks."""
    raise NotImplementedError


def leaderboard_score(retention: float, plasticity: float) -> float:
    """Composite leaderboard score. 0.6 * retention + 0.4 * plasticity."""
    return 0.6 * retention + 0.4 * plasticity


def print_results_table(results: dict, forgetting: dict | None = None) -> None:
    """Pretty-print results as ASCII table."""
    raise NotImplementedError


if __name__ == "__main__":
    raise NotImplementedError("CLI not yet implemented")
