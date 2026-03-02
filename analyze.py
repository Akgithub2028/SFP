"""
analyze.py — Drift computation, regression, publication plots.

~200 lines. Uses numpy, matplotlib, scikit-learn.
Reads results from JSON files on disk — no imports from other project files at runtime.

Usage:
    python analyze.py --run out/naive_seed42/
    python analyze.py --compare out/naive_* out/sfp_*
    python analyze.py --gate1 out/gate1/
    python analyze.py --paper out/full/
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


def compute_drift(
    acts_before: NDArray, acts_after: NDArray,
    basis: NDArray, r: int,
) -> float:
    """D = E_x || U_r^T a(x;θ_after) − U_r^T a(x;θ_before) ||²"""
    raise NotImplementedError


def compute_drift_all(run_dir: str) -> dict:
    """Compute drift for all task transitions in a run.

    Returns: {(task_from, task_to): {layer: {"top_r": float, "random_r": float, "full": float}}}
    """
    raise NotImplementedError


def forgetting_regression(
    drift_scores: NDArray, forgetting_scores: NDArray,
) -> dict:
    """Fit OLS: forgetting ~ drift.

    Returns: {"r2": float, "coef": float, "p_value": float, "ci_lower": float, "ci_upper": float}
    """
    raise NotImplementedError


def plot_pareto(
    results_by_method: dict,
    save_path: str = "fig_pareto.pdf",
) -> None:
    """Retention vs plasticity scatter with Pareto frontier."""
    raise NotImplementedError


def plot_drift_vs_forgetting(
    drift: NDArray, forgetting: NDArray,
    save_path: str = "fig_regression.pdf",
) -> None:
    """Scatter + regression line + R² annotation."""
    raise NotImplementedError


def plot_drift_heatmap(
    drift_by_layer_task: dict,
    save_path: str = "fig_drift_heatmap.pdf",
) -> None:
    """Layer (y) × task transition (x) heatmap."""
    raise NotImplementedError


def print_gate_results(
    regression_results: dict,
    pareto_results: dict | None = None,
) -> None:
    """Print GATE 1 / GATE 2 pass/fail with numbers."""
    raise NotImplementedError


if __name__ == "__main__":
    raise NotImplementedError("CLI not yet implemented")
