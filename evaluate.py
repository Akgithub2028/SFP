"""
evaluate.py — Evaluation orchestrator + forgetting metrics.

~200 lines. Imports from tasks/*.py. CLI entry point.

Usage:
    python evaluate.py --checkpoint out/task_2/
    python evaluate.py --run out/my_run/
    python evaluate.py --aggregate out/run1 out/run2 --format leaderboard
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path

# ── Primary metric per task (used for forgetting / retention) ────────────

PRIMARY_METRIC = {
    "math": "math_accuracy",
    "code": "code_pass_at_1",
    "ifeval": "ifeval_accuracy",
    "safety": "safety_score",
    "domain": "domain_qa_accuracy",
}

ALL_TASKS = list(PRIMARY_METRIC.keys())


# ── Core evaluation ─────────────────────────────────────────────────────

def evaluate_all(
    model,
    tokenizer,
    task_names: list[str],
    mode: str = "fast",
) -> dict[str, dict[str, float]]:
    """Run all task evals.

    Returns: {task_name: {metric_name: value}}
    """
    results = {}
    for name in task_names:
        mod = importlib.import_module(f"tasks.{name}")
        results[name] = mod.evaluate(model, tokenizer, mode=mode)
    return results


# ── Forgetting metrics ──────────────────────────────────────────────────

def compute_forgetting(results_history: list[dict]) -> dict[str, float]:
    """F_{i,t} = best_i - current_i for each task's primary metric.

    results_history: list of dicts from evaluate_all, one per training stage.
    Returns {task_name: forgetting_value} (non-negative; 0 = no forgetting).
    """
    if not results_history:
        return {}

    current = results_history[-1]
    forgetting = {}

    for task, metric in PRIMARY_METRIC.items():
        best = -float("inf")
        for stage in results_history:
            if task in stage and metric in stage[task]:
                best = max(best, stage[task][metric])
        if task in current and metric in current[task]:
            forgetting[task] = max(0.0, best - current[task][metric])

    return forgetting


def compute_bwt(results_history: list[dict]) -> float:
    """Backward transfer: mean forgetting across tasks seen before the final eval."""
    forgetting = compute_forgetting(results_history)
    if not forgetting:
        return 0.0
    # Negative BWT = forgetting occurred; we report magnitude here
    return sum(forgetting.values()) / len(forgetting)


def compute_average_accuracy(results_history: list[dict], task_names: list[str]) -> float:
    """Average Accuracy (AA): mean of primary metrics across all tasks after all training."""
    if not results_history:
        return 0.0
    final = results_history[-1]
    scores = []
    for t in task_names:
        m = PRIMARY_METRIC.get(t, "")
        if t in final and m in final[t]:
            scores.append(final[t][m])
    return sum(scores) / len(scores) if scores else 0.0


def compute_confidence_interval(
    scores: list[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for a list of per-example scores.

    Returns: (mean, ci_lower, ci_upper).
    """
    import random as _rng

    if not scores:
        return 0.0, 0.0, 0.0

    rng = _rng.Random(seed)
    n = len(scores)
    means = []
    for _ in range(n_bootstrap):
        sample = [scores[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)

    means.sort()
    alpha = (1 - ci) / 2
    lo = means[int(alpha * n_bootstrap)]
    hi = means[int((1 - alpha) * n_bootstrap)]
    return sum(scores) / n, lo, hi


def compute_forward_transfer(results_history: list[dict], task_names: list[str]) -> float:
    """Forward Transfer (FWT): average first-seen accuracy (proxy — actual FWT needs zero-shot baselines)."""
    if len(results_history) < 2 or len(task_names) < 2:
        return 0.0
    scores = []
    for i in range(1, len(task_names)):
        if i < len(results_history):
            t = task_names[i]
            m = PRIMARY_METRIC.get(t, "")
            if t in results_history[i] and m in results_history[i][t]:
                scores.append(results_history[i][t][m])
    return sum(scores) / len(scores) if scores else 0.0


def compute_retention_plasticity(
    results_history: list[dict],
    task_names: list[str],
) -> tuple[float, float]:
    """Compute retention and plasticity from a training run.

    retention  = mean accuracy on OLD tasks (all except the last) after all training.
    plasticity = accuracy on the MOST RECENTLY trained task.

    task_names should be in training order.
    """
    if not results_history or not task_names:
        return 0.0, 0.0

    final = results_history[-1]

    # Plasticity: performance on the last task trained
    last_task = task_names[-1]
    metric = PRIMARY_METRIC.get(last_task, "")
    plasticity = final.get(last_task, {}).get(metric, 0.0)

    # Retention: mean performance on all prior tasks
    old_tasks = task_names[:-1]
    if not old_tasks:
        return plasticity, plasticity

    old_scores = []
    for t in old_tasks:
        m = PRIMARY_METRIC.get(t, "")
        if t in final and m in final[t]:
            old_scores.append(final[t][m])

    retention = sum(old_scores) / len(old_scores) if old_scores else 0.0
    return retention, plasticity


def leaderboard_score(retention: float, plasticity: float) -> float:
    """Composite leaderboard score. 0.6 * retention + 0.4 * plasticity."""
    return 0.6 * retention + 0.4 * plasticity


# ── Display ─────────────────────────────────────────────────────────────

def print_results_table(results: dict, forgetting: dict | None = None) -> None:
    """Pretty-print results as ASCII table."""
    # Collect all (task, metric, value) rows
    rows: list[tuple[str, str, float]] = []
    for task in sorted(results):
        for metric, val in sorted(results[task].items()):
            rows.append((task, metric, val))

    if not rows:
        print("(no results)")
        return

    # Column widths
    tw = max(len(r[0]) for r in rows)
    mw = max(len(r[1]) for r in rows)
    header_forget = forgetting is not None
    fw = 10 if header_forget else 0

    hdr = f"{'Task':<{tw}}  {'Metric':<{mw}}  {'Value':>8}"
    if header_forget:
        hdr += f"  {'Forget':>{fw}}"
    sep = "-" * len(hdr)

    print(sep)
    print(hdr)
    print(sep)

    for task, metric, val in rows:
        line = f"{task:<{tw}}  {metric:<{mw}}  {val:>8.4f}"
        if header_forget and task in (forgetting or {}):
            pm = PRIMARY_METRIC.get(task)
            if metric == pm:
                line += f"  {forgetting[task]:>{fw}.4f}"
            else:
                line += f"  {'':>{fw}}"
        print(line)

    print(sep)


def _print_leaderboard_line(name: str, retention: float, plasticity: float) -> None:
    score = leaderboard_score(retention, plasticity)
    print(f"| {name:<20} | {retention:.4f} | {plasticity:.4f} | {score:.4f} |")


# ── Run directory I/O ───────────────────────────────────────────────────

def _load_run_history(run_dir: str) -> list[dict]:
    """Load results_history from a run directory.

    Expects run_dir/task_0/results.json, task_1/results.json, etc.
    """
    run_path = Path(run_dir)
    history = []
    stage = 0
    while True:
        results_file = run_path / f"task_{stage}" / "results.json"
        if not results_file.exists():
            break
        with open(results_file) as f:
            history.append(json.load(f))
        stage += 1
    return history


# ── CLI ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SFP evaluation orchestrator")
    parser.add_argument("--checkpoint", type=str, help="Load model from PATH and evaluate")
    parser.add_argument("--run", type=str, help="Read results history from a run directory")
    parser.add_argument("--aggregate", type=str, nargs="+", help="Aggregate results across runs")
    parser.add_argument("--format", type=str, default="table", choices=["table", "leaderboard"])
    parser.add_argument("--mode", type=str, default="fast", choices=["smoke", "fast", "full"])
    parser.add_argument("--tasks", type=str, nargs="+", default=ALL_TASKS, choices=ALL_TASKS)
    args = parser.parse_args()

    if args.checkpoint:
        from model import load_model

        model, tokenizer = load_model(args.checkpoint)
        results = evaluate_all(model, tokenizer, args.tasks, mode=args.mode)
        print_results_table(results)

    elif args.run:
        history = _load_run_history(args.run)
        if not history:
            print(f"No results found in {args.run}")
            return

        forgetting = compute_forgetting(history)
        bwt = compute_bwt(history)
        aa = compute_average_accuracy(history, args.tasks)
        fwt = compute_forward_transfer(history, args.tasks)
        retention, plasticity = compute_retention_plasticity(history, args.tasks)

        if args.format == "leaderboard":
            run_name = os.path.basename(os.path.normpath(args.run))
            _print_leaderboard_line(run_name, retention, plasticity)
        else:
            print_results_table(history[-1], forgetting=forgetting)
            print(f"\nBWT: {bwt:.4f}  AA: {aa:.4f}  FWT: {fwt:.4f}")
            print(f"Retention: {retention:.4f}  Plasticity: {plasticity:.4f}")
            print(f"Leaderboard: {leaderboard_score(retention, plasticity):.4f}")

    elif args.aggregate:
        print(f"| {'Run':<20} | {'Ret':>7} | {'Plas':>7} | {'AA':>7} | {'Score':>7} |")
        print(f"|{'-'*22}|{'-'*9}|{'-'*9}|{'-'*9}|{'-'*9}|")
        for run_dir in args.aggregate:
            history = _load_run_history(run_dir)
            if not history:
                continue
            run_name = os.path.basename(os.path.normpath(run_dir))
            retention, plasticity = compute_retention_plasticity(history, args.tasks)
            aa = compute_average_accuracy(history, args.tasks)
            score = leaderboard_score(retention, plasticity)
            print(f"| {run_name:<20} | {retention:.4f} | {plasticity:.4f} | {aa:.4f} | {score:.4f} |")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
