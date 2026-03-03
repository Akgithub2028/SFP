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

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy import stats
from sklearn.linear_model import LinearRegression

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
    }
)

# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------


def compute_drift(
    acts_before: NDArray,
    acts_after: NDArray,
    basis: NDArray,
    r: int,
) -> float:
    """D = E_x || U_r^T a(x;theta_after) - U_r^T a(x;theta_before) ||^2."""
    U_r = basis[:, :r]  # [hidden_dim, r]
    proj_before = acts_before @ U_r  # [n, r]
    proj_after = acts_after @ U_r  # [n, r]
    diff = proj_after - proj_before  # [n, r]
    return float(np.mean(np.sum(diff**2, axis=1)))


def compute_drift_all(run_dir: str) -> dict:
    """Compute drift for all task transitions in a run.

    Expects run_dir to contain task_0/, task_1/, ... directories, each with:
      - activations_{layer}.npy   [n_samples, hidden_dim]
      - pca_basis_{layer}.npy     [hidden_dim, k]
      - random_basis_{layer}.npy  [hidden_dim, k]

    Returns: {(task_from, task_to): {layer: {"top_r": float, "random_r": float, "full": float}}}
    """
    run = Path(run_dir)
    task_dirs = sorted(
        [d for d in run.iterdir() if d.is_dir() and d.name.startswith("task_")],
        key=lambda d: int(d.name.split("_")[1]),
    )
    if len(task_dirs) < 2:
        return {}

    # Discover layers from first task dir
    layer_names = sorted(
        {f.stem.replace("activations_", "") for f in task_dirs[0].glob("activations_*.npy")}
    )

    # Load config for r (default 32)
    config_path = run / "config.json"
    r = 32
    if config_path.exists():
        with open(config_path) as f:
            r = json.load(f).get("preserve_r", 32)

    results: dict = {}
    for i in range(len(task_dirs) - 1):
        t_from = int(task_dirs[i].name.split("_")[1])
        t_to = int(task_dirs[i + 1].name.split("_")[1])
        layer_results: dict = {}

        for layer in layer_names:
            acts_before = np.load(task_dirs[i] / f"activations_{layer}.npy")
            acts_after = np.load(task_dirs[i + 1] / f"activations_{layer}.npy")

            pca_path = task_dirs[i] / f"pca_basis_{layer}.npy"
            basis = np.load(pca_path) if pca_path.exists() else np.eye(acts_before.shape[1])

            rand_path = task_dirs[i] / f"random_basis_{layer}.npy"
            rand_basis = np.load(rand_path) if rand_path.exists() else np.eye(acts_before.shape[1])

            r_eff = min(r, basis.shape[1], acts_before.shape[1])
            layer_results[layer] = {
                "top_r": compute_drift(acts_before, acts_after, basis, r_eff),
                "random_r": compute_drift(acts_before, acts_after, rand_basis, r_eff),
                "full": compute_drift(
                    acts_before, acts_after, np.eye(acts_before.shape[1]), acts_before.shape[1]
                ),
            }
        results[(t_from, t_to)] = layer_results

    return results


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


def forgetting_regression(
    drift_scores: NDArray,
    forgetting_scores: NDArray,
) -> dict:
    """Fit OLS: forgetting ~ drift.

    Returns: {"r2": float, "coef": float, "p_value": float, "ci_lower": float, "ci_upper": float}
    """
    X = drift_scores.reshape(-1, 1)
    y = forgetting_scores.ravel()
    reg = LinearRegression().fit(X, y)
    r2 = float(reg.score(X, y))
    coef = float(reg.coef_[0])

    n = len(y)
    y_pred = reg.predict(X)
    residuals = y - y_pred
    se = np.sqrt(np.sum(residuals**2) / (n - 2)) / np.sqrt(np.sum((X.ravel() - X.mean()) ** 2))
    t_stat = coef / se if se > 0 else np.inf
    p_value = float(2 * stats.t.sf(abs(t_stat), df=n - 2))
    t_crit = float(stats.t.ppf(0.975, df=n - 2))
    ci_lower = coef - t_crit * se
    ci_upper = coef + t_crit * se

    return {
        "r2": r2,
        "coef": coef,
        "p_value": p_value,
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]


def plot_pareto(
    results_by_method: dict,
    save_path: str = "fig_pareto.pdf",
) -> None:
    """Retention vs plasticity scatter with Pareto frontier."""
    fig, ax = plt.subplots(figsize=(5, 4))

    all_points: list[tuple[float, float, str]] = []
    for idx, (method, pts) in enumerate(results_by_method.items()):
        x = np.array([p["plasticity"] for p in pts])
        y = np.array([p["retention"] for p in pts])
        ax.scatter(
            x,
            y,
            label=method,
            marker=MARKERS[idx % len(MARKERS)],
            color=COLORS[idx % len(COLORS)],
            s=60,
            zorder=3,
        )
        for xi, yi in zip(x, y, strict=True):
            all_points.append((xi, yi, method))

    # Pareto frontier (maximize both)
    pts_arr = np.array([(p[0], p[1]) for p in all_points])
    if len(pts_arr) > 0:
        sorted_idx = np.argsort(-pts_arr[:, 0])
        frontier = []
        max_y = -np.inf
        for i in sorted_idx:
            if pts_arr[i, 1] > max_y:
                frontier.append(i)
                max_y = pts_arr[i, 1]
        frontier_pts = pts_arr[frontier]
        frontier_pts = frontier_pts[np.argsort(frontier_pts[:, 0])]
        ax.plot(
            frontier_pts[:, 0],
            frontier_pts[:, 1],
            "k--",
            linewidth=1.5,
            alpha=0.6,
            label="Pareto frontier",
        )

    ax.set_xlabel("Plasticity")
    ax.set_ylabel("Retention")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_drift_vs_forgetting(
    drift: NDArray,
    forgetting: NDArray,
    save_path: str = "fig_regression.pdf",
) -> None:
    """Scatter + regression line + R² annotation."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(drift, forgetting, s=30, alpha=0.7, color=COLORS[0], zorder=3)

    reg = forgetting_regression(drift, forgetting)
    x_line = np.linspace(drift.min(), drift.max(), 100)
    lr = LinearRegression().fit(drift.reshape(-1, 1), forgetting)
    y_line = lr.predict(x_line.reshape(-1, 1))
    ax.plot(x_line, y_line, "k-", linewidth=1.5)

    ax.annotate(
        f"$R^2 = {reg['r2']:.3f}$\n$p = {reg['p_value']:.2e}$",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "wheat", "alpha": 0.5},
    )

    ax.set_xlabel("Drift (projected MSE)")
    ax.set_ylabel("Forgetting")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_drift_heatmap(
    drift_by_layer_task: dict,
    save_path: str = "fig_drift_heatmap.pdf",
) -> None:
    """Layer (y) x task transition (x) heatmap.

    drift_by_layer_task: {(from, to): {layer: float}} -- uses top_r drift values.
    """
    transitions = sorted(drift_by_layer_task.keys())
    if not transitions:
        return
    layers = sorted(drift_by_layer_task[transitions[0]].keys())

    matrix = np.array([[drift_by_layer_task[t].get(ly, 0.0) for t in transitions] for ly in layers])

    fig, ax = plt.subplots(figsize=(max(4, len(transitions) * 0.8 + 2), max(3, len(layers) * 0.4)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xticks(range(len(transitions)))
    ax.set_xticklabels([f"{t[0]}→{t[1]}" for t in transitions])
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers)
    ax.set_xlabel("Task transition")
    ax.set_ylabel("Layer")
    fig.colorbar(im, ax=ax, label="Drift")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


# ---------------------------------------------------------------------------
# Gate reporting
# ---------------------------------------------------------------------------


def print_gate_results(
    regression_results: dict,
    pareto_results: dict | None = None,
) -> None:
    """Print GATE 1 / GATE 2 pass/fail with numbers."""
    top_r2 = regression_results.get("top_r2", regression_results.get("r2", 0.0))
    rand_r2 = regression_results.get("random_r2", 0.0)
    gate1_pass = top_r2 > 0.5 and top_r2 > rand_r2 * 2
    status1 = "✓ PASS" if gate1_pass else "✗ FAIL"
    print(f"GATE 1: drift(top-32) R²={top_r2:.2f}, drift(random-32) R²={rand_r2:.2f}  {status1}")

    if pareto_results is not None:
        dominated = pareto_results.get("sfp_dominates", set())
        dom_str = ", ".join(sorted(dominated)) if dominated else "none"
        gate2_pass = len(dominated) >= 2
        status2 = "✓ PASS" if gate2_pass else "✗ FAIL"
        memory = pareto_results.get("memory", "?")
        print(f"GATE 2: sfp dominates {{{dom_str}}} at M={memory}            {status2}")


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def _load_results_json(run_dir: str) -> dict:
    """Load results.json from a run directory."""
    p = Path(run_dir) / "results.json"
    if not p.exists():
        raise FileNotFoundError(f"No results.json in {run_dir}")
    with open(p) as f:
        return json.load(f)


def _run_analysis(run_dir: str) -> None:
    """Full analysis for a single run."""
    drift_all = compute_drift_all(run_dir)
    if not drift_all:
        print(f"No task transitions found in {run_dir}")
        return

    # Build heatmap dict (top_r only)
    heatmap_data = {
        trans: {layer: v["top_r"] for layer, v in layers.items()}
        for trans, layers in drift_all.items()
    }
    out = Path(run_dir)
    plot_drift_heatmap(heatmap_data, str(out / "fig_drift_heatmap.pdf"))
    print(f"Drift analysis complete for {run_dir}")


def _run_gate1(run_dir: str) -> None:
    """Gate 1 analysis: drift → forgetting regression."""
    drift_all = compute_drift_all(run_dir)
    results = _load_results_json(run_dir)
    forgetting = results.get("forgetting", {})

    top_drifts, rand_drifts, forget_vals = [], [], []
    for (t_from, _t_to), layers in drift_all.items():
        for _layer, d in layers.items():
            key = f"task_{t_from}"
            if key in forgetting:
                top_drifts.append(d["top_r"])
                rand_drifts.append(d["random_r"])
                forget_vals.append(forgetting[key])

    top_drifts = np.array(top_drifts)
    rand_drifts = np.array(rand_drifts)
    forget_vals = np.array(forget_vals)

    top_reg = forgetting_regression(top_drifts, forget_vals)
    rand_reg = forgetting_regression(rand_drifts, forget_vals)

    out = Path(run_dir)
    plot_drift_vs_forgetting(top_drifts, forget_vals, str(out / "fig_regression.pdf"))
    print_gate_results({"top_r2": top_reg["r2"], "random_r2": rand_reg["r2"]})


def _run_gate2(base_dir: str) -> None:
    """Gate 2 analysis: Pareto dominance."""
    base = Path(base_dir)
    methods: dict[str, list] = {}
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        rp = d / "results.json"
        if not rp.exists():
            continue
        with open(rp) as f:
            r = json.load(f)
        method = r.get("method", d.name.split("_")[0])
        methods.setdefault(method, []).append(r)

    if not methods:
        print(f"No results found in {base_dir}")
        return

    plot_pareto(methods, str(base / "fig_pareto.pdf"))

    # Check SFP dominance
    sfp_pts = methods.get("sfp", [])
    if sfp_pts:
        dominated = set()
        for m, pts in methods.items():
            if m == "sfp":
                continue
            if all(
                any(
                    s.get("retention", 0) >= p.get("retention", 0)
                    and s.get("plasticity", 0) >= p.get("plasticity", 0)
                    for s in sfp_pts
                )
                for p in pts
            ):
                dominated.add(m)

        memory = sfp_pts[0].get("memory", "?")
        print_gate_results(
            {"top_r2": 0, "random_r2": 0},
            {"sfp_dominates": dominated, "memory": memory},
        )


def _run_paper(base_dir: str) -> None:
    """Generate all paper figures."""
    _run_gate1(base_dir)
    _run_gate2(base_dir)
    print("Paper figures generated.")


def _run_order_sensitivity(run_dirs: list[str]) -> None:
    """Analyze order sensitivity: mean ± std of metrics across task orderings per method."""
    methods: dict[str, list[dict]] = {}
    for d in run_dirs:
        try:
            r = _load_results_json(d)
        except FileNotFoundError:
            print(f"Warning: no results.json in {d}, skipping")
            continue
        method = r.get("method", Path(d).name.split("_")[0])
        methods.setdefault(method, []).append(r)

    if not methods:
        print("No results found.")
        return

    print("=" * 60)
    print("ORDER SENSITIVITY ANALYSIS")
    print("=" * 60)

    method_scores: dict[str, dict[str, tuple[float, float]]] = {}
    for method, runs in sorted(methods.items()):
        retentions = [r.get("retention", 0.0) for r in runs]
        plasticities = [r.get("plasticity", 0.0) for r in runs]
        scores = [r.get("score", 0.0) for r in runs]

        ret_mean, ret_std = float(np.mean(retentions)), float(np.std(retentions))
        pla_mean, pla_std = float(np.mean(plasticities)), float(np.std(plasticities))
        lb_mean, lb_std = float(np.mean(scores)), float(np.std(scores))

        method_scores[method] = {
            "retention": (ret_mean, ret_std),
            "plasticity": (pla_mean, pla_std),
            "score": (lb_mean, lb_std),
        }

        print(f"\n{method} ({len(runs)} orderings):")
        print(f"  retention   : {ret_mean:.4f} ± {ret_std:.4f}")
        print(f"  plasticity  : {pla_mean:.4f} ± {pla_std:.4f}")
        print(f"  score       : {lb_mean:.4f} ± {lb_std:.4f}")

    # Check ranking stability across score
    if len(method_scores) >= 2:
        print("\n" + "-" * 60)
        ranked = sorted(method_scores.items(), key=lambda x: x[1]["score"][0], reverse=True)
        print("Ranking by score (mean):")
        for i, (m, scores) in enumerate(ranked, 1):
            mean, std = scores["score"]
            print(f"  {i}. {m}: {mean:.4f} ± {std:.4f}")

        # Check if top method's mean - std > second's mean + std (non-overlapping)
        top_m, top_s = ranked[0][1]["score"]
        sec_m, sec_s = ranked[1][1]["score"]
        if top_m - top_s > sec_m + sec_s:
            print(f"\nRanking is STABLE: {ranked[0][0]} > {ranked[1][0]} (non-overlapping ±1σ)")  # noqa: RUF001
        else:
            print(f"\nRanking is UNSTABLE: {ranked[0][0]} vs {ranked[1][0]} have overlapping ±1σ")  # noqa: RUF001


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFP analysis and plotting")
    parser.add_argument("--run", type=str, help="Single run analysis")
    parser.add_argument("--compare", nargs="+", help="Compare multiple runs")
    parser.add_argument("--gate1", type=str, help="Gate 1: drift → forgetting regression")
    parser.add_argument("--gate2", type=str, help="Gate 2: Pareto dominance check")
    parser.add_argument("--paper", type=str, help="Generate all paper figures")
    parser.add_argument(
        "--order-sensitivity", nargs="+", help="Order sensitivity: mean ± std across orderings"
    )
    args = parser.parse_args()

    if args.run:
        _run_analysis(args.run)
    elif args.compare:
        methods: dict[str, list] = {}
        for d in args.compare:
            r = _load_results_json(d)
            method = r.get("method", Path(d).name.split("_")[0])
            methods.setdefault(method, []).append(r)
        plot_pareto(methods)
    elif args.gate1:
        _run_gate1(args.gate1)
    elif args.gate2:
        _run_gate2(args.gate2)
    elif args.paper:
        _run_paper(args.paper)
    elif args.order_sensitivity:
        _run_order_sensitivity(args.order_sensitivity)
    else:
        parser.print_help()
