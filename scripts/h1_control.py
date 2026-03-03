"""
scripts/h1_control.py — H1 control experiment: top-r drift vs total drift vs random drift.

The core falsification test for Hypothesis 1. If R²(top-r drift, forgetting) ≈ R²(‖Δa‖, forgetting),
then H1 is vacuous — forgetting just correlates with total model change, not subspace-specific drift.

Protocol:
  1. Load model + anchor checkpoint
  2. Train naive method on math→code, saving checkpoints every N steps
  3. At each checkpoint: collect activations, compute drifts, evaluate forgetting
  4. Regress forgetting against each drift predictor
  5. Report R² for top-r, random-r, total, bottom-r

H1 survives ONLY if:
  R²(top-r) >> R²(random-r)  AND  R²(top-r) > R²(‖Δa‖)

Usage:
    python scripts/h1_control.py                         # CPU, ~10 min
    python scripts/h1_control.py --steps 500 --every 25  # GPU, more data points
    python scripts/h1_control.py --model Qwen/Qwen2.5-1.5B-Instruct --steps 500
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data
from features import build_pca_basis, collect_activations, get_layer_names
from model import load_model


# ---------------------------------------------------------------------------
# Drift computation
# ---------------------------------------------------------------------------


def compute_projected_drift(
    acts_anchor: torch.Tensor,
    acts_current: torch.Tensor,
    basis: torch.Tensor,
    r: int,
) -> float:
    """||U_r^T (a_current - a_anchor)||^2 averaged over samples."""
    U_r = basis[:, :r]  # [hidden, r]
    diff = acts_current - acts_anchor  # [n, hidden]
    projected_diff = diff @ U_r  # [n, r]
    return float(torch.mean(torch.sum(projected_diff**2, dim=1)).item())


def compute_total_drift(
    acts_anchor: torch.Tensor,
    acts_current: torch.Tensor,
) -> float:
    """||a_current - a_anchor||^2 averaged over samples."""
    diff = acts_current - acts_anchor
    return float(torch.mean(torch.sum(diff**2, dim=1)).item())


def make_random_basis(hidden_dim: int, k: int) -> torch.Tensor:
    """Random orthonormal basis via QR decomposition."""
    M = torch.randn(hidden_dim, k)
    Q, _ = torch.linalg.qr(M)
    return Q  # [hidden_dim, k]


# ---------------------------------------------------------------------------
# Quick math eval (lightweight — no HF dataset download)
# ---------------------------------------------------------------------------

_MATH_QUESTIONS = [
    ("What is 15 + 27?", "42"),
    ("What is 8 * 7?", "56"),
    ("What is 144 / 12?", "12"),
    ("What is 100 - 37?", "63"),
    ("What is 23% of 200?", "46"),
    ("What is 2^10?", "1024"),
    ("What is 9 * 9?", "81"),
    ("What is 256 / 16?", "16"),
    ("What is 50 + 75?", "125"),
]


@torch.no_grad()
def eval_math_quick(model, tokenizer, device: str) -> float:
    """Quick math eval returning accuracy (0-1). No dataset download needed."""
    model.eval()
    correct = 0
    for question, gold in _MATH_QUESTIONS:
        inputs = tokenizer(question, return_tensors="pt").to(device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        if gold in text:
            correct += 1
    return correct / len(_MATH_QUESTIONS)


@torch.no_grad()
def eval_math_perplexity(
    model, tokenizer, dataset: list[dict], device: str, max_samples: int = 50
) -> float:
    """Average cross-entropy loss on math examples (perplexity-based forgetting proxy).

    Works even when exact-match accuracy is 0% (e.g. SmolLM2-135M), because
    the loss is continuous and captures how well the model *models* the math
    distribution, not whether it can generate correct answers.
    """
    model.eval()
    samples = dataset[:max_samples]
    total_loss = 0.0
    n = 0
    for i in range(0, len(samples), 8):
        chunk = samples[i : i + 8]
        batch = data.get_batch(chunk, batch_size=len(chunk), tokenizer=tokenizer, device=device)
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        total_loss += out.loss.item() * len(chunk)
        n += len(chunk)
    return total_loss / n if n > 0 else 0.0


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


def ols_r2(x: np.ndarray, y: np.ndarray) -> float:
    """Simple OLS R² between 1D arrays."""
    if len(x) < 3:
        return float("nan")
    x = x.reshape(-1, 1)
    from sklearn.linear_model import LinearRegression

    reg = LinearRegression().fit(x, y)
    return float(reg.score(x, y))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_h1_results(records: list[dict], out_dir: str) -> None:
    """Generate the H1 control scatter plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.array([r["step"] for r in records])

    # Use perplexity-based forgetting as primary metric; fall back to exact-match
    forgetting_ppl = np.array([r.get("forgetting_ppl", 0.0) for r in records])
    forgetting_acc = np.array([r["forgetting"] for r in records])
    if np.any(forgetting_ppl > 0):
        forgetting = forgetting_ppl
        forget_label = "Forgetting (Δ loss)"
    else:
        forgetting = forgetting_acc
        forget_label = "Forgetting (acc drop)"

    # Aggregate drift across layers (mean)
    layers = [k for k in records[0] if k.startswith("drift_")]
    drift_types = {}
    for dtype in ["top_r", "random_r", "bottom_r", "total"]:
        key = f"drift_{dtype}"
        if key in records[0]:
            drift_types[dtype] = np.array([r[key] for r in records])
        else:
            # Average across layers
            layer_keys = [k for k in records[0] if k.startswith(f"drift_{dtype}_")]
            if layer_keys:
                vals = np.mean([np.array([r[k] for r in records]) for k in layer_keys], axis=0)
                drift_types[dtype] = vals

    # Compute R² for each
    r2_results = {}
    for dtype, drift_vals in drift_types.items():
        r2 = ols_r2(drift_vals, forgetting)
        r2_results[dtype] = r2

    # Print summary
    print("\n" + "=" * 60)
    print("  H1 CONTROL EXPERIMENT RESULTS")
    print("=" * 60)
    for dtype, r2 in sorted(r2_results.items(), key=lambda x: -x[1]):
        print(f"  R²({dtype:>10s} drift, forgetting) = {r2:.4f}")
    print("=" * 60)

    # Gate check
    top_r2 = r2_results.get("top_r", 0)
    rand_r2 = r2_results.get("random_r", 0)
    total_r2 = r2_results.get("total", 0)

    pass_vs_random = top_r2 > rand_r2 * 2 if rand_r2 > 0 else top_r2 > 0.1
    pass_vs_total = top_r2 > total_r2
    pass_threshold = top_r2 >= 0.5

    print(f"\n  Gate checks:")
    print(f"    R²(top-r) >> R²(random-r)?  {'✓' if pass_vs_random else '✗'}  ({top_r2:.3f} vs {rand_r2:.3f})")
    print(f"    R²(top-r) > R²(total)?      {'✓' if pass_vs_total else '✗'}  ({top_r2:.3f} vs {total_r2:.3f})")
    print(f"    R²(top-r) ≥ 0.5?            {'✓' if pass_threshold else '✗'}  ({top_r2:.3f})")
    overall = pass_vs_random and pass_vs_total and pass_threshold
    print(f"\n  GATE 1: {'✓ PASS — H1 survives' if overall else '✗ FAIL — H1 is in trouble'}")
    print()

    # Scatter plots
    fig, axes = plt.subplots(1, len(drift_types), figsize=(4 * len(drift_types), 3.5))
    if len(drift_types) == 1:
        axes = [axes]

    for ax, (dtype, drift_vals) in zip(axes, drift_types.items()):
        r2 = r2_results[dtype]
        ax.scatter(drift_vals, forgetting, s=20, alpha=0.7)

        # Regression line
        if len(drift_vals) >= 2:
            from sklearn.linear_model import LinearRegression
            reg = LinearRegression().fit(drift_vals.reshape(-1, 1), forgetting)
            x_line = np.linspace(drift_vals.min(), drift_vals.max(), 50)
            y_line = reg.predict(x_line.reshape(-1, 1))
            ax.plot(x_line, y_line, "k-", linewidth=1.5)

        ax.set_xlabel(f"{dtype} drift")
        ax.set_ylabel(forget_label)
        ax.set_title(f"R² = {r2:.3f}")
        ax.annotate(
            f"R² = {r2:.3f}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "wheat", "alpha": 0.5},
        )

    fig.suptitle("H1 Control: Drift Predictors of Forgetting", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_h1_control.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig_h1_control.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Plots saved to {out_dir}/fig_h1_control.{{pdf,png}}")

    # Forgetting trajectory
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    ax2.plot(steps, forgetting, "o-", markersize=4)
    ax2.set_xlabel("Training step")
    ax2.set_ylabel(forget_label)
    ax2.set_title("Math forgetting during code training")
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "fig_forgetting_trajectory.pdf"), bbox_inches="tight")
    fig2.savefig(os.path.join(out_dir, "fig_forgetting_trajectory.png"), bbox_inches="tight", dpi=150)
    plt.close(fig2)


# ---------------------------------------------------------------------------
# R-sweep diagnostic
# ---------------------------------------------------------------------------


def sweep_r_diagnostic(
    records: list[dict],
    anchor_acts: dict[str, torch.Tensor],
    final_acts: dict[str, torch.Tensor],
    pca_bases: dict[str, torch.Tensor],
    pca_variances: dict[str, torch.Tensor],
    layer_names: list[str],
    out_dir: str,
    pca_k: int,
) -> None:
    """Sweep over r values to diagnose whether PCA basis or dimensionality is the issue.

    For each r in [1, 2, 4, 8, 16, 32, 64, min(pca_k, hidden_dim)]:
      - Compute R²(top-r PCA drift, forgetting) using all checkpoint records
      - Compute cumulative variance explained by top-r PCA dims

    If variance explained is high but R² is low → PCA is the wrong basis.
    If both track together → forgetting IS in high-variance dirs, r was too small.

    Per-checkpoint top-r drift is approximated by scaling the stored per-layer
    total drift by the fraction captured in top-r PCA dims at the final checkpoint.
    This assumes the spectral profile of drift is roughly stable across checkpoints.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hidden_dim = pca_bases[layer_names[0]].shape[0]
    max_r = min(pca_k, hidden_dim)
    r_values = sorted(set(
        r for r in [1, 2, 4, 8, 16, 32, 64, max_r] if r <= max_r
    ))

    # Get forgetting array from records
    forgetting_ppl = np.array([r.get("forgetting_ppl", 0.0) for r in records])
    forgetting_acc = np.array([r["forgetting"] for r in records])
    forgetting = forgetting_ppl if np.any(forgetting_ppl > 0) else forgetting_acc

    # Total variance per layer for normalization
    total_variance = {
        ln: float(pca_variances[ln].sum().item()) for ln in layer_names
    }

    # Precompute per-layer drift fraction for each r at the final checkpoint
    # frac[ln][r] = (top-r PCA drift) / (total drift) at final checkpoint
    drift_frac = {}
    for ln in layer_names:
        a_anchor = anchor_acts[ln].float()
        a_final = final_acts[ln].float()
        diff = a_final - a_anchor
        total = float(torch.mean(torch.sum(diff**2, dim=1)).item())
        U = pca_bases[ln]
        fracs = {}
        for r in r_values:
            proj = diff @ U[:, :r]
            top_r = float(torch.mean(torch.sum(proj**2, dim=1)).item())
            fracs[r] = top_r / total if total > 0 else 0.0
        drift_frac[ln] = fracs

    r2_per_r = []
    var_explained_per_r = []

    for r in r_values:
        # Variance explained: average across layers
        var_exp = float(np.mean([
            pca_variances[ln].cpu().numpy()[:r].sum() / total_variance[ln] * 100.0
            for ln in layer_names
        ]))
        var_explained_per_r.append(var_exp)

        # Approximate top-r drift per record by scaling stored total drift
        drift_at_r = []
        for rec in records:
            layer_drifts = []
            for ln in layer_names:
                total_ln = rec.get(f"drift_total_{ln}", 0.0)
                layer_drifts.append(total_ln * drift_frac[ln][r])
            drift_at_r.append(float(np.mean(layer_drifts)))
        drift_at_r = np.array(drift_at_r)

        r2_per_r.append(ols_r2(drift_at_r, forgetting))

    # Print results
    print("\n" + "=" * 60)
    print("  R-SWEEP DIAGNOSTIC")
    print("=" * 60)
    print(f"  {'r':>5s}  {'Var Explained (%)':>18s}  {'Forgetting R²':>14s}")
    print("  " + "-" * 42)
    for r, ve, r2 in zip(r_values, var_explained_per_r, r2_per_r):
        print(f"  {r:5d}  {ve:18.1f}  {r2:14.4f}")
    print("=" * 60)

    # Interpretation
    mid_idx = len(r_values) // 2
    mid_var = var_explained_per_r[mid_idx]
    mid_r2 = r2_per_r[mid_idx]
    if mid_var > 80 and mid_r2 < 0.3:
        print("  → Variance is high but R² is low: PCA may be the wrong basis.")
        print("    High-variance directions ≠ forgetting-relevant directions.")
    elif mid_r2 > 0.5 and mid_var > 50:
        print("  → Both track together: forgetting IS in high-variance directions.")
        print("    Increasing r should improve prediction.")
    else:
        print("  → Mixed signal: check the plot for detailed structure.")
    print()

    # Dual-axis plot
    fig, ax1 = plt.subplots(figsize=(6, 4))

    color_var = "#2196F3"
    color_r2 = "#FF5722"

    ax1.set_xlabel("r (number of PCA components)")
    ax1.set_ylabel("Variance Explained (%)", color=color_var)
    ax1.plot(r_values, var_explained_per_r, "o-", color=color_var, label="Var. Explained (%)")
    ax1.tick_params(axis="y", labelcolor=color_var)
    ax1.set_ylim(0, 105)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Forgetting R²", color=color_r2)
    ax2.plot(r_values, r2_per_r, "s-", color=color_r2, label="Forgetting R²")
    ax2.tick_params(axis="y", labelcolor=color_r2)
    ax2.set_ylim(-0.05, 1.05)

    ax1.set_xscale("log", base=2)
    ax1.set_xticks(r_values)
    ax1.set_xticklabels([str(r) for r in r_values])

    fig.suptitle("R-Sweep: Variance Explained vs Forgetting R²", y=1.02)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_r_sweep.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig_r_sweep.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  R-sweep plot saved to {out_dir}/fig_r_sweep.{{pdf,png}}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="H1 control: top-r drift vs total drift")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--steps", type=int, default=200, help="Total training steps")
    parser.add_argument("--every", type=int, default=20, help="Checkpoint every N steps")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--pca_k", type=int, default=64, help="PCA components")
    parser.add_argument("--preserve_r", type=int, default=32, help="Top-r dims to track")
    parser.add_argument("--memory", type=int, default=64, help="Memory buffer size for activations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="out/h1_control")
    parser.add_argument("--old_task", default="math", help="Task to measure forgetting on")
    parser.add_argument("--new_task", default="code", help="Task to train on")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sweep_r", action="store_true", help="Run r-sweep diagnostic after training")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("=" * 60)
    print("  H1 Control Experiment")
    print("=" * 60)
    print(f"  model     : {args.model}")
    print(f"  old_task  : {args.old_task} (measure forgetting)")
    print(f"  new_task  : {args.new_task} (train on)")
    print(f"  steps     : {args.steps} (checkpoint every {args.every})")
    print(f"  pca_k     : {args.pca_k}, preserve_r: {args.preserve_r}")
    print(f"  device    : {device}")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model(
        name=args.model,
        lora_rank=args.lora_rank,
        device=device,
    )

    # Discover probe layers
    layer_names = get_layer_names(model)
    print(f"Probe layers: {layer_names}")

    # Load old-task memory for activation collection
    old_data = data.load_task(args.old_task, split="train", max_samples=500)
    memory_set = random.Random(args.seed).sample(old_data, min(args.memory, len(old_data)))

    # Baseline: eval + collect anchor activations
    print("\nCollecting anchor activations...")
    model.eval()
    baseline_acc = eval_math_quick(model, tokenizer, device)
    baseline_ppl = eval_math_perplexity(model, tokenizer, memory_set, device)
    print(f"  Baseline math accuracy: {baseline_acc:.2%}")
    print(f"  Baseline math loss:     {baseline_ppl:.4f}")

    anchor_acts = collect_activations(model, memory_set, tokenizer, layer_names)
    # anchor_acts: {layer_name: [n, hidden]}

    # Build PCA basis from anchor activations
    pca_bases = {}
    pca_variances = {}
    random_bases = {}
    for layer_name in layer_names:
        a = anchor_acts[layer_name].float()
        u, variance = build_pca_basis(a, k=args.pca_k)
        pca_bases[layer_name] = u  # [hidden, k]
        pca_variances[layer_name] = variance  # [k]
        random_bases[layer_name] = make_random_basis(a.shape[1], args.pca_k)

    # Load new-task training data
    new_data = data.load_task(args.new_task, split="train", max_samples=1000)
    print(f"Loaded {len(new_data)} training examples for '{args.new_task}'")

    # Training loop with checkpointing
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    records = []
    t0 = time.time()

    # Record step 0
    records.append({
        "step": 0,
        "forgetting": 0.0,
        "forgetting_ppl": 0.0,
        "math_acc": baseline_acc,
        "math_ppl": baseline_ppl,
        **{f"drift_top_r_{ln}": 0.0 for ln in layer_names},
        **{f"drift_random_r_{ln}": 0.0 for ln in layer_names},
        **{f"drift_bottom_r_{ln}": 0.0 for ln in layer_names},
        **{f"drift_total_{ln}": 0.0 for ln in layer_names},
        "drift_top_r": 0.0,
        "drift_random_r": 0.0,
        "drift_bottom_r": 0.0,
        "drift_total": 0.0,
    })

    print(f"\nTraining {args.new_task} for {args.steps} steps...")
    model.train()
    for step in range(1, args.steps + 1):
        batch = data.get_batch(
            new_data, batch_size=args.batch_size, tokenizer=tokenizer, device=device
        )
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 5 == 0:
            print(f"  step {step}/{args.steps} | loss {out.loss.item():.4f}", flush=True)

        # Checkpoint: collect activations + eval
        if step % args.every == 0 or step == args.steps:
            model.eval()

            # Evaluate forgetting
            # Skip expensive generation-based eval if baseline acc is 0
            if baseline_acc > 0:
                current_acc = eval_math_quick(model, tokenizer, device)
            else:
                current_acc = 0.0
            forgetting = max(0.0, baseline_acc - current_acc)
            current_ppl = eval_math_perplexity(model, tokenizer, memory_set, device)
            forgetting_ppl = max(0.0, current_ppl - baseline_ppl)

            # Collect current activations
            current_acts = collect_activations(model, memory_set, tokenizer, layer_names)

            # Compute drifts per layer
            record = {
                "step": step,
                "forgetting": forgetting,
                "forgetting_ppl": forgetting_ppl,
                "math_acc": current_acc,
                "math_ppl": current_ppl,
            }

            layer_drifts = {"top_r": [], "random_r": [], "bottom_r": [], "total": []}
            for ln in layer_names:
                a_anchor = anchor_acts[ln].float()
                a_current = current_acts[ln].float()

                # Top-r drift (PCA basis, first r components)
                r = min(args.preserve_r, args.pca_k)
                top_r_drift = compute_projected_drift(a_anchor, a_current, pca_bases[ln], r)

                # Random-r drift
                random_r_drift = compute_projected_drift(a_anchor, a_current, random_bases[ln], r)

                # Bottom-r drift (PCA basis, last r components)
                k = pca_bases[ln].shape[1]
                bottom_basis = pca_bases[ln][:, max(0, k - r):]
                bottom_r_drift = compute_projected_drift(
                    a_anchor, a_current, bottom_basis, bottom_basis.shape[1]
                )

                # Total drift
                total_drift = compute_total_drift(a_anchor, a_current)

                record[f"drift_top_r_{ln}"] = top_r_drift
                record[f"drift_random_r_{ln}"] = random_r_drift
                record[f"drift_bottom_r_{ln}"] = bottom_r_drift
                record[f"drift_total_{ln}"] = total_drift

                layer_drifts["top_r"].append(top_r_drift)
                layer_drifts["random_r"].append(random_r_drift)
                layer_drifts["bottom_r"].append(bottom_r_drift)
                layer_drifts["total"].append(total_drift)

            # Mean across layers
            for dtype in layer_drifts:
                record[f"drift_{dtype}"] = float(np.mean(layer_drifts[dtype]))

            records.append(record)
            elapsed = time.time() - t0
            print(
                f"  [ckpt step={step}] acc={current_acc:.2%} "
                f"ppl={current_ppl:.4f} "
                f"forget={forgetting:.3f} forget_ppl={forgetting_ppl:.4f} "
                f"drift_top={record['drift_top_r']:.4f} "
                f"drift_rand={record['drift_random_r']:.4f} "
                f"drift_total={record['drift_total']:.4f} "
                f"({elapsed:.0f}s)",
                flush=True,
            )

            final_acts = current_acts
            model.train()

    # Save raw data
    results_path = os.path.join(args.out_dir, "h1_control.json")
    with open(results_path, "w") as f:
        json.dump(
            {
                "config": vars(args),
                "baseline_accuracy": baseline_acc,
                "baseline_perplexity": baseline_ppl,
                "layer_names": layer_names,
                "records": records,
            },
            f,
            indent=2,
        )
    print(f"\nRaw data saved to {results_path}")

    # Analysis + plots
    if len(records) >= 3:
        plot_h1_results(records, args.out_dir)
    else:
        print("Too few data points for regression (need ≥ 3)")

    # R-sweep diagnostic
    if args.sweep_r and len(records) >= 3:
        sweep_r_diagnostic(
            records, anchor_acts, final_acts,
            pca_bases, pca_variances, layer_names,
            args.out_dir, args.pca_k,
        )

    print(f"\nTotal time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
