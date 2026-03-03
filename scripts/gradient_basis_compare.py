"""
scripts/gradient_basis_compare.py — Compare PCA vs gradient-informed basis for H1.

Runs the same h1_control protocol but tracks drift in both PCA and gradient
subspaces, reporting R² for each. If gradient basis yields higher R², it
strengthens the case that task-relevant directions matter more than variance.

Usage:
    python scripts/gradient_basis_compare.py                          # CPU, ~15 min
    python scripts/gradient_basis_compare.py --model Qwen/Qwen2.5-1.5B-Instruct --steps 500 --every 25
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data
from features import build_gradient_basis, build_pca_basis, collect_activations, get_layer_names
from model import load_model


# ---------------------------------------------------------------------------
# Drift computation (reused from h1_control)
# ---------------------------------------------------------------------------


def compute_projected_drift(
    acts_anchor: torch.Tensor,
    acts_current: torch.Tensor,
    basis: torch.Tensor,
    r: int,
) -> float:
    """||U_r^T (a_current - a_anchor)||^2 averaged over samples."""
    U_r = basis[:, :r]
    diff = (acts_current - acts_anchor).float()
    projected_diff = diff @ U_r.float()
    return float(torch.mean(torch.sum(projected_diff**2, dim=1)).item())


def compute_total_drift(
    acts_anchor: torch.Tensor,
    acts_current: torch.Tensor,
) -> float:
    diff = (acts_current - acts_anchor).float()
    return float(torch.mean(torch.sum(diff**2, dim=1)).item())


def make_random_basis(hidden_dim: int, k: int) -> torch.Tensor:
    M = torch.randn(hidden_dim, k)
    Q, _ = torch.linalg.qr(M)
    return Q


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------


@torch.no_grad()
def eval_math_perplexity(
    model, tokenizer, dataset: list[dict], device: str, max_samples: int = 50,
) -> float:
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


def ols_r2(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    x = x.reshape(-1, 1)
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(x, y)
    return float(reg.score(x, y))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="PCA vs gradient basis comparison for H1")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--every", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--pca_k", type=int, default=64)
    parser.add_argument("--preserve_r", type=int, default=32)
    parser.add_argument("--memory", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="out/grad_basis_compare")
    parser.add_argument("--old_task", default="math")
    parser.add_argument("--new_task", default="code")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("=" * 60)
    print("  Gradient Basis vs PCA Comparison")
    print("=" * 60)
    print(f"  model     : {args.model}")
    print(f"  old_task  : {args.old_task}, new_task: {args.new_task}")
    print(f"  steps     : {args.steps} (every {args.every})")
    print(f"  pca_k     : {args.pca_k}, preserve_r: {args.preserve_r}")
    print(f"  device    : {device}")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model(name=args.model, lora_rank=args.lora_rank, device=device)

    layer_names = get_layer_names(model)
    print(f"Probe layers: {layer_names}")

    # Load data
    old_data = data.load_task(args.old_task, split="train", max_samples=500)
    memory_set = random.Random(args.seed).sample(old_data, min(args.memory, len(old_data)))
    new_data = data.load_task(args.new_task, split="train", max_samples=1000)

    # Baseline
    print("\nBaseline evaluation...")
    baseline_ppl = eval_math_perplexity(model, tokenizer, memory_set, device)
    print(f"  Baseline math loss: {baseline_ppl:.4f}")

    # Collect anchor activations
    print("Collecting anchor activations...")
    anchor_acts = collect_activations(model, memory_set, tokenizer, layer_names)

    # Build PCA basis
    print("Building PCA basis...")
    pca_bases = {}
    for ln in layer_names:
        a = anchor_acts[ln].float()
        u, _var = build_pca_basis(a, k=args.pca_k)
        pca_bases[ln] = u

    # Build gradient basis
    print("Building gradient basis...")
    grad_bases = build_gradient_basis(
        model, tokenizer, memory_set, layer_names, k=args.pca_k, device=device,
    )

    # Random basis (control)
    random_bases = {}
    for ln in layer_names:
        hidden_dim = anchor_acts[ln].shape[1]
        random_bases[ln] = make_random_basis(hidden_dim, args.pca_k)

    # Training loop
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr,
    )

    records = []
    r = min(args.preserve_r, args.pca_k)
    t0 = time.time()

    # Step 0
    records.append({
        "step": 0, "forgetting_ppl": 0.0, "math_ppl": baseline_ppl,
        "drift_pca_top": 0.0, "drift_grad_top": 0.0,
        "drift_random": 0.0, "drift_total": 0.0,
    })

    print(f"\nTraining {args.new_task} for {args.steps} steps...")
    model.train()
    for step in range(1, args.steps + 1):
        batch = data.get_batch(new_data, batch_size=args.batch_size, tokenizer=tokenizer, device=device)
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 5 == 0 or step == 1:
            print(f"  step {step}/{args.steps} | loss {out.loss.item():.4f}", flush=True)

        if step % args.every == 0 or step == args.steps:
            model.eval()

            current_ppl = eval_math_perplexity(model, tokenizer, memory_set, device)
            forgetting_ppl = max(0.0, current_ppl - baseline_ppl)

            current_acts = collect_activations(model, memory_set, tokenizer, layer_names)

            # Compute drifts (mean across layers)
            drifts = {"pca_top": [], "grad_top": [], "random": [], "total": []}
            for ln in layer_names:
                a_anc = anchor_acts[ln].float()
                a_cur = current_acts[ln].float()

                drifts["pca_top"].append(compute_projected_drift(a_anc, a_cur, pca_bases[ln], r))
                drifts["grad_top"].append(compute_projected_drift(a_anc, a_cur, grad_bases[ln], r))
                drifts["random"].append(compute_projected_drift(a_anc, a_cur, random_bases[ln], r))
                drifts["total"].append(compute_total_drift(a_anc, a_cur))

            record = {
                "step": step,
                "forgetting_ppl": forgetting_ppl,
                "math_ppl": current_ppl,
                "drift_pca_top": float(np.mean(drifts["pca_top"])),
                "drift_grad_top": float(np.mean(drifts["grad_top"])),
                "drift_random": float(np.mean(drifts["random"])),
                "drift_total": float(np.mean(drifts["total"])),
            }
            records.append(record)

            elapsed = time.time() - t0
            print(
                f"  [ckpt {step}] ppl={current_ppl:.4f} Δppl={forgetting_ppl:.4f} "
                f"pca={record['drift_pca_top']:.4f} grad={record['drift_grad_top']:.4f} "
                f"rand={record['drift_random']:.4f} total={record['drift_total']:.4f} "
                f"({elapsed:.0f}s)",
                flush=True,
            )
            model.train()

    # Analysis
    forgetting = np.array([r["forgetting_ppl"] for r in records])
    drift_pca = np.array([r["drift_pca_top"] for r in records])
    drift_grad = np.array([r["drift_grad_top"] for r in records])
    drift_rand = np.array([r["drift_random"] for r in records])
    drift_total = np.array([r["drift_total"] for r in records])

    r2_pca = ols_r2(drift_pca, forgetting)
    r2_grad = ols_r2(drift_grad, forgetting)
    r2_rand = ols_r2(drift_rand, forgetting)
    r2_total = ols_r2(drift_total, forgetting)

    print("\n" + "=" * 60)
    print("  GRADIENT BASIS vs PCA COMPARISON")
    print("=" * 60)
    print(f"  R²(PCA top-{r})       = {r2_pca:.4f}")
    print(f"  R²(gradient top-{r})  = {r2_grad:.4f}")
    print(f"  R²(random-{r})        = {r2_rand:.4f}")
    print(f"  R²(total drift)       = {r2_total:.4f}")
    print()
    winner = "gradient" if r2_grad > r2_pca else "PCA"
    margin = abs(r2_grad - r2_pca)
    print(f"  Winner: {winner} (margin: {margin:.4f})")
    print(f"  Gradient > PCA?  {'✓' if r2_grad > r2_pca else '✗'}")
    print(f"  Gradient > random? {'✓' if r2_grad > r2_rand else '✗'}")
    print(f"  PCA > random?      {'✓' if r2_pca > r2_rand else '✗'}")
    print("=" * 60)

    # Save
    results = {
        "config": vars(args),
        "baseline_ppl": baseline_ppl,
        "layer_names": layer_names,
        "records": records,
        "r2": {
            "pca_top": r2_pca,
            "grad_top": r2_grad,
            "random": r2_rand,
            "total": r2_total,
        },
        "winner": winner,
        "margin": margin,
    }
    results_path = os.path.join(args.out_dir, "grad_basis_compare.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Plot
    plot_comparison(records, results["r2"], r, args.out_dir)
    print(f"Total time: {time.time() - t0:.0f}s")


def plot_comparison(records: list[dict], r2: dict, r: int, out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    forgetting = np.array([rec["forgetting_ppl"] for rec in records])

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    configs = [
        ("drift_pca_top", f"PCA top-{r}", r2["pca_top"], "#E53935"),
        ("drift_grad_top", f"Gradient top-{r}", r2["grad_top"], "#1E88E5"),
        ("drift_random", f"Random-{r}", r2["random"], "#43A047"),
        ("drift_total", "Total drift", r2["total"], "#FDD835"),
    ]

    for ax, (key, label, r2_val, color) in zip(axes, configs):
        vals = np.array([rec[key] for rec in records])
        ax.scatter(vals, forgetting, s=20, alpha=0.7, color=color)

        if len(vals) >= 2:
            from sklearn.linear_model import LinearRegression
            reg = LinearRegression().fit(vals.reshape(-1, 1), forgetting)
            x_line = np.linspace(vals.min(), vals.max(), 50)
            ax.plot(x_line, reg.predict(x_line.reshape(-1, 1)), "k-", linewidth=1.5)

        ax.set_xlabel(f"{label} drift")
        ax.set_ylabel("Forgetting (Δ loss)")
        ax.set_title(f"{label}\nR² = {r2_val:.3f}")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_grad_vs_pca.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig_grad_vs_pca.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Plot saved to {out_dir}/fig_grad_vs_pca.{{pdf,png}}")


if __name__ == "__main__":
    main()
