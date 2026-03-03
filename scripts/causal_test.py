"""
scripts/causal_test.py — Causal intervention test for H1.

H1 claims top-r PCA dimensions are causally important for old-task capability.
Correlation (R²) isn't causation. This script directly perturbs activations at
inference time and measures the effect on old-task performance.

Protocol:
  1. Load model + old-task (math) memory set
  2. Build PCA basis + rank importance (ablation-based)
  3. At inference time on math eval, register forward hooks that add noise
     to specific activation subspaces:
       (a) top-32 important PCA dims
       (b) random-32 PCA dims
       (c) bottom-32 PCA dims
       (d) no intervention (baseline)
  4. Evaluate math perplexity for each intervention × noise scale
  5. Plot bar chart + save JSON results

Expected result: if H1 is correct, perturbing top-32 dims should cause much more
math degradation than perturbing random-32 or bottom-32 dims.

Usage:
    python scripts/causal_test.py                                  # CPU, ~15 min
    python scripts/causal_test.py --model Qwen/Qwen2.5-1.5B-Instruct --memory 128
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
from features import build_pca_basis, collect_activations, get_layer_names, rank_importance
from model import load_model


# ---------------------------------------------------------------------------
# Noise injection hooks
# ---------------------------------------------------------------------------


def _make_noise_hook(
    basis_cols: torch.Tensor,
    scale: float,
    generator: torch.Generator,
):
    """Create a forward hook that adds Gaussian noise in a given subspace.

    The hook computes:
        act_modified = act + U_r @ z,  where z ~ N(0, scale² I_r)

    Args:
        basis_cols: [hidden_dim, r] orthonormal columns spanning the subspace.
        scale: Standard deviation of noise per dimension.
        generator: Torch RNG for reproducibility.
    """
    def hook_fn(_module, _input, output):
        # Detect output format
        is_tensor = isinstance(output, torch.Tensor)
        is_tuple = isinstance(output, tuple)
        if is_tensor:
            h = output
        elif is_tuple:
            h = output[0]
        elif hasattr(output, "last_hidden_state"):
            h = output.last_hidden_state
        else:
            return output

        # basis_cols is CPU; move lazily to device + dtype
        U = basis_cols.to(h.device, dtype=h.dtype)  # [hidden, r]
        r_dim = U.shape[1]
        # Generate noise on CPU then cast (generator may be CPU-only)
        z = torch.randn(
            *h.shape[:-1], r_dim,
        ).to(device=h.device, dtype=h.dtype) * scale  # [..., r]
        noise = z @ U.T  # [..., hidden]
        h_new = h + noise

        if is_tensor:
            return h_new
        if is_tuple:
            return (h_new, *output[1:])
        output.last_hidden_state = h_new
        return output

    return hook_fn


# ---------------------------------------------------------------------------
# Perplexity eval (adapted from h1_control.py)
# ---------------------------------------------------------------------------


@torch.no_grad()
def eval_math_perplexity(
    model, tokenizer, dataset: list[dict], device: str, max_samples: int = 50,
) -> float:
    """Average cross-entropy loss on math examples."""
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
# Subspace construction helpers
# ---------------------------------------------------------------------------


def make_random_basis(hidden_dim: int, k: int) -> torch.Tensor:
    """Random orthonormal basis via QR decomposition."""
    M = torch.randn(hidden_dim, k)
    Q, _ = torch.linalg.qr(M)
    return Q  # [hidden_dim, k]


def select_dims(basis: torch.Tensor, importance: torch.Tensor, mode: str, r: int) -> torch.Tensor:
    """Select r columns from basis according to mode.

    Args:
        mode: "top" (most important), "bottom" (least important), "random".

    Returns:
        [hidden_dim, r] tensor.
    """
    k = basis.shape[1]
    r = min(r, k)
    if mode == "top":
        indices = torch.argsort(importance, descending=True)[:r]
    elif mode == "bottom":
        indices = torch.argsort(importance, descending=False)[:r]
    elif mode == "random":
        indices = torch.randperm(k)[:r]
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return basis[:, indices]


def eigenvalue_scales(variance: torch.Tensor) -> torch.Tensor:
    """Typical activation scale per PCA dim = sqrt(eigenvalue)."""
    return torch.sqrt(variance.clamp(min=1e-12))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_results(results: dict, out_dir: str) -> None:
    """Bar chart: perplexity increase per intervention type × noise scale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    interventions = results["interventions"]
    noise_scales = results["noise_scales"]
    baseline_ppl = results["baseline_ppl"]

    # Group by intervention type
    types = ["top", "bottom", "random"]
    type_labels = {"top": "Top-r", "bottom": "Bottom-r", "random": "Random-r"}
    colors = {"top": "#E53935", "bottom": "#43A047", "random": "#1E88E5"}

    n_scales = len(noise_scales)
    n_types = len(types)
    bar_width = 0.22
    x = np.arange(n_scales)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for i, t in enumerate(types):
        ppl_increases = []
        for s in noise_scales:
            key = f"{t}_{s}"
            ppl = interventions.get(key, {}).get("ppl", baseline_ppl)
            ppl_increases.append(ppl - baseline_ppl)
        ax.bar(
            x + i * bar_width,
            ppl_increases,
            bar_width,
            label=type_labels[t],
            color=colors[t],
            alpha=0.85,
        )

    ax.set_xlabel("Noise scale (× σ)")
    ax.set_ylabel("Perplexity increase (Δ loss)")
    ax.set_title("Causal Intervention: Perplexity Increase by Subspace")
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels([f"{s}σ" for s in noise_scales])
    ax.legend()
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_causal_test.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig_causal_test.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Plot saved to {out_dir}/fig_causal_test.{{pdf,png}}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="H1 causal intervention test")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--memory", type=int, default=128, help="Memory buffer size")
    parser.add_argument("--pca_k", type=int, default=128, help="PCA components")
    parser.add_argument("--preserve_r", type=int, default=32, help="Number of dims to perturb")
    parser.add_argument("--noise_scales", type=float, nargs="+", default=[0.5, 1.0, 2.0, 5.0])
    parser.add_argument("--eval_samples", type=int, default=50, help="Samples for perplexity eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="out/causal_test")
    parser.add_argument("--old_task", default="math", help="Task to measure forgetting on")
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
    print("  H1 Causal Intervention Test")
    print("=" * 60)
    print(f"  model       : {args.model}")
    print(f"  old_task    : {args.old_task}")
    print(f"  memory      : {args.memory}")
    print(f"  pca_k       : {args.pca_k}, preserve_r: {args.preserve_r}")
    print(f"  noise_scales: {args.noise_scales}")
    print(f"  device      : {device}")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model(
        name=args.model,
        lora_rank=16,
        device=device,
    )

    # Discover probe layers
    layer_names = get_layer_names(model)
    print(f"Probe layers: {layer_names}")

    # Load old-task memory set
    old_data = data.load_task(args.old_task, split="train", max_samples=500)
    memory_set = random.Random(args.seed).sample(old_data, min(args.memory, len(old_data)))
    eval_set = memory_set[:args.eval_samples]

    # Baseline perplexity (no intervention)
    print("\nEvaluating baseline perplexity...")
    baseline_ppl = eval_math_perplexity(model, tokenizer, eval_set, device, args.eval_samples)
    print(f"  Baseline math loss: {baseline_ppl:.4f}")

    # Collect activations + build PCA basis
    print("\nCollecting activations + building PCA basis...")
    acts = collect_activations(model, memory_set, tokenizer, layer_names)

    pca_bases = {}
    pca_variances = {}
    for ln in layer_names:
        a = acts[ln].float()
        u, variance = build_pca_basis(a, k=args.pca_k)
        pca_bases[ln] = u  # [hidden, k]
        pca_variances[ln] = variance  # [k]

    # Rank importance via ablation (per-dim zeroing)
    print("\nRanking dimension importance (ablation)...")

    def _neg_ppl(m, tok, mem):
        return -eval_math_perplexity(m, tok, mem, device, max_samples=min(20, len(mem)))

    importance_per_layer = {}
    for ln in layer_names:
        print(f"  Layer {ln}...")
        imp = rank_importance(
            model, tokenizer, pca_bases[ln], memory_set[:32], _neg_ppl, ln,
        )
        importance_per_layer[ln] = imp
        top_idx = torch.argsort(imp, descending=True)[:5]
        print(f"    top-5 importance: {imp[top_idx].tolist()}")

    # Build subspace selections per layer
    r = min(args.preserve_r, args.pca_k)
    subspaces = {}  # {(layer_name, mode): [hidden, r]}
    for ln in layer_names:
        for mode in ["top", "bottom", "random"]:
            subspaces[(ln, mode)] = select_dims(
                pca_bases[ln], importance_per_layer[ln], mode, r,
            )

    # Compute noise scale reference: sqrt(eigenvalue) per dim, averaged
    dim_scales = {}  # {(layer_name, mode): mean_scale}
    for ln in layer_names:
        ev = pca_variances[ln]
        scales = eigenvalue_scales(ev)
        for mode in ["top", "bottom", "random"]:
            if mode == "top":
                idx = torch.argsort(importance_per_layer[ln], descending=True)[:r]
            elif mode == "bottom":
                idx = torch.argsort(importance_per_layer[ln], descending=False)[:r]
            else:
                idx = torch.randperm(len(ev))[:r]
            # Mean eigenvalue scale across selected dims
            dim_scales[(ln, mode)] = float(scales[idx].mean().item())

    # Run interventions
    print(f"\nRunning interventions ({len(args.noise_scales)} scales × 3 types)...")
    t0 = time.time()

    intervention_results = {}

    for scale_mult in args.noise_scales:
        for mode in ["top", "bottom", "random"]:
            key = f"{mode}_{scale_mult}"
            print(f"  {key}...", end=" ", flush=True)

            # Register hooks on all probe layers
            hooks = []
            gen = torch.Generator(device=device)
            gen.manual_seed(args.seed)

            modules = dict(model.named_modules())
            for ln in layer_names:
                U_r = subspaces[(ln, mode)]
                noise_scale = scale_mult * dim_scales[(ln, mode)]
                hook = _make_noise_hook(U_r, noise_scale, gen)
                hooks.append(modules[ln].register_forward_hook(hook))

            # Evaluate
            ppl = eval_math_perplexity(model, tokenizer, eval_set, device, args.eval_samples)

            # Remove hooks
            for h in hooks:
                h.remove()

            delta = ppl - baseline_ppl
            intervention_results[key] = {
                "mode": mode,
                "noise_scale_mult": scale_mult,
                "ppl": ppl,
                "delta_ppl": delta,
            }
            print(f"ppl={ppl:.4f} (Δ={delta:+.4f})")

    elapsed = time.time() - t0
    print(f"\nInterventions complete ({elapsed:.0f}s)")

    # Summary
    print("\n" + "=" * 60)
    print("  CAUSAL INTERVENTION RESULTS")
    print("=" * 60)
    print(f"  Baseline ppl: {baseline_ppl:.4f}")
    print(f"  {'Intervention':<20s}  {'PPL':>8s}  {'Δ PPL':>10s}")
    print("  " + "-" * 42)

    for scale_mult in args.noise_scales:
        for mode in ["top", "bottom", "random"]:
            key = f"{mode}_{scale_mult}"
            r_data = intervention_results[key]
            print(f"  {key:<20s}  {r_data['ppl']:8.4f}  {r_data['delta_ppl']:+10.4f}")
        print()

    # Gate check: at each noise scale, top should degrade more than bottom and random
    print("  Gate checks (top-r Δppl > random-r AND top-r Δppl > bottom-r):")
    all_pass = True
    for scale_mult in args.noise_scales:
        top_delta = intervention_results[f"top_{scale_mult}"]["delta_ppl"]
        rand_delta = intervention_results[f"random_{scale_mult}"]["delta_ppl"]
        bot_delta = intervention_results[f"bottom_{scale_mult}"]["delta_ppl"]
        pass_rand = top_delta > rand_delta
        pass_bot = top_delta > bot_delta
        ok = pass_rand and pass_bot
        if not ok:
            all_pass = False
        print(
            f"    {scale_mult}σ: top={top_delta:+.4f} > rand={rand_delta:+.4f}? "
            f"{'✓' if pass_rand else '✗'}  "
            f"top={top_delta:+.4f} > bot={bot_delta:+.4f}? "
            f"{'✓' if pass_bot else '✗'}  "
            f"{'✓' if ok else '✗'}"
        )

    print(f"\n  GATE: {'✓ PASS — top-r dims are causally important' if all_pass else '✗ FAIL — no causal evidence for top-r'}")
    print("=" * 60)

    # Save results
    full_results = {
        "config": vars(args),
        "baseline_ppl": baseline_ppl,
        "noise_scales": args.noise_scales,
        "layer_names": layer_names,
        "interventions": intervention_results,
        "dim_scales": {f"{ln}_{mode}": dim_scales[(ln, mode)] for ln in layer_names for mode in ["top", "bottom", "random"]},
        "gate_pass": all_pass,
    }
    results_path = os.path.join(args.out_dir, "causal_test.json")
    with open(results_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Plot
    plot_results(full_results, args.out_dir)

    print(f"Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
