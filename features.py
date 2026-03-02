"""
features.py — Activation hooks, PCA subspace construction, importance ranking.

~150 lines. Uses torch + numpy. No imports from other project files.
"""

from __future__ import annotations

import torch
from torch import Tensor


def get_layer_names(
    model,
    positions: list[float] | None = None,
) -> list[str]:
    """Return layer module names at given relative depth positions.

    Args:
        model: A transformer model.
        positions: Relative depth positions, e.g. [0.25, 0.5, 0.75].
            Defaults to [0.25, 0.5, 0.75].

    Returns:
        List of module name strings (e.g. "model.layers.8").
    """
    if positions is None:
        positions = [0.25, 0.5, 0.75]

    # Find transformer layers by scanning named_modules for ".layers.0" or ".h.0"
    all_names = dict(model.named_modules())
    prefix = None
    n_layers = 0

    for name in all_names:
        if name.endswith(".layers.0"):
            prefix = name.rsplit(".0", 1)[0]
            n_layers = sum(1 for n in all_names if n.startswith(prefix + ".") and n[len(prefix)+1:].isdigit())
            break
        if name.endswith(".h.0"):
            prefix = name.rsplit(".0", 1)[0]
            n_layers = sum(1 for n in all_names if n.startswith(prefix + ".") and n[len(prefix)+1:].isdigit())
            break

    if prefix is None or n_layers == 0:
        raise ValueError(
            f"Cannot detect transformer layers in model modules."
        )

    indices = [int(p * n_layers) for p in positions]
    indices = [min(i, n_layers - 1) for i in indices]
    return [f"{prefix}.{i}" for i in indices]


def collect_activations(
    model,
    dataset: list[dict],
    tokenizer,
    layer_names: list[str],
    pool: str = "mean",
    max_samples: int = 512,
    device: str = "auto",
) -> dict[str, Tensor]:
    """Forward pass over dataset, collect activations at specified layers.

    Args:
        pool: "mean" for token-mean pooling, "last" for last token.

    Returns:
        {layer_name: tensor of shape [n_samples, hidden_dim]}
    """
    if device == "auto":
        device = str(next(model.parameters()).device)

    samples = dataset[:max_samples]
    buffers: dict[str, list[Tensor]] = {name: [] for name in layer_names}
    hooks = []

    def _make_hook(name: str):
        def hook_fn(_module, _input, output):
            # output can be a tuple (hidden_states, ...) or a plain tensor
            h = output[0] if isinstance(output, tuple) else output
            if pool == "last":
                buffers[name].append(h[:, -1, :].detach().cpu())
            else:
                buffers[name].append(h.mean(dim=1).detach().cpu())

        return hook_fn

    # Register hooks
    for name in layer_names:
        module = dict(model.named_modules())[name]
        hooks.append(module.register_forward_hook(_make_hook(name)))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    with torch.no_grad():
        for sample in samples:
            text = sample.get("input", sample.get("text", ""))
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=False,
            ).to(device)
            model(**enc)

    for h in hooks:
        h.remove()

    return {name: torch.cat(tensors, dim=0) for name, tensors in buffers.items()}


def build_pca_basis(activations: Tensor, k: int = 128) -> tuple[Tensor, Tensor]:
    """Compute PCA basis via SVD.

    Args:
        activations: [n_samples, hidden_dim] tensor.
        k: Number of principal components to keep.

    Returns:
        (U, explained_variance) where U is [hidden_dim, k].
    """
    X = activations.float()
    X = X - X.mean(dim=0, keepdim=True)

    # SVD: X = U_full @ diag(S) @ Vh  →  columns of Vh^T are PCs in hidden-dim space
    _U_full, S, Vh = torch.linalg.svd(X, full_matrices=False)

    k = min(k, S.shape[0], Vh.shape[0])
    U = Vh[:k].T  # [hidden_dim, k]
    explained_variance = (S[:k] ** 2) / (X.shape[0] - 1)
    return U, explained_variance


def rank_importance(
    model,
    tokenizer,
    basis: Tensor,
    memory_set: list[dict],
    eval_fn,
    layer_name: str,
) -> Tensor:
    """For each PC dimension j, zero it and measure metric drop.

    Args:
        eval_fn: callable(model, tokenizer, memory_set) -> float (higher is better).

    Returns:
        importance scores tensor of shape [k].
    """
    k = basis.shape[1]
    baseline = eval_fn(model, tokenizer, memory_set)
    importance = torch.zeros(k)

    module = dict(model.named_modules())[layer_name]

    for j in range(k):
        direction = basis[:, j].clone()

        def _make_ablate_hook(col: Tensor):
            col_dev = None

            def hook_fn(_module, _input, output):
                nonlocal col_dev
                h = output[0] if isinstance(output, tuple) else output
                if col_dev is None:
                    col_dev = col.to(h.device).float()
                proj = torch.einsum("bsh,h->bs", h.float(), col_dev)
                h = h - torch.einsum("bs,h->bsh", proj, col_dev)
                if isinstance(output, tuple):
                    return (h, *output[1:])
                return h

            return hook_fn

        handle = module.register_forward_hook(_make_ablate_hook(direction))
        score = eval_fn(model, tokenizer, memory_set)
        handle.remove()

        importance[j] = max(baseline - score, 0.0)

    return importance


def select_top_r(basis: Tensor, importance: Tensor, r: int = 32) -> Tensor:
    """Select top-r most important dimensions from basis.

    Returns:
        U_r of shape [hidden_dim, r].
    """
    r = min(r, basis.shape[1])
    indices = torch.argsort(importance, descending=True)[:r]
    return basis[:, indices]
