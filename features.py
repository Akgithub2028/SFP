"""
features.py — Activation hooks, PCA subspace construction, importance ranking.

~150 lines. Uses torch + numpy. No imports from other project files.
"""

from __future__ import annotations
import torch
from torch import Tensor
import numpy as np


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
    raise NotImplementedError


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
    raise NotImplementedError


def build_pca_basis(activations: Tensor, k: int = 128) -> tuple[Tensor, Tensor]:
    """Compute PCA basis via SVD.

    Args:
        activations: [n_samples, hidden_dim] tensor.
        k: Number of principal components to keep.

    Returns:
        (U, explained_variance) where U is [hidden_dim, k].
    """
    raise NotImplementedError


def rank_importance(
    model,
    tokenizer,
    basis: Tensor,
    memory_set: list[dict],
    eval_fn,
    layer_name: str,
) -> Tensor:
    """For each PC dimension j, zero it and measure metric drop.

    Returns:
        importance scores tensor of shape [k].
    """
    raise NotImplementedError


def select_top_r(basis: Tensor, importance: Tensor, r: int = 32) -> Tensor:
    """Select top-r most important dimensions from basis.

    Returns:
        U_r of shape [hidden_dim, r].
    """
    raise NotImplementedError
