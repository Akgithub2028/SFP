"""
methods.py — All continual learning methods as plain functions.

~250 lines. Each method is a function that returns a loss tensor.
No classes, no inheritance. Just functions.
"""

from __future__ import annotations
import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Callable


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def naive_loss(model, batch: dict, **kw) -> Tensor:
    """Standard cross-entropy loss. No forgetting mitigation. ~5 lines."""
    raise NotImplementedError


def replay_loss(model, batch: dict, memory_batch: dict | None = None,
                replay_ratio: float = 0.1, **kw) -> Tensor:
    """Mix new-task CE with memory replay CE at given ratio. ~15 lines."""
    raise NotImplementedError


def logit_distill_loss(model, batch: dict, teacher=None,
                       memory_batch: dict | None = None,
                       temperature: float = 2.0, alpha: float = 0.5, **kw) -> Tensor:
    """LwF-style: CE on new task + KL(teacher || student) on memory. ~25 lines."""
    raise NotImplementedError


def hidden_distill_loss(model, batch: dict, teacher=None,
                        memory_batch: dict | None = None,
                        layers: list[str] | None = None, **kw) -> Tensor:
    """POD-style: CE + L2 match of normalized hidden states at selected layers. ~30 lines."""
    raise NotImplementedError


def orthogonal_loss(model, batch: dict,
                    prev_lora_weights: dict | None = None,
                    ortho_lambda: float = 1.0, **kw) -> Tensor:
    """Constrain new LoRA to be orthogonal to previous adapters. ~30 lines."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# SFP (our method)
# ---------------------------------------------------------------------------

def sfp_loss(model, batch: dict, memory_batch: dict | None = None,
             basis: dict | None = None, anchor_acts: dict | None = None,
             lam: float = 0.1, **kw) -> Tensor:
    """Sparse Feature Preservation loss.

    L = L_new + λ * Σ_ℓ || U_r^T a_ℓ(x;θ) − anchor_acts_ℓ ||²

    Args:
        basis: {layer_name: U_r tensor [hidden_dim, r]}
        anchor_acts: {layer_name: projected activations [n, r] at anchor checkpoint}
        lam: Preservation weight.
    """
    raise NotImplementedError


def sfp_random_basis_loss(model, batch: dict, memory_batch: dict | None = None,
                          basis: dict | None = None, anchor_acts: dict | None = None,
                          lam: float = 0.1, **kw) -> Tensor:
    """Same as SFP but with random orthonormal basis. Control for PCA."""
    raise NotImplementedError


def sfp_random_selection_loss(model, batch: dict, memory_batch: dict | None = None,
                              basis: dict | None = None, anchor_acts: dict | None = None,
                              lam: float = 0.1, **kw) -> Tensor:
    """Same as SFP with PCA basis but random dim selection. Control for importance."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Setup functions (called at task boundaries)
# ---------------------------------------------------------------------------

def method_setup(method_name: str, model, tokenizer,
                 memory_buffers: dict, **kw) -> dict:
    """Method-specific setup at each task boundary.

    Returns a state dict passed to the loss function each step.
    E.g., for distillation methods: creates teacher snapshot.
    For SFP: builds PCA basis + ranks importance + caches anchors.
    """
    raise NotImplementedError


def sfp_setup(model, tokenizer, memory_buffers: dict,
              layers: list[str] | None = None,
              k: int = 128, r: int = 32) -> dict:
    """Build PCA basis, rank importance, select top-r, cache anchor activations.

    Returns: {"basis": {...}, "anchor_acts": {...}}
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

METHODS: dict[str, Callable] = {
    "naive": naive_loss,
    "replay": replay_loss,
    "distill": logit_distill_loss,
    "hidden_distill": hidden_distill_loss,
    "orthogonal": orthogonal_loss,
    "sfp": sfp_loss,
    "sfp_random_basis": sfp_random_basis_loss,
    "sfp_random_selection": sfp_random_selection_loss,
}
