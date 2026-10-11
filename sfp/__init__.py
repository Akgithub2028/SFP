"""Subspace Feature Preservation (SFP) package."""

from __future__ import annotations

from sfp import analyze, data, evaluate, features, forget, methods, model, train
from sfp.analyze import compute_drift
from sfp.data import get_batch, load_task
from sfp.evaluate import PRIMARY_METRIC, compute_forgetting, evaluate_all
from sfp.features import build_pca_basis, collect_activations, get_layer_names, rank_importance
from sfp.methods import METHODS, method_setup
from sfp.model import load_model, save_checkpoint

__all__ = [
    "analyze",
    "data",
    "evaluate",
    "features",
    "forget",
    "methods",
    "model",
    "train",
    "compute_drift",
    "load_task",
    "get_batch",
    "evaluate_all",
    "PRIMARY_METRIC",
    "compute_forgetting",
    "collect_activations",
    "build_pca_basis",
    "rank_importance",
    "get_layer_names",
    "METHODS",
    "method_setup",
    "load_model",
    "save_checkpoint",
]
