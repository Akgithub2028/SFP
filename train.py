"""
train.py — Continual fine-tuning training loop.

~300 lines. THE main file. Readable top-to-bottom like nanoGPT's train.py.

Usage:
    python train.py                                    # naive, tiny model, 2 tasks
    python train.py --method replay --memory 128       # replay buffer
    python train.py --method sfp --memory 128          # our method
    python train.py --config configs/fast.yaml         # config-driven
    torchrun --nproc_per_node=8 train.py --config configs/full.yaml  # 8×H100
"""

from __future__ import annotations


if __name__ == "__main__":
    raise NotImplementedError("Training loop not yet implemented")
