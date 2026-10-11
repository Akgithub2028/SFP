# Reproducibility Guide

This document provides step-by-step instructions to reproduce all empirical experiments, formal mathematical proofs, and paper figures.

---

## 1. Hardware Requirements

| Experiment Tier | Target Model | Hardware Requirement | Approx. Runtime |
|-----------------|--------------|----------------------|-----------------|
| **Quick Falsification** | SmolLM2-135M | Modern CPU (4+ cores, 8GB RAM) | ~5 minutes |
| **Fast 1-GPU Validation** | SmolLM2-135M / Qwen-1.5B | 1x NVIDIA RTX 3090/4090 or A10G (16GB VRAM) | ~25 minutes |
| **Full 5-Task Benchmark** | Qwen2.5-1.5B-Instruct | 1x NVIDIA A10G / A100 (24GB+ VRAM) | ~45 min / run |
| **Full 9-Run Sequence** | Qwen2.5-1.5B-Instruct | Modal Cloud Cluster or 8x A10G | ~6 hours total |

---

## 2. Environment Setup

### 2.1 Dependencies via `uv` (Recommended)

```bash
# Clone the repository
git clone https://github.com/Akgithub2028/sfp.git && cd sfp

# Sync dependencies in isolated virtual environment
uv sync
source .venv/bin/activate
```

### 2.2 Standard Python / Pip

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

---

## 3. Reproduction Commands

### 3.1 Quick Falsification Smoke Test (~5 min, CPU)
Observe sequential capability degradation on SmolLM2-135M without requiring a GPU:

```bash
python -m sfp.forget
```

### 3.2 Single-Method Training
Train a model using Subspace Feature Preservation (SFP):

```bash
# Train SFP with memory budget of 128 samples
python -m sfp.train --method sfp --memory 128 --model HuggingFaceTB/SmolLM2-135M-Instruct

# Fast GPU validation run
python -m sfp.train --config configs/fast.yaml --method sfp
```

### 3.3 Baseline Comparison
Compare naive sequential fine-tuning against SFP:

```bash
python -m sfp.train --config configs/fast.yaml --method naive --out_dir out/naive
python -m sfp.train --config configs/fast.yaml --method sfp --out_dir out/sfp

# Compute drift, R² regressions, and comparison plots
python -m sfp.analyze --compare out/naive out/sfp
```

### 3.4 Causal Intervention Test (Hypothesis H1)
Evaluate whether perturbing top-$r$ PCA dimensions causes superlinear capability drop:

```bash
python scripts/causal_test.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --pca_k 128 \
    --preserve_r 32 \
    --memory 128 \
    --eval_samples 50 \
    --device auto
```

### 3.5 Full Sequential Benchmark Run
To execute the multi-order multi-seed benchmark:

```bash
bash runs/leaderboard.sh sfp 128
```

---

## 4. Formal Proof Verification (Lean 4)

The mathematical reduction is verified in Lean 4 with 0 `sorry` axioms.

```bash
# Build the Lean formalization using Docker
docker build -t sfp-lean lean-sfp/
docker run --rm sfp-lean build

# Audit the sorry count across all proofs (expected: 0 matches)
docker run --rm sfp-lean env grep -rn sorry SFP/
```

Or locally if `elan` / `lake` are installed:
```bash
cd lean-sfp
lake build
grep -rn "sorry" SFP/
```

---

## 5. Paper Compilation

Compile the research paper LaTeX source:

```bash
# Compile paper PDF via Dockerized toolchain
make paper

# Output will be generated at:
# ./sfp.pdf
```
