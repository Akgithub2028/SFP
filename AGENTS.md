# SFP — Autonomous Research Loop

## What this project is

SFP (Sparse Feature Preservation) investigates catastrophic forgetting in LLMs.
The core claim: forgetting is low-dimensional and can be prevented by preserving
a small set of important activation features during fine-tuning.

## The research loop

This project runs a 4-stage loop. Each stage has concrete artifacts and commands.
The loop can be run autonomously — prompt a hypothesis, test it, formalize it, write it up.

```
  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
  │  HYPOTHESIZE │────▶│  EXPERIMENT  │────▶│  FORMALIZE   │────▶│  WRITE UP   │
  │  (text)      │     │  (python)    │     │  (lean)      │     │  (latex)     │
  └──────┬───────┘     └──────┬───────┘     └──────┬───────┘     └──────┬───────┘
         │                    │                    │                    │
         │  dev/RESEARCH.md   │  out/*, analyze.py │  lean-sfp/SFP/    │  paper/sections/
         │                    │  train.py, eval    │  docker build     │  make paper
         └────────────────────┴────────────────────┴────────────────────┘
                                    ▲     revisit if falsified     │
                                    └──────────────────────────────┘
```

### Stage 1: Hypothesize

Write the hypothesis in plain text in `dev/RESEARCH.md` under the hypotheses section.
A good hypothesis is:
- **Falsifiable**: has a clear R² threshold or comparison baseline
- **Mechanistic**: says WHERE in the model something happens, not just that it does
- **Connected**: relates to the reduction chain (H1 → SFP bounds forgetting)

Current hypotheses:
- **H1**: Forgetting is low-dimensional — drift in top-r PCA dims predicts forgetting (R² ≥ 0.6)
- **H2**: Preserving top-r features improves the stability-plasticity Pareto frontier
- **H3**: Preserved features are mechanistically coherent (interpretable, causal)

### Stage 2: Experiment

Run the experiment to test the hypothesis:

```bash
# Quick falsification (~5 min, CPU)
python train.py --config configs/demo.yaml --method sfp

# Compare methods
python train.py --config configs/fast.yaml --method naive
python train.py --config configs/fast.yaml --method sfp
python analyze.py --compare out/naive out/sfp

# Full benchmark (GPU, ~45 min)
bash runs/full.sh
```

Key files:
- `train.py` — training loop, ~300 lines
- `evaluate.py` — eval harness + forgetting metrics
- `analyze.py` — drift computation, R² regression, Pareto plots
- `methods.py` — all methods as plain loss functions
- `features.py` — activation hooks, PCA, importance ranking
- `data.py` — datasets + memory buffers

**Gate rule**: if an experiment falsifies a hypothesis, update RESEARCH.md before proceeding.
Don't formalize claims that don't survive experiments.

### Stage 3: Formalize

Formalize the mathematical claims in Lean 4 (in `lean-sfp/`):

```bash
# Build the Lean formalization
docker build -t sfp-lean lean-sfp/
docker run --rm sfp-lean build

# Check sorry count (goal: zero for paper-critical theorems)
docker run --rm sfp-lean env grep -rn sorry SFP/
```

The formalization structure:
```
lean-sfp/
  SFP/
    LinearAlgebra/Projections.lean  — orthogonal projection properties
    Loss/Preservation.lean          — preservation loss definition + bounds
    Loss/GradientOrthogonality.lean — gradient orthogonality at stationarity
    Paper/Reduction.lean            — THE MAIN RESULT: H1 + SFP → forgetting bound
    Paper/SFPTheorems.lean          — index of all paper-referenced results
```

**The reduction pattern**: every formal result should either:
1. Be a standard fact (projection properties) — cite Mathlib
2. Reduce a claim to H1 — this is where the value is
3. Verify a property of the loss function — correctness guarantee

The key theorem is `sfp_reduces_forgetting`:
```
IF   forgetting ≤ C · ‖P_W(Δa)‖        (H1, empirical)
AND  preservationLoss ≤ δ                (SFP guarantee)
THEN forgetting ≤ C · √δ                (verified)
```

This means: **the only unverified assumption is H1. Everything else is machine-checked.**

### Stage 4: Write up

Write the paper in LaTeX (in `paper/`):

```bash
make paper    # builds via Docker, outputs sfp.pdf
```

Paper sections map to the loop:
- **§1 Introduction** — motivation, the forgetting problem
- **§2 Related Work** — EWC, LwF, PODNet, orthogonal LoRA, replay
- **§3 Method** — SFP definition, formally verified properties, reduction theorem
- **§4 Experiments** — H1 validation (R²), H2 validation (Pareto), H3 (interp)
- **§5 Conclusion**

When writing the method section, reference Lean theorem names directly:
```latex
\begin{theorem}[Main Reduction, \texttt{sfp\_reduces\_forgetting}]
If $\text{forgetting} \leq C \|P_W(\Delta a)\|$ and $\mathcal{L}_{\text{pres}} \leq \delta$,
then $\text{forgetting} \leq C\sqrt{\delta}$.
\end{theorem}
```

## Running the full loop for a new hypothesis

To run the complete loop for a new hypothesis:

1. **State it**: Add to `dev/RESEARCH.md` with falsification criteria
2. **Test it**: Write the experiment in Python, run it, check the gate
3. **Formalize it**: If it survives, add the math to `lean-sfp/SFP/`, build, verify
4. **Write it**: Add to the appropriate `paper/sections/*.tex`, rebuild PDF

Example — testing a new hypothesis "importance ranking via gradient norm beats ablation":

```bash
# 1. Add hypothesis to RESEARCH.md (with R² threshold)
# 2. Implement gradient-based importance in features.py
# 3. Run experiment
python train.py --config configs/fast.yaml --method sfp_grad_importance
python analyze.py --compare out/sfp out/sfp_grad_importance
# 4. If it wins: formalize the gradient importance bound in Lean
# 5. If it loses: note in RESEARCH.md as falsified, move on
```

## Conventions

- **All Docker**: Lean builds in Docker (`sfp-lean`), LaTeX builds in Docker (`sfp-latex`). No local installations required.
- **Small diffs**: when editing methods.py or features.py, make the smallest change. Methods are plain functions, not classes.
- **Gate before formalize**: never spend time formalizing a claim that hasn't survived an experiment.
- **Sorry budget**: `sorry` is fine for standard Mathlib facts (projections, Pythagorean). Never `sorry` the reduction theorems.
- **Configs**: `demo.yaml` for ~5 min CPU tests, `fast.yaml` for 1-GPU validation, `full.yaml` for the real benchmark.

## Current status

### Proved (no sorry)
- `sfp_reduces_forgetting`: the main reduction H1 → forgetting bound ✅
- `gradient_orthogonality`: P(∇L_new) = 0 at preserved stationary point ✅
- `gradient_projection_at_stationary`: P(∇L_new) = -2λv (general) ✅
- `preservationLoss_nonneg`, `_eq_zero_iff`, `subspace_drift_bound` ✅

### Needs proof (sorry, standard facts)
- `orthogonalProjection_selfAdjoint`, `_norm_le`, `_complement`
- `orthonormal_projection_formula`
- `preservationLoss_le_total_drift`, `drift_pythagorean_decomposition`
- `stability_plasticity_tradeoff` (plasticity half)

### Needs experiments
- H1: R² of top-r drift vs. forgetting (target: ≥ 0.6)
- H2: Pareto frontier comparison at M=128
- H3: Feature interpretability / causal ablation
