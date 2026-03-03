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

- **All Docker**: Lean and LaTeX build in Docker. The `sfp-lean` image caches Mathlib and the toolchain; source files are volume-mounted at build time so you never rebuild the image just to iterate on proofs. Run `make lean-build` once to create the image, then `make lean` to build with your local changes. `make lean-sorry` runs a local grep — no Docker needed.
- **Small diffs**: when editing methods.py or features.py, make the smallest change. Methods are plain functions, not classes.
- **Gate before formalize**: never spend time formalizing a claim that hasn't survived an experiment.
- **Sorry budget**: `sorry` is fine for standard Mathlib facts (projections, Pythagorean). Never `sorry` the reduction theorems.
- **Configs**: `demo.yaml` for ~5 min CPU tests, `fast.yaml` for 1-GPU validation, `full.yaml` for the real benchmark.

## Development rules (learned the hard way)

### Always test locally before Modal
**NEVER fire off a Modal GPU run without testing locally first.** Modal runs cost money, take minutes to build images, and have slow debug cycles. Always:
1. Run on CPU with SmolLM2-135M first: `--memory 16 --pca_k 16 --preserve_r 8`
2. Use minimal eval: `--eval_samples 10 --noise_scales 1.0`
3. Verify no crashes, correct output shapes, plausible values
4. THEN launch on Modal with `--detach` for the real run

### bfloat16 dtype discipline
Qwen and most modern models load as **bfloat16**. Any code that touches activations must preserve dtype:
- Forward hooks: always cast injected tensors to `h.dtype` — never assume float32
- `torch.randn()` produces float32 by default — always `.to(dtype=h.dtype)` after
- `torch.Generator` on CPU can't be used with CUDA tensors — generate on CPU then `.to(device, dtype)`
- After any arithmetic in a hook, verify the output dtype matches the input: `assert h_new.dtype == h.dtype`
- PCA/SVD operations upcast to float32 — that's fine for analysis, but cast back before injecting into the model

### Transformer hook output format
Transformer decoder layers return `BaseModelOutputWithPast` (not a plain tuple). When writing forward hooks:
- Check `isinstance(output, tuple)` first (some layers return tuples)
- Then check `hasattr(output, '__getitem__')` — BaseModelOutput supports indexing
- `output[0]` is always the hidden states tensor
- When returning modified output, preserve the original type — mutate in place for dataclass-like outputs

### Experiment script pattern
All experiment scripts (h1_control.py, causal_test.py, etc.) follow the same pattern:
- Live in `scripts/`
- Self-contained: import from `features`, `model`, `data` but no cross-script imports
- Argparse with `--model`, `--device auto`, `--seed`, `--out_dir`
- Save results as JSON + plots as PDF/PNG to `out_dir`
- Print a summary table to stdout
- Gate check at the end (pass/fail with clear criteria)

### Modal wrappers
For GPU experiments, create a thin `scripts/modal_*.py` wrapper:
- Reuse the same pip install list as `modal_bench.py`
- Use `modal.Volume` for persistent results
- Always use `--detach` for long runs
- Download results with `modal volume get <vol> <remote> <local>`

### Research log discipline
Every experiment gets an entry in `dev/RESEARCH.md` under `## Experiment Log`:
- Date, goal, full config, results table, gate check, interpretation, next steps
- Be honest about failures — record FAIL results, not just passes
- Note caveats (model size, data points, statistical power)

## Current status

### Proved (no sorry)
- `sfp_reduces_forgetting`: the main reduction H1 → forgetting bound ✅
- `sfp_reduces_forgetting_multilayer`: per-layer version ✅
- `stability_plasticity_tradeoff`: stability bound + plasticity lower bound ✅
- `gradient_orthogonality`: P(∇L_new) = 0 at preserved stationary point ✅
- `gradient_in_complement`: ∇L_new ∈ W⊥ at preserved stationary point ✅
- `gradient_projection_at_stationary`: P(∇L_new) = -2λv (general) ✅
- `orthogonalProjection_idempotent`, `_selfAdjoint`, `_norm_le`, `_complement` ✅
- `preservationLoss_nonneg`, `_eq_zero_iff`, `subspace_drift_bound` ✅
- `preservationLoss_le_total_drift`, `drift_pythagorean_decomposition` ✅
- `multiLayerPreservationLoss_nonneg` ✅

### Fully proved
All theorems are now verified with 0 `sorry` remaining.

### H1 status (as of 2026-03-03)
- **135M (CPU, correlational)**: FAIL — R² ranking inverted (bottom > random > total > top). Scale artifact.
- **1.5B (GPU, correlational)**: PARTIAL PASS — R²(top-r)=0.933 > total=0.888 > random=0.850 > bottom=0.827. Correct ranking, but margin over random is modest.
- **135M (CPU, causal)**: PASS — top Δppl=+0.85 >> random +0.21 >> bottom +0.02. Ablation importance identifies causally important dims even at small scale.
- **1.5B (GPU, causal)**: PASS — top Δppl=+0.248 >> random +0.100 >> bottom +0.009 at 1σ. Gate passes at all 4 noise scales (0.5σ, 1σ, 2σ, 5σ). Top/bottom ratio reaches 64× at 2σ.
- **Decision**: H1 has both correlational (R²=0.933) and causal (28× top/bottom at 1σ) support at 1.5B. Keep PCA, strengthen with gradient-informed basis.

### Paper status
All sections written (intro, related, method, experiments, conclusion, appendix). Key results tables populated with 1.5B correlational and causal results. Paper hosted at /paper.pdf on the website (Cloudflare Workers). Pending: full baseline comparison, gradient basis comparison.

### Needs experiments
- ~~H1 causal test (1.5B)~~: ✅ DONE — PASS at all noise scales
- H1 gradient basis: ⏳ RUNNING on Modal (app ap-caoh7G3CAWyw7ktYATkdKs). Download: `modal volume get sfp-grad-compare-results grad_compare/ out/grad_compare/`
- Full baseline comparison: SFP vs naive, replay, LwF, hidden distill, ortho LoRA
- H2: Pareto frontier comparison at M=128
- H3: Feature interpretability / causal ablation

### Website
- Deployed on Cloudflare Workers: `npx wrangler deploy` from `web/`
- Paper PDF served at `/paper.pdf` (copied from `sfp.pdf` to `web/public/paper.pdf`)
- After rebuilding paper with `make paper`, also `cp sfp.pdf web/public/paper.pdf`, rebuild web (`npm run build`), and redeploy (`npx wrangler deploy`)
