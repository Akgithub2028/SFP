# sfp

**Does your LLM forget? Find out in 5 minutes.**

```
python forget.py
```

sfp is a minimal harness for studying catastrophic forgetting in LLMs. Fine-tune a model on sequential tasks, watch it forget, and try methods to prevent it. All methods compete on a standardized benchmark with a [live leaderboard](https://sfp-leaderboard.pages.dev).

## Quick start

```bash
git clone https://github.com/paradigmxyz/sfp.git && cd sfp
uv sync
python forget.py
```

**With a GPU:**
```bash
python train.py --method sfp --memory 128
```

## The idea

When you fine-tune an LLM on Task A (math) then Task B (code), it forgets math. This is **catastrophic forgetting** — a key obstacle in deploying LLMs that learn sequentially (alignment updates, domain adaptation, safety patches, personalization).

We investigate a specific, falsifiable hypothesis:

> **H1:** Measurable regressions are predictable from activation drift in a small subspace. Drift in the top-*r* PCA dimensions (~32 per layer) at mid-depth layers predicts old-task accuracy drops (R² ≥ 0.6); drift in random-*r* dimensions does not.

If supported, you don't need to preserve the entire model — just the right features. That's **Sparse Feature Preservation (SFP)**: find the important activation directions via PCA + ablation, then add a loss term that anchors only those during new task training. ~40 lines of code.

### What the formalization proves

We formalize the mathematical reduction in [Lean 4](lean-sfp/):

```
IF   forgetting ≤ C · ‖P_W(Δa)‖        (H1 — empirical, tested by experiments)
AND  L_pres ≤ δ                          (SFP — minimize the preservation loss)
THEN forgetting ≤ C · √δ                (verified in Lean, no sorry)
```

Lean verifies the *mathematical implication* — a chain of inequalities in an abstract inner product space. It does not verify the empirical premises: whether H1 holds for real models, whether C is small enough to be non-vacuous, or whether training achieves a small δ. The experiments exist to test those.

What the formalization buys you: no hand-wavy "therefore" steps, explicit assumptions, and machine-checked algebra. What it doesn't buy you: guarantees about real LLM training dynamics.

### The research loop

```
  HYPOTHESIZE ──▶ EXPERIMENT ──▶ FORMALIZE ──▶ WRITE UP
  (text)          (python)       (lean 4)      (latex)
  dev/RESEARCH.md train.py       lean-sfp/     paper/
```

1. State a hypothesis with falsification criteria
2. Test it in Python — if it fails, stop
3. If it survives, formalize the math in Lean 4
4. Write it up in LaTeX, referencing theorem names directly

Gate rule: never formalize a claim that hasn't survived an experiment.

### Threats to validity

- **Model scale**: primary experiments use small models (135M–1.5B) with LoRA. Low-rank adapters constrain updates to low-dimensional subspaces, which could make "low-dimensional forgetting" more likely by construction. Full fine-tuning and larger models are needed to confirm H1 generalizes.
- **Task order**: results may be sensitive to curriculum ordering. We test multiple permutations but a single benchmark cannot cover all settings.
- **Benchmark coverage**: GSM8K, HumanEval, IFEval, safety, and domain QA are a practical but limited slice of LLM capabilities.
- **Leaderboard score**: the composite 0.6·retention + 0.4·plasticity is a convenience ranking, not a scientific metric. Full Pareto frontiers and per-task breakdowns are the real results.

## File structure

```
.
├── forget.py           # 5-minute demo: watch forgetting happen
├── train.py            # Continual training loop (~300 lines)
├── evaluate.py         # Eval harness + forgetting metrics (AA, BWT, FWT)
├── analyze.py          # Drift analysis, R² regression, Pareto plots
├── methods.py          # All methods: naive, replay, distill, sfp
├── features.py         # Activation hooks, PCA subspace, importance
├── data.py             # Dataset loading + memory buffers
├── model.py            # Model loading + LoRA setup
├── configs/            # demo (5min CPU), fast (1 GPU), full (8×H100)
├── lean-sfp/           # Lean 4 formalization (builds in Docker)
│   └── SFP/
│       ├── LinearAlgebra/Projections.lean
│       ├── Loss/Preservation.lean
│       ├── Loss/GradientOrthogonality.lean
│       └── Paper/Reduction.lean    ← the main theorem
├── paper/              # LaTeX paper (builds in Docker)
├── tasks/              # Eval tasks: math, code, ifeval, safety
├── runs/               # Experiment scripts
└── dev/
    ├── RESEARCH.md     # Full research briefing
    └── LEADERBOARD.md  # How to submit
```

## Commands

```bash
python forget.py                    # watch forgetting happen (5 min, CPU)
python train.py --method sfp        # train with SFP
python analyze.py --compare a b     # compare two runs
make lean                           # build Lean formalization (Docker)
make lean-sorry                     # check unproved theorems
make paper                          # build PDF (Docker)
```

## Contributing

```bash
cp submissions/_example.py submissions/my_method.py
# edit your loss function
python submit.py submissions/my_method.py
```

A GPU spins up in the cloud, runs the benchmark (3 seeds, ~45 min), and your score appears on the [leaderboard](https://sfp-leaderboard.pages.dev) automatically.

## Cite

```bibtex
@misc{sfp,
  author = {Paradigm},
  title = {sfp: Does your LLM forget? Find out in 5 minutes.},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/paradigmxyz/sfp}
}
```

## License

MIT
