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

When you fine-tune an LLM on Task A (math) then Task B (code), it forgets math. This is **catastrophic forgetting** — one of the biggest unsolved problems in deploying LLMs that learn over time.

We investigate a specific hypothesis about why:

> **H1:** Forgetting is low-dimensional. Activation drift in a small feature subspace (~32 dimensions per layer) predicts forgetting on old tasks.

If true, you don't need to preserve the entire model — just the right features. That's **Sparse Feature Preservation (SFP)**: find the important activation directions via PCA + ablation, then add a loss term that anchors only those during new task training. ~40 lines of code.

### The reduction

We formalize this as a reduction in [Lean 4](lean-sfp/):

```
IF   forgetting ≤ C · ‖P_W(Δa)‖        (H1 — empirical, validated by experiments)
AND  L_pres ≤ δ                          (SFP — minimize the preservation loss)
THEN forgetting ≤ C · √δ                (proved in Lean, no sorry)
```

**The only unverified assumption is H1.** Everything else — the loss bounds subspace drift, gradients stay orthogonal to preserved features, the Pythagorean decomposition into stability vs. plasticity — is [formally verified](lean-sfp/SFP/Paper/Reduction.lean). The experiments exist to validate H1.

### The research loop

The project runs a 4-stage loop:

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

## File structure

```
.
├── forget.py           # 5-minute demo: watch forgetting happen
├── train.py            # Continual training loop (~300 lines)
├── evaluate.py         # Eval harness + forgetting metrics
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
