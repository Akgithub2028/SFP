# Sparse Feature Preservation

**A benchmark and method for studying catastrophic forgetting in large language models.**

Sequential fine-tuning of LLMs — alignment updates, domain adaptation, safety patches — causes capability regressions on previously learned tasks. We provide (1) a reproducible benchmark for measuring these regressions, (2) a falsifiable hypothesis about their low-dimensional structure, and (3) a formally verified reduction from that hypothesis to a tractable preservation objective.

```bash
git clone https://github.com/paradigmxyz/sfp.git && cd sfp
uv sync
python forget.py          # observe forgetting on SmolLM2-135M (~5 min, CPU)
python train.py --method sfp --memory 128   # train with SFP (GPU)
```

## Hypothesis

We test whether measurable regressions during continual fine-tuning are predictable from activation drift in a small subspace:

> **H1.** Let *a(θ)* denote activations at a probed layer and *W* a rank-*r* subspace identified via PCA + ablation-based importance ranking. Drift in the projection onto *W* predicts old-task accuracy drops (R² ≥ 0.6), while drift in a random rank-*r* subspace does not, even after controlling for total activation drift ‖Δa‖.

If H1 is supported, preserving the full model is unnecessary — anchoring a small set of activation directions suffices.

### Current experimental status

**Correlational evidence (Qwen2.5-1.5B, Math→Code):** H1 partially supported. Top-*r* drift predicts forgetting with R²=0.933, outperforming random (0.850) and bottom-ranked (0.827) dimensions — the correct ranking, though the margin over random is modest.

| Predictor | Dims | R² |
|-----------|------|-----|
| **Top-*r* (ablation)** | 32 | **0.933** |
| Total drift | 128 | 0.888 |
| Random dims | 32 | 0.850 |
| Bottom-*r* (ablation) | 32 | 0.827 |

**Causal evidence (SmolLM2-135M, noise injection):** Even at 135M scale where correlational H1 fails, perturbing top-*r* dims causes 4× more performance degradation than random dims:

| Intervention | Δ perplexity |
|-------------|-------------|
| Top-*r* | +0.85 |
| Random | +0.21 |
| Bottom-*r* | +0.02 |

**Scale dependence:** H1 fails at 135M (R² ranking inverted), passes at 1.5B. The causal test passes at both scales.

## Method

**Sparse Feature Preservation (SFP)** adds a regularization term that penalizes drift in the identified subspace during new-task training:

$$\mathcal{L} = \mathcal{L}_{\text{new}} + \lambda \sum_{\ell} \| U_r^\top\, a_\ell(\theta) - U_r^\top\, a_\ell(\theta^*) \|^2$$

where *U_r* contains the top-*r* important PCA directions at layer ℓ, and θ* is the anchor checkpoint. The procedure: (1) collect activations on a memory buffer, (2) compute PCA basis, (3) rank dimensions by ablation impact on old-task performance, (4) select top-*r*, (5) add the preservation loss. Implementation is ~40 lines in [`methods.py`](methods.py).

## Formal verification

The mathematical reduction is verified in [Lean 4](lean-sfp/) (Mathlib v4.17.0, 0 `sorry` remaining):

```
IF   forgetting ≤ C · ‖P_W(Δa)‖        (H1 — empirical, tested by experiments)
AND  L_pres ≤ δ                          (SFP training objective)
THEN forgetting ≤ C · √δ                (Theorem sfp_reduces_forgetting, verified)
```

Lean verifies the implication — a chain of inequalities in an abstract inner product space — not the empirical premises. Whether H1 holds, whether *C* is non-vacuous, and whether SGD achieves small δ are empirical questions addressed by the experiments. The formalization makes assumptions explicit and prevents algebraic errors in the reduction; it does not provide guarantees about training dynamics.

Additional verified results: gradient orthogonality at stationary points, Pythagorean stability-plasticity decomposition, projection properties. See [`SFPTheorems.lean`](lean-sfp/SFP/Paper/SFPTheorems.lean) for the full index.

## Benchmark

The benchmark evaluates continual fine-tuning across a sequence of tasks using standard metrics from the continual learning literature:

| Task | Training data | Evaluation | Primary metric |
|------|--------------|------------|----------------|
| Math | GSM8K | GSM8K test (exact match) | `math_accuracy` |
| Code | MBPP | HumanEval pass@1 (sandboxed execution) | `code_pass_at_1` |
| Instruction following | Alpaca | IFEval (constraint satisfaction) | `ifeval_accuracy` |
| Safety | HH-RLHF | Refusal rate × (1 − over-refusal rate) | `safety_score` |
| Domain | SciQ / PubMed | Perplexity + science QA | `domain_qa_accuracy` |

### Protocol

Following best practices from CLEAR, CTrL, and Avalanche benchmarks:

- **Zero-shot baseline** recorded before any training — the reference for distinguishing "the model forgot" from "the model was already bad"
- **3 task orderings** tested (task order is the #1 variance source in CL; fixed-order benchmarks let methods overfit the sequence)
- **3 seeds per ordering** (9 total runs per method)
- **Setup budget**: method setup at each task boundary is capped at 120s wall-clock to prevent smuggling unbounded computation
- **Code sandboxing**: HumanEval execution uses Docker isolation (no network, 256MB RAM) with OS-level fallback

### Scoring

The leaderboard score is the **harmonic mean** of retention and plasticity:

```
score = 2 · retention · plasticity / (retention + plasticity)
```

This is analogous to F1 = harmonic mean of precision and recall. Unlike a linear combination (which lets a do-nothing method score 0.6 from perfect retention alone), the harmonic mean requires performing well on *both* axes. If either metric falls below 0.1, the score is clamped to 0.

Results also report **Pareto optimality** — whether a method is dominated on both axes by any other.

**Metrics:** Average Accuracy (AA), Backward Transfer (BWT), Forward Transfer (FWT), per-task forgetting, retention-plasticity Pareto frontiers.

**Baselines:** naive (no mitigation), replay, logit distillation (LwF), hidden-state distillation (PODNet-style), orthogonal LoRA, SFP, and ablation controls (random basis, random selection).

### Threats to validity

- **LoRA confound.** Low-rank adapters constrain updates to low-dimensional subspaces, which could make low-dimensional forgetting more likely by construction. Full fine-tuning experiments are needed to confirm H1 generalizes.
- **Model scale.** Primary experiments use 135M–1.5B parameters. H1 is scale-dependent (fails at 135M, passes at 1.5B). Results at 7B+ are unknown.
- **Task coverage.** H1 validation focuses on Math→Code. Different task pairs may exhibit different forgetting geometries.
- **Bound tightness.** The constant *C* in the reduction is estimated empirically. If *C* is large or unstable across settings, the bound is vacuous.
- **Layer selection.** Pre-registered probe layers at L/4, L/2, 3L/4 may not be optimal for all models/tasks.
- **Task ordering.** We test 3 permutations; exhaustive testing of all 120 orderings (5!) is computationally prohibitive. The 3 chosen orderings rotate which task appears last, mitigating last-task bias in plasticity measurement.

## Repository structure

```
train.py            Continual training loop
evaluate.py         Evaluation harness (AA, BWT, FWT, forgetting)
analyze.py          Drift analysis, R² regression, Pareto plots
methods.py          All methods as plain loss functions (~9 methods)
features.py         Activation hooks, PCA, gradient basis, importance ranking
data.py             Dataset loading, memory buffers
model.py            Model loading, LoRA configuration
forget.py           Quick demo: observe forgetting (~5 min CPU)
scripts/            Experiment scripts (h1_control.py, causal_test.py, modal_*.py)
configs/            Experiment configs (demo, fast, full)
tasks/              Per-task evaluation (math, code, ifeval, safety, domain)
lean-sfp/           Lean 4 formalization (0 sorry, Dockerized build)
paper/              LaTeX source (Dockerized build)
web/                Leaderboard website (React + Cloudflare Workers)
dev/RESEARCH.md     Full research briefing + experiment log
```

## Usage

```bash
python forget.py                        # demo: observe forgetting (~5 min CPU)
python train.py --method sfp            # train with SFP
python train.py --config configs/fast.yaml  # 1-GPU validation run
python analyze.py --compare out/a out/b # compare methods
make lean                               # verify Lean formalization
make paper                              # build paper PDF
```

## Submitting a method

```bash
cp submissions/_example.py submissions/my_method.py
# implement your loss function
python submit.py submissions/my_method.py
```

Submissions are benchmarked on cloud GPUs across 3 task orderings × 3 seeds (9 runs, ~6h total). Results appear on the [leaderboard](https://sfp-leaderboard.pages.dev) with mean ± std across all runs and Pareto optimality status.

## References

- Kirkpatrick et al., 2017. Overcoming catastrophic forgetting in neural networks (EWC).
- Li & Hoiem, 2017. Learning without Forgetting (LwF).
- Douillard et al., 2020. PODNet: Pooled Outputs Distillation (ECCV).
- Wang et al., 2024. Continual Learning for Large Language Models (arXiv:2404.16789).

## Citation

```bibtex
@misc{sfp2026,
  author = {Paradigm},
  title = {Sparse Feature Preservation: A Benchmark for Catastrophic Forgetting in LLMs},
  year = {2026},
  url = {https://github.com/paradigmxyz/sfp}
}
```

## License

MIT
