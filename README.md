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

## Method

**Sparse Feature Preservation (SFP)** adds a regularization term that penalizes drift in the identified subspace during new-task training:

$$\mathcal{L} = \mathcal{L}_{\text{new}} + \lambda \sum_{\ell} \| U_r^\top\, a_\ell(\theta) - U_r^\top\, a_\ell(\theta^*) \|^2$$

where *U_r* contains the top-*r* important PCA directions at layer ℓ, and θ* is the anchor checkpoint. The procedure: (1) collect activations on a memory buffer, (2) compute PCA basis, (3) rank dimensions by ablation impact on old-task performance, (4) select top-*r*, (5) add the preservation loss. Implementation is ~40 lines in [`methods.py`](methods.py).

## Formal verification

The mathematical reduction is verified in [Lean 4](lean-sfp/) (Mathlib v4.17.0, 1 `sorry` remaining):

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
| Code | MBPP | HumanEval pass@1 (execution) | `code_pass_at_1` |
| Instruction following | Alpaca | IFEval (constraint satisfaction) | `ifeval_accuracy` |
| Safety | HH-RLHF | Refusal rate × (1 − over-refusal rate) | `safety_score` |
| Domain | PubMed | Perplexity + science QA | `domain_qa_accuracy` |

**Metrics:** Average Accuracy (AA), Backward Transfer (BWT), Forward Transfer (FWT), per-task forgetting, retention-plasticity Pareto frontiers. The composite leaderboard score (0.6·retention + 0.4·plasticity) is a convenience ranking only.

**Baselines:** naive (no mitigation), replay, logit distillation (LwF), hidden-state distillation (PODNet-style), orthogonal LoRA, SFP, and ablation controls (random basis, random selection).

### Threats to validity

- **Model scale.** Primary experiments use 135M–1.5B parameters with LoRA. Low-rank adapters constrain updates to low-dimensional subspaces, which could make low-dimensional forgetting more likely by construction. Full fine-tuning experiments are needed to confirm H1 generalizes.
- **Task order.** Results may be sensitive to curriculum ordering. Multiple permutations should be tested.
- **Benchmark coverage.** Five tasks spanning math, code, instruction following, safety, and domain knowledge. This is a practical but limited slice of LLM capabilities.
- **Bound tightness.** The constant *C* in the reduction is estimated empirically. If *C* is large or unstable across settings, the bound is vacuous.

## Repository structure

```
train.py            Continual training loop
evaluate.py         Evaluation harness (AA, BWT, FWT, forgetting)
analyze.py          Drift analysis, R² regression, Pareto plots
methods.py          All methods as plain loss functions
features.py         Activation hooks, PCA, importance ranking
data.py             Dataset loading, memory buffers
model.py            Model loading, LoRA configuration
configs/            Experiment configs (demo, fast, full)
tasks/              Per-task evaluation (math, code, ifeval, safety, domain)
lean-sfp/           Lean 4 formalization (Dockerized build)
paper/              LaTeX source (Dockerized build)
dev/RESEARCH.md     Full research briefing
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

Submissions are benchmarked on cloud GPUs (3 seeds, ~45 min) and results appear on the [leaderboard](https://sfp-leaderboard.pages.dev).

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
