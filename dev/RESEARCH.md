# SFP Research Briefing

*Consolidated from design discussions. Read this to get up to speed in 30 minutes.*

## Problem

LLMs are continually updated: alignment tuning, safety patches, domain adaptation, personalization. Sequential updates cause **capability regressions** — performance on previously validated behaviors degrades. In the continual learning literature this is called catastrophic forgetting.

This is distinct from the "retrain from scratch" paradigm most production LLM pipelines use. We focus on the setting where a model is updated *in place* over a sequence of tasks — relevant for personalization, on-device adaptation, and rapid iteration cycles where full retraining is too expensive.

## Why forgetting happens

Neural networks store knowledge in distributed, entangled weight patterns. There are no "task slots." When you update weights for Task B, nothing constrains the updates to preserve Task A performance. If gradients conflict (∇L_A · ∇L_B < 0), Task A degrades.

## Existing approaches (and why they're insufficient)

| Method | What it preserves | Limitation |
|--------|-------------------|------------|
| EWC (Kirkpatrick 2017) | Important weights | Global; doesn't know *what* weights encode |
| LwF (Li & Hoiem 2017) | Output logits | Doesn't preserve internal representations |
| PODNet (Douillard 2020) | Hidden states (full) | Preserves too much; hurts plasticity |
| Orthogonal LoRA | Parameter directions | Geometry-based, not feature-based |
| Replay | Data samples | Relies on data availability |

None answer: **where inside the model does forgetting actually happen?**

## Our hypothesis

**H1**: Measurable regressions are **predictable from low-dimensional activation drift**.
- Drift in top-r PCA coordinates at mid-depth layers predicts old-task accuracy drops (R² ≥ 0.6).
- Drift in random-r coordinates does not (R² ≈ 0).
- This must hold after controlling for total drift ‖Δa‖ (to rule out "drift = drift" confound).

**H2**: Preserving only high-importance features improves the stability-plasticity Pareto frontier.
- SFP dominates ≥2 baselines at small memory budgets.

**H3**: Preserved features are mechanistically coherent (interpretable, causally linked to behavior).

## Method: Sparse Feature Preservation (SFP)

1. **Hook activations** at mid-layers (depth/4, depth/2, 3*depth/4)
2. **Build PCA basis** from memory set activations
3. **Rank importance** by ablation: zero each PC dim, measure old-task drop
4. **Select top-r** dimensions
5. **Preservation loss**: L = L_new + λ || U_r^T a(θ) − U_r^T a(θ_anchor) ||²

That's it. ~40 lines of code.

## Key prior work

- Kirkpatrick et al., 2017 — Elastic Weight Consolidation (EWC)
- Li & Hoiem, 2017 — Learning without Forgetting (LwF)
- Douillard et al., 2020 — PODNet (ECCV)
- Wang et al., 2024 — Continual Learning for LLMs (arXiv:2404.16789)
- "Continual Learning meets Mechanistic Interpretability" (Jan 2026, arXiv)
- Sennrich et al., 2015 — BPE for NLP

## Evaluation (NeurIPS-grade)

- **LiveBench** — contamination-resistant, objective scoring (arXiv:2406.19314)
- **IFEval** — automatically verifiable instruction-following (arXiv:2311.07911)
- **SWE-bench Verified** — real GitHub issue patching (arXiv:2310.06770)
- **GSM8K** — math reasoning (secondary)
- **HumanEval** — code generation (secondary)

## Risks

1. **LoRA confound**: Low-rank adapters constrain updates to a low-dimensional subspace by construction. "Forgetting is low-dimensional" might be an artifact of the parameter-efficient training, not a property of forgetting. *Mitigation*: validate H1 under full fine-tuning at least once.
2. **Vacuous bound via C**: The reduction theorem gives forgetting ≤ C·√δ but C is estimated empirically. If C is huge or unstable, the bound is meaningless. *Mitigation*: report C distributions across seeds/tasks; show they're bounded and stable.
3. **Layer cherry-picking**: Testing H1 at depth/4, depth/2, 3·depth/4 and reporting the best result is p-hacking. *Mitigation*: pre-register probe layers or report all layers.
4. **Forgetting not low-dimensional** → try different layers/basis; report negative result honestly.
5. **Effect disappears at scale** → validate on mid-scale (1.5B) first.
6. **"Just distillation" criticism** → random-basis and random-selection controls prove PCA + importance matter beyond generic representation anchoring.

## Novelty wedge

The general idea "anchor internal activations" is NOT new (PODNet exists).
What's new:
1. **Selective** preservation: evidence that a small subset (~32 dims/layer) suffices, with ablation controls showing PCA + importance ranking outperform random subspaces.
2. **Falsifiable measurement protocol**: the H1 regression test is a reproducible claim others can validate or falsify on their own models/tasks.
3. **LLM continual instruction-tuning** setting with strong baselines (not just vision CL benchmarks).
4. **Formal verification** of the mathematical reduction — prevents hand-wavy algebra, makes assumptions explicit (not a "guarantee").
5. Potential **mechanistic link** to interpretability (if SAE features outperform PCA — future work).
