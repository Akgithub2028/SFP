# SFP Research Briefing

*Consolidated from design discussions. Read this to get up to speed in 30 minutes.*

## Problem

LLMs are continually fine-tuned: new domains, new alignment, new capabilities. Sequential fine-tuning causes **catastrophic forgetting** — performance on earlier tasks degrades as new tasks are trained.

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

**H1**: Forgetting is **low-dimensional** — concentrated in a small feature subspace.
- Drift in top-r feature coordinates predicts forgetting (R² ≥ 0.6).
- Drift in random-r coordinates does not.

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

1. **Forgetting not low-dimensional** → try different layers/basis; report negative result
2. **Effect disappears at scale** → validate on mid-scale first
3. **"Just distillation" criticism** → random-basis and random-selection controls prove PCA + importance matter

## Novelty wedge

The general idea "anchor internal activations" is NOT new (PODNet exists).
What's new: (1) feature-like basis (PCA/SAE), (2) evidence that only a *small* set matters, (3) LLM continual instruction-tuning setting with strong baselines, (4) mechanistic link to interpretability.
