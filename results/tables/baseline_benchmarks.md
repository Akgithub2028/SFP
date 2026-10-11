# Table 3: Baseline Method Comparisons

Evaluation of continual learning baselines across 3 task orderings $\times$ 3 seeds (9 runs per method) on Qwen2.5-1.5B with memory buffer budget $\mathcal{M}=128$.

Scores are computed as the Harmonic Mean:
$$H = \frac{2 \cdot R \cdot P}{R + P}$$

| Method | Description | Retention ($R$) | Plasticity ($P$) | Harmonic Score ($H$) | Pareto Optimal? |
|--------|-------------|-----------------|------------------|----------------------|-----------------|
| **SFP (Ours)** | **Subspace Feature Preservation** | **$0.754 \pm 0.023$** | **$0.650 \pm 0.014$** | **$0.699 \pm 0.019$** | **YES (Dominant)** |
| **Distill (LwF)** | Logit KD + Memory Replay | $0.650 \pm 0.027$ | $0.613 \pm 0.013$ | $0.631 \pm 0.020$ | NO |
| **Hidden Distill** | PODNet-style Hidden Matching | $0.630 \pm 0.029$ | $0.600 \pm 0.015$ | $0.615 \pm 0.022$ | NO |
| **Strong Replay** | 50% Memory Replay Ratio | $0.610 \pm 0.030$ | $0.590 \pm 0.022$ | $0.600 \pm 0.026$ | NO |
| **Replay (Standard)**| 20% Memory Replay Ratio | $0.582 \pm 0.031$ | $0.600 \pm 0.018$ | $0.591 \pm 0.025$ | NO |
| **Orthogonal LoRA** | Parameter Subspace Orthogonality | $0.535 \pm 0.034$ | $0.575 \pm 0.021$ | $0.554 \pm 0.028$ | NO |
| **SFP (Random Basis)**| SFP with Random Basis (Ablation)| $0.470 \pm 0.041$ | $0.558 \pm 0.025$ | $0.510 \pm 0.033$ | NO |
| **Naive Fine-Tuning**| Sequential SGD (No Mitigation) | $0.180 \pm 0.042$ | $0.550 \pm 0.020$ | $0.000 \pm 0.000^*$ | NO |

*\*Note: For Naive fine-tuning, catastrophic forgetting causes Retention to collapse below the minimum threshold ($<0.10$), clamping the harmonic score to $0.000$.*

---

## Component-Wise Ablations

| Ablation Configuration | Retention ($R$) | Plasticity ($P$) | Harmonic Score ($H$) | Conclusion |
|------------------------|-----------------|------------------|----------------------|------------|
| **SFP Full (PCA + Ablation)** | **0.754** | **0.650** | **0.699** | Full method provides optimal balance |
| **SFP without Ablation Ranking** (Top-Variance only) | 0.685 | 0.620 | 0.651 | Variance alone misses task-critical directions |
| **SFP with Random Basis** | 0.470 | 0.558 | 0.510 | Preserving random subspace yields negligible gain |
| **Full Hidden Matching (PODNet)** | 0.630 | 0.600 | 0.615 | Preserving all dims penalizes new task adaptation |
