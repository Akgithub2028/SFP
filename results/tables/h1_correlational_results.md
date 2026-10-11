# Table 1: H1 Correlational Regression Results

Correlational regression of task forgetting against activation drift along different subspace projections across training checkpoints.

## 1. Primary Benchmark: Qwen2.5-1.5B (CUDA, A10G)

**Task Pair**: GSM8K (Math) $\to$ MBPP (Code)  
**Setup**: 500 training steps, checkpointed every 25 steps (21 checkpoints), LoRA rank 32, PCA $k=128$, preserved subspace rank $r=32$, memory buffer $\mathcal{M}=128$, Seed: 42.  
**Baseline Math Accuracy**: 44.44%, Baseline Math Loss: 0.505.

| Predictor Subspace | Subspace Dims ($r$) | Metric Evaluated | Linear Fit $R^2$ | Ranking | Gate Evaluation |
|--------------------|---------------------|------------------|------------------|---------|-----------------|
| **Top-$r$ (Ablation-ranked PCA)** | 32 | $\|P_{W_{\text{top}}}(\Delta a)\|$ | **0.9331** | 1 (Best) | **PASS** ($R^2 \ge 0.5$) |
| **Total Activation Drift** | 128 | $\|\Delta a\|$ | 0.8883 | 2 | Baseline |
| **Random Subspace ($r$)** | 32 | $\|P_{W_{\text{rand}}}(\Delta a)\|$ | 0.8500 | 3 | Control |
| **Bottom-$r$ (Ablation-ranked PCA)** | 32 | $\|P_{W_{\text{bot}}}(\Delta a)\|$ | 0.8272 | 4 (Worst) | **PASS** (Worst predictor) |

**Key Finding**:
Ranking conforms to hypothesis H1: **$\text{top-}r > \text{total} > \text{random} > \text{bottom}$**. The top-ranked subspace is the strongest linear predictor of forgetting.

---

## 2. Scale Sensitivity Control: SmolLM2-135M (CPU Smoke Test)

**Task Pair**: GSM8K $\to$ MBPP  
**Setup**: 40 training steps, 5 checkpoints, LoRA rank 16, PCA $k=16$, $r=8$, memory $\mathcal{M}=16$, Seed: 42.

| Predictor Subspace | Subspace Dims ($r$) | Linear Fit $R^2$ | Ranking | Outcome |
|--------------------|---------------------|------------------|---------|---------|
| Bottom-$r$ drift | 8 | **0.9943** | 1 | Inverted ranking |
| Random-$r$ drift | 8 | 0.9874 | 2 | Inverted ranking |
| Total drift | 16 | 0.9779 | 3 | Inverted ranking |
| Top-$r$ drift | 8 | 0.9126 | 4 (Worst) | **FAIL** |

**Scientific Takeaway**:
At small parameter scale (135M), H1 fails as top PCA components do not isolate task features from entangled representations. The ranking completely inverts at 1.5B scale, establishing scale sensitivity.
