# Table 2: H1 Causal Intervention Test Results

Causal intervention test injecting Gaussian perturbation noise $\eta \sim \mathcal{N}(0, \sigma^2 I)$ into selected activation subspaces during inference on the anchor task.

## 1. Primary Benchmark: Qwen2.5-1.5B (CUDA, A10G)

**Model**: Qwen/Qwen2.5-1.5B-Instruct  
**PCA Dim**: $k=128$, **Preserved Subspace Rank**: $r=32$, **Memory**: $\mathcal{M}=128$, **Evaluation Samples**: 50  
**Probed Layers**: Layer 7 ($L/4$), Layer 14 ($L/2$), Layer 21 ($3L/4$)  
**Baseline Anchor Perplexity**: 1.657 (Loss: 0.505)

| Noise Magnitude ($\sigma$) | Top-$r$ $\Delta\text{PPL}$ | Random-$r$ $\Delta\text{PPL}$ | Bottom-$r$ $\Delta\text{PPL}$ | Causal Ratio ($\frac{\text{Top}}{\text{Random}}$) | Causal Ratio ($\frac{\text{Top}}{\text{Bottom}}$) | Gate Verification |
|-----------------------------|----------------------------|-------------------------------|--------------------------------|---------------------------------------------------|---------------------------------------------------|-------------------|
| **$0.5\sigma$** | +0.042 | +0.024 | +0.001 | **$1.8\times$** | **$51\times$** | **PASS** ($\text{Top} > \text{Rand} > \text{Bot}$) |
| **$1.0\sigma$** | **+0.248** | +0.100 | +0.009 | **$2.5\times$** | **$28\times$** | **PASS** |
| **$2.0\sigma$** | **+2.604** | +0.810 | +0.041 | **$3.2\times$** | **$64\times$** | **PASS** |
| **$5.0\sigma$** | **+8.709** | +5.041 | +0.328 | **$1.7\times$** | **$27\times$** | **PASS** |

### Key Observations:
1. **Consistent Monotonic Ranking**: At all noise scales, $\Delta\text{PPL}(\text{Top}) > \Delta\text{PPL}(\text{Random}) > \Delta\text{PPL}(\text{Bottom})$ strictly holds.
2. **Superlinear Degradation**: The impact of perturbing top-$r$ features scales superlinearly with noise magnitude, reaching a $64\times$ differential over bottom features at $2.0\sigma$.
3. **Inertness of Bottom Features**: Bottom-$r$ dimensions can absorb substantial perturbations ($5.0\sigma$) with almost negligible impact on task perplexity ($+0.328$), proving that representation drift is not uniformly harmful across dimensions.

---

## 2. SmolLM2-135M Causal Test (CPU Control)

**Model**: SmolLM2-135M-Instruct  
**PCA Dim**: $k=4$, $r=2$, memory $\mathcal{M}=8$, eval samples: 4, $\sigma = 1.0$.

| Intervention Subspace | Perplexity (PPL) | $\Delta\text{PPL}$ | Relative Degrade |
|-----------------------|------------------|-------------------|------------------|
| **Top-$r$ ($1.0\sigma$)** | 1.9281 | **+0.8548** | **$4.1\times$ over random** |
| **Random-$r$ ($1.0\sigma$)** | 1.2800 | +0.2068 | Baseline control |
| **Bottom-$r$ ($1.0\sigma$)** | 1.0931 | +0.0198 | Negligible effect |

**Takeaway**:
In contrast to the correlational regression (which failed at 135M), the **causal intervention test succeeds across both scales** (135M and 1.5B). Ablation ranking accurately isolates causally active directions even when small-scale correlational tracking is noisy.
