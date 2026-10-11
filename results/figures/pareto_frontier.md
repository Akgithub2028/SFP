# Figure: Stability-Plasticity Pareto Frontier

Visualization of the trade-off between Stability (Retention, x-axis) and Plasticity (New-Task Learning, y-axis) on Qwen2.5-1.5B with memory buffer budget $\mathcal{M}=128$.

```
Plasticity (P)
  ▲
0.70│                                      
    │                                
0.65│                                ★ SFP (0.754, 0.650) [Pareto Optimal]
    │                          
0.60│                    ◆ Distill (0.650, 0.613)
    │               ▲ Replay (0.582, 0.600)   ■ Hidden Distill (0.630, 0.600)
    │           ● Strong Replay (0.610, 0.590)
0.55│       ▼ Orthogonal LoRA (0.535, 0.575)
    │   ○ SFP Random Basis (0.470, 0.558)
    │
0.50│ ✕ Naive (0.180, 0.550)
    │
0.00└────────────────────────────────────────────────────────► Retention (R)
    0.10     0.30     0.50     0.60     0.70     0.80
```

### Analysis:
- **Naive Fine-Tuning** exhibits severe catastrophic forgetting ($R = 0.180$), leaving old tasks completely degraded.
- **Hidden Distillation (PODNet)** over-regularizes activation representations, improving retention ($0.630$) but constraining plasticity ($0.600$).
- **SFP** anchors only the top-$r=32$ functional directions, preserving old task capabilities ($R = 0.754$) while freeing the orthogonal complement $W^\perp$ for plasticity ($P = 0.650$). SFP strictly Pareto-dominates all baselines.
