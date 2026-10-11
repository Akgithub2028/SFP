# Figure: Dimensionality Sweep Diagnostic (r-Sweep)

Diagnostic sweep evaluating the relationship between preserved subspace rank $r$, activation variance explained, and forgetting predictability $R^2$ on Qwen2.5-1.5B.

## Empirical Data

| Preserved Rank ($r$) | Cumulative Variance Explained (%) | Forgetting Fit $R^2$ | Marginal Variance Gain (%) | Marginal $R^2$ Delta |
|----------------------|-----------------------------------|----------------------|----------------------------|----------------------|
| **1** | 96.2% | 0.854 | 96.2% | +0.854 |
| **2** | 96.4% | 0.854 | +0.2% | +0.000 |
| **4** | 96.9% | 0.871 | +0.5% | +0.017 |
| **8** | 97.4% | 0.878 | +0.5% | +0.007 |
| **16** | 98.0% | 0.879 | +0.6% | +0.001 |
| **32** | 98.7% | 0.882 | +0.7% | +0.003 |
| **64** | 99.4% | 0.883 | +0.7% | +0.001 |
| **128 (Total)** | 100.0% | 0.884 | +0.6% | +0.001 |

## Visualization

```
100% ┼─────────────────────────────────────────────────── Var Explained
     │  ●      ●       ●       ●       ●       ●       ●
 90% ┼─────────────────────────────────────────────────── Forgetting R²
     │  ▲      ▲       ▲       ▲       ▲       ▲       ▲
 80% ┼
     │
     └───────┬───────┬───────┬───────┬───────┬───────┬───► Subspace Rank r
            r=1     r=4     r=8    r=16    r=32    r=64   r=128
```

### Key Takeaway:
The very first principal component ($r=1$) accounts for **96.2% of activation variance** and captures **$R^2 = 0.854$ of the forgetting signal**. Increasing rank from $r=32$ to $r=128$ yields virtually no marginal increase in forgetting predictability ($+0.002$), while quadrupling the regularization constraint on model parameters. This provides strong empirical justification for setting $r \in [16, 32]$.
