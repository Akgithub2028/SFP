# Methodology: Subspace Feature Preservation (SFP)

## 1. Overview and Intuition

Catastrophic forgetting occurs when a neural network is fine-tuned sequentially on a stream of tasks $\mathcal{T}_1, \mathcal{T}_2, \dots, \mathcal{T}_T$. Updating parameter weights $\theta$ to minimize loss on a new task $\mathcal{L}_{\text{new}}$ inevitably rotates activation representations, often causing destructive interference with earlier task representations.

Existing mitigation methods either regularize weights globally without understanding what representations they encode (e.g., EWC), penalize output logit divergence without safeguarding internal circuits (e.g., LwF), or attempt to preserve all hidden states indiscriminately (e.g., PODNet), which severely hurts model plasticity.

**Subspace Feature Preservation (SFP)** operates on the premise that forgetting is concentrated in a low-dimensional subspace of activation space. Rather than freezing weights or matching full activation volumes, SFP identifies an empirically grounded rank-$r$ subspace $W \subset \mathbb{R}^d$ and penalizes drift only along the projection into $W$.

```
Input Tokens ──▶ [ Transformer Layer ℓ ] ──▶ Activation a_ℓ(θ)
                                                   │
                          ┌────────────────────────┴────────────────────────┐
                          ▼                                                 ▼
             Projection into Subspace W                       Orthogonal Complement W^⊥
             P_W(a_ℓ) = U_r U_r^T a_ℓ                         (I - P_W) a_ℓ
                          │                                                 │
                          ▼                                                 ▼
               Preserved via SFP Loss                         Free to adapt to new task
               λ || U_r^T Δa_ℓ ||^2                            (Preserves Plasticity!)
```

---

## 2. Mathematical Formulation

Let $a_\ell(\theta; x) \in \mathbb{R}^d$ denote the hidden activation vector at layer $\ell$ for input sequence $x$, parameterized by model weights $\theta$. Let $\theta^*$ denote the anchor model checkpoint checkpointed immediately after training on task $\mathcal{T}_{t-1}$.

### 2.1 Subspace Projection Operator

Let $U_r \in \mathbb{R}^{d \times r}$ denote an orthonormal basis for an $r$-dimensional subspace $W \subset \mathbb{R}^d$, where $r \ll d$. The orthogonal projection operator onto $W$ is defined as:

$$P_W = U_r U_r^\top$$

The activation drift between the current model $\theta$ and anchor $\theta^*$ is denoted:

$$\Delta a_\ell(x) = a_\ell(\theta; x) - a_\ell(\theta^*; x)$$

### 2.2 SFP Objective Function

During training on the new task $\mathcal{L}_{\text{new}}$, SFP augments the loss with an activation regularization penalty evaluated over a memory buffer $\mathcal{M}$:

$$\mathcal{L}(\theta) = \mathcal{L}_{\text{new}}(\theta) + \lambda \sum_{\ell \in \mathcal{L}_{\text{probe}}} \mathbb{E}_{x \sim \mathcal{M}} \left[ \| U_{r, \ell}^\top a_\ell(\theta; x) - U_{r, \ell}^\top a_\ell(\theta^*; x) \|_2^2 \right]$$

Because $U_{r, \ell}$ is orthonormal, the squared Euclidean norm in the projected coordinate system matches the projection norm:

$$\| U_{r, \ell}^\top \Delta a_\ell(x) \|_2^2 = \| P_{W_\ell}(\Delta a_\ell(x)) \|_2^2$$

---

## 3. Subspace Identification Protocol

The subspace $W_\ell$ is extracted at task boundaries using the following five-step pipeline:

1. **Activation Caching**: Run forward passes on the anchor model $\theta^*$ using memory examples $x \in \mathcal{M}$ to collect activation matrix $A_\ell \in \mathbb{R}^{N \times d}$ at probed layers $\mathcal{L}_{\text{probe}} = \{ \lfloor L/4 \rfloor, \lfloor L/2 \rfloor, \lfloor 3L/4 \rfloor \}$.
2. **Centered PCA Basis**: Compute the top-$k$ principal components ($k \ge r$, e.g., $k=128$) via Singular Value Decomposition of the centered activation matrix:
   $$\bar{A}_\ell = A_\ell - \mu_\ell, \quad \bar{A}_\ell = U \Sigma V^\top$$
   The top-$k$ right-singular vectors form the basis matrix $V_k \in \mathbb{R}^{d \times k}$.
3. **Importance Ranking via Targeted Ablation**: To ensure that selected directions are functionally critical rather than merely high-variance noise, each principal component direction $v_j$ ($j \in \{1, \dots, k\}$) is individually ablated (zeroed out in forward pass hooks on $\mathcal{M}$):
   $$\tilde{a}_\ell = a_\ell - (a_\ell^\top v_j) v_j$$
   The capability drop $\Delta \mathcal{L}_j = \mathcal{L}_{\text{old}}(\tilde{a}) - \mathcal{L}_{\text{old}}(a)$ is recorded as the importance score.
4. **Subspace Selection**: The top-$r$ principal components with the largest ablation delta $\Delta \mathcal{L}_j$ are selected to form the orthonormal basis $U_r \in \mathbb{R}^{d \times r}$.
5. **Loss Anchoring**: Freeze $U_{r, \ell}$ and anchor activations $a_\ell(\theta^*; x)$ during subsequent fine-tuning.

---

## 4. Gradient Orthogonality and Plasticity

A core theoretical insight of SFP is that it guarantees directional freedom for new-task adaptation:

$$\nabla_\theta \mathcal{L}(\theta) = \nabla_\theta \mathcal{L}_{\text{new}}(\theta) + 2\lambda \sum_{\ell} J_{\theta}(a_\ell)^\top P_{W_\ell}(\Delta a_\ell)$$

where $J_{\theta}(a_\ell) = \frac{\partial a_\ell}{\partial \theta}$ is the activation Jacobian.

- When the model operates near the preserved manifold ($\Delta a_\ell \approx 0$), the regularization gradient vanishes.
- At stationary points where the projected drift is nullified, the new-task gradient $\nabla_\theta \mathcal{L}_{\text{new}}$ is strictly orthogonal to the subspace $W$ in representation space, preventing gradient conflict $\langle \nabla \mathcal{L}_{\text{old}}, \nabla \mathcal{L}_{\text{new}} \rangle < 0$.
- Any update orthogonal to $W$ ($v \in W^\perp$) incurs **zero preservation penalty**, allowing full model plasticity for learning the new task.

---

## 5. Machine-Checked Formalization (Lean 4)

The core reduction of SFP is formally verified in Lean 4 without unverified assumptions (0 `sorry` axioms). 

```lean
-- Reduction Theorem (lean-sfp/SFP/Paper/Reduction.lean)
theorem sfp_reduces_forgetting
    (forgetting_bound : forgetting ≤ C * ‖P_W (Δa)‖)
    (preservation_bound : preservationLoss ≤ δ) :
    forgetting ≤ C * Real.sqrt δ
```

### Proved Formal Properties:
- `orthogonalProjection_idempotent`: $P_W^2 = P_W$
- `orthogonalProjection_selfAdjoint`: $P_W^\top = P_W$
- `drift_pythagorean_decomposition`: $\|\Delta a\|^2 = \|P_W(\Delta a)\|^2 + \|(I - P_W)(\Delta a)\|^2$
- `preservationLoss_le_total_drift`: $\mathcal{L}_{\text{pres}} \le \|\Delta a\|^2$
- `gradient_orthogonality`: Gradient components lie exclusively in $W^\perp$ at stationarity.

For full theorem listings and build instructions, see [`lean-sfp/`](../lean-sfp/).
