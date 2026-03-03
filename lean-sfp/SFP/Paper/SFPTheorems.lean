/-
  SFP/Paper/SFPTheorems.lean

  The complete set of formal statements referenced in the paper.
  Each theorem is proved in the corresponding module.

  Status:
    ✅ = fully proved (no sorry)
    🔶 = type-checks but proof uses sorry

  ## Projection Properties (§3.1)
  - ✅ `orthogonalProjection_idempotent`: P² = P
  - ✅ `orthogonalProjection_selfAdjoint`: ⟨Px,y⟩ = ⟨x,Py⟩
  - ✅ `orthogonalProjection_norm_le`: ‖Px‖ ≤ ‖x‖
  - ✅ `orthogonalProjection_complement`: (I-P)x = P_{W⊥}x
  - 🔶 `orthonormal_projection_formula`: Px = Σᵢ ⟨x,uᵢ⟩uᵢ

  ## Preservation Loss (§3.2)
  - ✅ `preservationLoss_nonneg`: L_pres ≥ 0
  - ✅ `preservationLoss_eq_zero_iff`: L_pres = 0 ↔ P(a-a*) = 0
  - ✅ `subspace_drift_bound`: ‖P(a-a*)‖ = √L_pres
  - ✅ `preservationLoss_le_total_drift`: L_pres ≤ ‖a-a*‖²
  - ✅ `drift_pythagorean_decomposition`: ‖Δa‖² = L_pres + ‖P⊥(Δa)‖²
  - ✅ `multiLayerPreservationLoss_nonneg`: multi-layer loss ≥ 0

  ## Gradient Orthogonality (§3.3)
  - ✅ `gradient_orthogonality`: P(∇L_new) = 0 at preserved stationary point
  - ✅ `gradient_in_complement`: ∇L_new ∈ W⊥ at preserved stationary point
  - ✅ `gradient_projection_at_stationary`: P(∇L_new) = -2λv (general case)

  ## Main Reduction (§3.4)
  - ✅ `sfp_reduces_forgetting`: H1 + L_pres≤δ → forgetting ≤ C√δ
  - ✅ `sfp_reduces_forgetting_multilayer`: per-layer version
  - ✅ `stability_plasticity_tradeoff`: stability bound + plasticity lower bound
-/

import SFP.LinearAlgebra.Projections
import SFP.Loss.Preservation
import SFP.Loss.GradientOrthogonality
import SFP.Paper.Reduction
