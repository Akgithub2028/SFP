/-
  SFP/Loss/Preservation.lean

  Formalization of the SFP preservation loss and its key properties.
-/

import Mathlib.Analysis.InnerProductSpace.Projection
import SFP.LinearAlgebra.Projections

open scoped InnerProductSpace

namespace SFP

section PreservationLoss

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
variable [CompleteSpace E]
variable (W : Submodule ℝ E) [CompleteSpace W]

/-- The SFP preservation loss for a single layer.
    Paper reference: Definition 3. -/
noncomputable def preservationLoss (a a_star : E) : ℝ :=
  ‖(orthogonalProjection W (a - a_star) : E)‖ ^ 2

/-- Preservation loss is nonneg. Paper reference: Proposition 2(i). -/
theorem preservationLoss_nonneg (a a_star : E) :
    0 ≤ preservationLoss W a a_star := by
  unfold preservationLoss; positivity

/-- L_pres = 0 ↔ P(a-a*) = 0. Paper reference: Proposition 2(ii). -/
theorem preservationLoss_eq_zero_iff (a a_star : E) :
    preservationLoss W a a_star = 0 ↔
    (orthogonalProjection W (a - a_star) : E) = 0 := by
  unfold preservationLoss
  rw [sq_eq_zero_iff, norm_eq_zero]

/-- ‖P(a-a*)‖ = √L_pres. Paper reference: Theorem 1. -/
theorem subspace_drift_bound (a a_star : E) :
    ‖(orthogonalProjection W (a - a_star) : E)‖ =
    Real.sqrt (preservationLoss W a a_star) := by
  unfold preservationLoss
  rw [Real.sqrt_sq (norm_nonneg _)]

/-- L_pres ≤ ‖a-a*‖². Paper reference: Proposition 3. -/
theorem preservationLoss_le_total_drift (a a_star : E) :
    preservationLoss W a a_star ≤ ‖a - a_star‖ ^ 2 := by
  unfold preservationLoss
  have h := SFP.orthogonalProjection_norm_le W (a - a_star)
  nlinarith [norm_nonneg (orthogonalProjection W (a - a_star) : E), norm_nonneg (a - a_star)]

/-- ‖Δa‖² = L_pres + ‖P⊥(Δa)‖² (Pythagorean).
    Paper reference: Theorem 2 (stability-plasticity decomposition). -/
theorem drift_pythagorean_decomposition (a a_star : E) :
    ‖a - a_star‖ ^ 2 =
    preservationLoss W a a_star +
    ‖(orthogonalProjection Wᗮ (a - a_star) : E)‖ ^ 2 := by
  unfold preservationLoss
  have h := norm_sq_eq_add_norm_sq_projection (a - a_star) W
  simp only [sq] at h ⊢
  simp only [Submodule.coe_norm] at h
  linarith

end PreservationLoss

section MultiLayer

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
variable [CompleteSpace E]

/-- Multi-layer preservation loss. Paper reference: Equation (3). -/
noncomputable def multiLayerPreservationLoss
    {L : ℕ} (W : Fin L → Submodule ℝ E) [∀ l, CompleteSpace (W l)]
    (a a_star : Fin L → E) : ℝ :=
  ∑ l : Fin L, preservationLoss (W l) (a l) (a_star l)

/-- Multi-layer loss is nonneg. -/
theorem multiLayerPreservationLoss_nonneg
    {L : ℕ} (W : Fin L → Submodule ℝ E) [∀ l, CompleteSpace (W l)]
    (a a_star : Fin L → E) :
    0 ≤ multiLayerPreservationLoss W a a_star := by
  unfold multiLayerPreservationLoss
  apply Finset.sum_nonneg
  intro l _
  exact preservationLoss_nonneg (W l) (a l) (a_star l)

/-- Total SFP loss: L = L_new + λ·L_pres. Paper reference: Equation (4). -/
noncomputable def sfpTotalLoss
    {L : ℕ} (W : Fin L → Submodule ℝ E) [∀ l, CompleteSpace (W l)]
    (lam : ℝ) (loss_new : ℝ) (a a_star : Fin L → E) : ℝ :=
  loss_new + lam * multiLayerPreservationLoss W a a_star

end MultiLayer

end SFP
