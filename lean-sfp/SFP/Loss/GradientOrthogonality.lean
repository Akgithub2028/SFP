/-
  SFP/Loss/GradientOrthogonality.lean

  The gradient orthogonality theorem for SFP.
  Paper reference: Theorem 3 (Gradient Orthogonality).
-/

import Mathlib.Analysis.InnerProductSpace.Projection
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Analysis.Calculus.Gradient.Basic

open scoped InnerProductSpace

namespace SFP

section GradientOrthogonality

variable {d : ℕ}

/-- **Gradient Orthogonality Theorem** (abstract version):
    If the gradient of L_new plus a penalty term vanishes, and the
    preservation constraint P(a - a*) = 0 holds, then the L_new gradient
    is orthogonal to the preserved subspace: P(∇L_new) = 0.

    We state this algebraically to avoid technicalities of `gradient`
    on specific types. The key insight: once preserved features are anchored,
    any remaining learning update must lie in the complement.

    Paper reference: Theorem 3. -/
theorem gradient_orthogonality
    {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    [CompleteSpace E]
    (W : Submodule ℝ E) [CompleteSpace W]
    (grad_L_new : E)
    (v : E) -- v = P(a - a*)
    (lam : ℝ) (hlam : 0 < lam)
    (hv_mem : v ∈ (W : Set E))
    -- Stationary: ∇L_new + 2λ v = 0
    (h_stationary : grad_L_new + (2 * lam) • v = 0)
    -- Preservation: P(a - a*) = 0, i.e., v = 0
    (h_preserved : v = 0) :
    (orthogonalProjection W grad_L_new : E) = 0 := by
  have : grad_L_new = 0 := by
    have := h_stationary
    rw [h_preserved] at this
    simp at this
    exact this
  simp [this, map_zero]

/-- **Corollary**: Under the same conditions, ∇L_new ∈ W⊥.
    Paper reference: Corollary 1. -/
theorem gradient_in_complement
    {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    [CompleteSpace E]
    (W : Submodule ℝ E) [CompleteSpace W]
    (grad_L_new : E)
    (v : E)
    (lam : ℝ) (hlam : 0 < lam)
    (hv_mem : v ∈ (W : Set E))
    (h_stationary : grad_L_new + (2 * lam) • v = 0)
    (h_preserved : v = 0) :
    grad_L_new ∈ Wᗮ := by
  have : grad_L_new = 0 := by
    rw [h_preserved] at h_stationary
    simp at h_stationary
    exact h_stationary
  rw [this]
  exact Submodule.zero_mem _

/-- **Stronger version**: Without assuming P(a-a*)=0, the gradient of L_new
    at a stationary point has its W-component equal to -2λP(a-a*).
    This shows the penalty *continuously pushes* the gradient out of W.
    Paper reference: Proposition 4. -/
theorem gradient_projection_at_stationary
    {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    [CompleteSpace E]
    (W : Submodule ℝ E) [CompleteSpace W]
    (grad_L_new : E)
    (v : E)
    (lam : ℝ)
    (hv_mem : v ∈ (W : Set E))
    (h_stationary : grad_L_new + (2 * lam) • v = 0) :
    (orthogonalProjection W grad_L_new : E) = -(2 * lam) • v := by
  have hgrad : grad_L_new = -(2 * lam) • v := by
    have h := h_stationary
    have : grad_L_new = -((2 * lam) • v) := eq_neg_of_add_eq_zero_left h
    simp only [neg_smul] at this ⊢
    exact this
  rw [hgrad]
  have hmem : -(2 * lam) • v ∈ (W : Set E) := Submodule.smul_mem W _ hv_mem
  exact orthogonalProjection_eq_self_iff.mpr hmem

end GradientOrthogonality

end SFP
