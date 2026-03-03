/-
  SFP/LinearAlgebra/Projections.lean

  Orthogonal projections and their properties, forming the mathematical
  foundation for Sparse Feature Preservation.
-/

import Mathlib.Analysis.InnerProductSpace.Projection
import Mathlib.Analysis.InnerProductSpace.PiL2

open scoped InnerProductSpace

abbrev V (d : ℕ) := EuclideanSpace ℝ (Fin d)

namespace SFP

section ProjectionProperties

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
variable [CompleteSpace E]
variable (W : Submodule ℝ E) [CompleteSpace W]

/-- Orthogonal projection is idempotent: P(Px) = Px.
    Paper reference: Proposition 1(i). -/
theorem orthogonalProjection_idempotent (x : E) :
    orthogonalProjection W (orthogonalProjection W x : E) =
    orthogonalProjection W x := by
  apply Subtype.ext
  simp [orthogonalProjection_mem_subspace_eq_self]

/-- Orthogonal projection is self-adjoint: ⟨Px, y⟩ = ⟨x, Py⟩.
    Paper reference: Proposition 1(ii). -/
theorem orthogonalProjection_selfAdjoint (x y : E) :
    ⟪(orthogonalProjection W x : E), y⟫_ℝ =
    ⟪x, (orthogonalProjection W y : E)⟫_ℝ := by
  have hPx_mem : (orthogonalProjection W x : E) ∈ (W : Set E) :=
    (orthogonalProjection W x).2
  have hPy_mem : (orthogonalProjection W y : E) ∈ (W : Set E) :=
    (orthogonalProjection W y).2
  have hx_diff : x - (orthogonalProjection W x : E) ∈ Wᗮ :=
    sub_orthogonalProjection_mem_orthogonal x
  have hy_diff : y - (orthogonalProjection W y : E) ∈ Wᗮ :=
    sub_orthogonalProjection_mem_orthogonal y
  have h1 : ⟪(orthogonalProjection W x : E), y - (orthogonalProjection W y : E)⟫_ℝ = 0 :=
    Submodule.inner_right_of_mem_orthogonal hPx_mem hy_diff
  have h2 : ⟪x - (orthogonalProjection W x : E), (orthogonalProjection W y : E)⟫_ℝ = 0 :=
    Submodule.inner_left_of_mem_orthogonal hPy_mem hx_diff
  rw [inner_sub_right] at h1
  rw [inner_sub_left] at h2
  linarith

/-- Orthogonal projection is nonexpansive: ‖Px‖ ≤ ‖x‖.
    Paper reference: Proposition 1(iii). -/
theorem orthogonalProjection_norm_le (x : E) :
    ‖(orthogonalProjection W x : E)‖ ≤ ‖x‖ := by
  -- Use Pythagorean theorem: ‖x‖² = ‖Px‖² + ‖x - Px‖²
  have h_ortho : ⟪(orthogonalProjection W x : E),
    x - (orthogonalProjection W x : E)⟫_ℝ = 0 :=
    Submodule.inner_right_of_mem_orthogonal (orthogonalProjection W x).2
      (sub_orthogonalProjection_mem_orthogonal x)
  -- x = Px + (x - Px)
  have h_add : x = (orthogonalProjection W x : E) + (x - (orthogonalProjection W x : E)) := by
    simp
  have h_norm_sq : ‖x‖ * ‖x‖ =
    ‖(orthogonalProjection W x : E)‖ * ‖(orthogonalProjection W x : E)‖ +
    ‖x - (orthogonalProjection W x : E)‖ * ‖x - (orthogonalProjection W x : E)‖ := by
    conv_lhs => rw [h_add]
    exact norm_add_sq_eq_norm_sq_add_norm_sq_of_inner_eq_zero _ _ h_ortho
  -- ‖Px‖² ≤ ‖Px‖² + ‖x-Px‖² = ‖x‖², so ‖Px‖ ≤ ‖x‖
  have h_le_sq : ‖(orthogonalProjection W x : E)‖ * ‖(orthogonalProjection W x : E)‖ ≤
    ‖x‖ * ‖x‖ := by nlinarith [mul_self_nonneg ‖x - (orthogonalProjection W x : E)‖]
  nlinarith [sq_nonneg (‖x‖ - ‖(orthogonalProjection W x : E)‖),
    norm_nonneg (orthogonalProjection W x : E), norm_nonneg x]

/-- The complement projection: x - Px = P_{W⊥}x.
    Formalizes the "ablation" operation.
    Paper reference: Definition 2 (ablation operator). -/
theorem orthogonalProjection_complement (x : E) :
    x - (orthogonalProjection W x : E) =
    (orthogonalProjection Wᗮ x : E) := by
  symm
  apply eq_orthogonalProjection_of_mem_of_inner_eq_zero
  · exact sub_orthogonalProjection_mem_orthogonal x
  · intro w hw
    -- Need: ⟪x - (x - Px), w⟫ = 0 for w ∈ Wᗮ, i.e. ⟪Px, w⟫ = 0
    simp only [sub_sub_cancel]
    exact Submodule.inner_right_of_mem_orthogonal (orthogonalProjection W x).2 hw

end ProjectionProperties

section OrthonormalBasis

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
variable [FiniteDimensional ℝ E]

/-- For an orthonormal family, Px = Σᵢ ⟨x, uᵢ⟩ uᵢ.
    Corresponds to projected = current_acts @ u_r in features.py.
    Paper reference: Equation (1). -/
theorem orthonormal_projection_formula
    {r : ℕ} (u : Fin r → E)
    (hu : Orthonormal ℝ u) (x : E) :
    (orthogonalProjection (Submodule.span ℝ (Set.range u))
      x : E) =
    ∑ i : Fin r, ⟪x, u i⟫_ℝ • u i := by
  classical
  simpa using hu.orthogonalProjection_eq_sum (x := x)

end OrthonormalBasis

end SFP
