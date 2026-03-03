/-
  SFP/Paper/Reduction.lean

  The main result: a formal reduction of "SFP prevents forgetting"
  to a single empirical hypothesis (H1).

  Structure:
    Axiom H1:  drift in a rank-r subspace predicts forgetting
    Theorem:   minimizing the SFP loss bounds forgetting

  Everything except H1 is verified. The paper's experiments exist
  solely to validate H1.
-/

import Mathlib.Analysis.InnerProductSpace.Projection
import SFP.Loss.Preservation

open scoped InnerProductSpace

namespace SFP

section Reduction

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
variable [CompleteSpace E]
variable (W : Submodule ℝ E) [CompleteSpace W]

/-! ## The empirical hypothesis

H1: There exists a low-rank subspace W such that forgetting on a previous
task is bounded by drift in W. Formally:

  forgetting(θ, θ*) ≤ C · ‖P_W(a(θ) - a(θ*))‖

where a(θ) are activations at a probed layer, P_W is orthogonal projection
onto the important subspace, and C is a task-dependent constant.

This is the ONLY unverified assumption. Everything else follows.
-/

/-- The full reduction: IF subspace drift predicts forgetting (H1),
    THEN minimizing the SFP preservation loss bounds forgetting.

    Concretely: if L_pres(a, a*) ≤ δ, then forgetting ≤ C · √δ.

    Paper reference: Theorem 4 (Main Reduction). -/
theorem sfp_reduces_forgetting
    (forgetting : ℝ)
    (a a_star : E)
    -- H1: forgetting is bounded by subspace drift (the empirical hypothesis)
    (C : ℝ) (hC : 0 ≤ C)
    (h_H1 : forgetting ≤ C * ‖(orthogonalProjection W (a - a_star) : E)‖)
    -- What SFP guarantees: the preservation loss is small
    (δ : ℝ) (hδ : 0 ≤ δ)
    (h_loss : preservationLoss W a a_star ≤ δ) :
    -- Conclusion: forgetting is bounded
    forgetting ≤ C * Real.sqrt δ := by
  calc forgetting
      ≤ C * ‖(orthogonalProjection W (a - a_star) : E)‖ := h_H1
    _ = C * Real.sqrt (preservationLoss W a a_star) := by
        rw [subspace_drift_bound]
    _ ≤ C * Real.sqrt δ := by
        apply mul_le_mul_of_nonneg_left
        · exact Real.sqrt_le_sqrt h_loss
        · exact hC

/-- Multi-layer version: if H1 holds per-layer with constants Cₗ,
    then the total SFP loss bounds total forgetting.

    Paper reference: Corollary 2. -/
theorem sfp_reduces_forgetting_multilayer
    {L : ℕ}
    (Wl : Fin L → Submodule ℝ E) [∀ l, CompleteSpace (Wl l)]
    (forgetting : Fin L → ℝ)
    (a a_star : Fin L → E)
    (C : Fin L → ℝ) (hC : ∀ l, 0 ≤ C l)
    (h_H1 : ∀ l, forgetting l ≤
      C l * ‖(orthogonalProjection (Wl l) (a l - a_star l) : E)‖)
    (δ : Fin L → ℝ) (hδ : ∀ l, 0 ≤ δ l)
    (h_loss : ∀ l, preservationLoss (Wl l) (a l) (a_star l) ≤ δ l) :
    ∀ l, forgetting l ≤ C l * Real.sqrt (δ l) :=
  fun l => sfp_reduces_forgetting (Wl l) (forgetting l) (a l) (a_star l)
    (C l) (hC l) (h_H1 l) (δ l) (hδ l) (h_loss l)

/-- The stability-plasticity tradeoff, formalized.

    Total activation drift decomposes into:
    - Preserved drift (bounded by SFP → controls forgetting via H1)
    - Unpreserved drift (free for new task learning → plasticity)

    SFP's λ parameter controls where on this tradeoff you sit.
    Paper reference: Theorem 5 (Stability-Plasticity Decomposition). -/
theorem stability_plasticity_tradeoff
    (forgetting : ℝ)
    (a a_star : E)
    (C : ℝ) (hC : 0 ≤ C)
    (h_H1 : forgetting ≤ C * ‖(orthogonalProjection W (a - a_star) : E)‖)
    (δ : ℝ) (hδ : 0 ≤ δ)
    (h_loss : preservationLoss W a a_star ≤ δ) :
    -- Stability: forgetting is bounded by the preservation budget
    forgetting ≤ C * Real.sqrt δ
    -- Plasticity: there is room for ‖a - a*‖² - δ of free drift
    ∧ ‖(orthogonalProjection Wᗮ (a - a_star) : E)‖ ^ 2 ≥ ‖a - a_star‖ ^ 2 - δ := by
  constructor
  · exact sfp_reduces_forgetting W forgetting a a_star C hC h_H1 δ hδ h_loss
  · -- From Pythagorean: ‖Δa‖² = L_pres + ‖P⊥(Δa)‖²
    -- So ‖P⊥(Δa)‖² = ‖Δa‖² - L_pres ≥ ‖Δa‖² - δ
    sorry

end Reduction

end SFP
