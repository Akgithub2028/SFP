# Limitations & Threats to Validity

Honest reporting of limitations and negative findings is essential for sound scientific progress. Below we detail the identified threats to validity, empirical boundaries, and open theoretical questions.

---

## 1. Threats to Validity

### 1.1 The LoRA Confound
All primary benchmark experiments utilize Low-Rank Adaptation (LoRA) with rank $r=64$. By definition, LoRA constrains weight updates to a low-dimensional parameter manifold ($\Delta W = B A$, where $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$). 

**Threat**: The observation that activation drift is low-dimensional may partly be an architectural consequence of low-rank parameterization rather than an intrinsic property of catastrophic forgetting.
**Mitigation & Status**: While initial tests suggest that activation drift remains concentrated even under full fine-tuning at modest scales, large-scale (7B+) full fine-tuning experiments remain necessary to rule out parameter rank constraints as a confounding factor.

### 1.2 Model Scale Sensitivity (Empirical Negative Result)
Hypothesis H1 states that drift in top-$r$ PCA dimensions predicts forgetting significantly better than random or bottom dimensions.
- **At 1.5B Scale (Qwen2.5-1.5B)**: H1 is supported ($R^2_{\text{top}} = 0.933 > R^2_{\text{random}} = 0.850 > R^2_{\text{bottom}} = 0.827$).
- **At 135M Scale (SmolLM2-135M)**: H1 **fails** in correlational tests; the ranking inverts ($R^2_{\text{bottom}} = 0.994 > R^2_{\text{random}} = 0.987 > R^2_{\text{top}} = 0.912$).

**Takeaway**: At small parameter scales, representations lack the dimensional disentanglement present in larger models. Highly expressive 1.5B+ models dedicate distinct subspaces to specialized competencies, whereas 135M models rely on highly compressed, entangled representations where even bottom-variance noise correlates strongly with task degradation.

### 1.3 Modest Margin Over Random Subspaces
While top-$r$ drift achieves the highest $R^2$ (0.933) at 1.5B scale, the margin over random projections ($R^2 = 0.850$) is modest rather than order-of-magnitude. This indicates that forgetting is moderately concentrated in high-variance directions, but activation drift exhibits non-negligible global magnitude across the representation space.

### 1.4 Bound Tightness via Lipschitz Constant $C$
The formal reduction theorem proves:
$$\text{Forgetting} \le C \sqrt{\delta}$$
where $\delta$ is the preservation loss bound and $C$ is an empirical constant connecting activation drift to performance degradation ($\text{Forgetting} \le C \|P_W(\Delta a)\|$).
- If $C$ is large or varies erratically across task sequences, the theoretical bound becomes vacuous in practice.
- Bounding $C$ rigorously requires Lipschitz continuity assumptions on the downstream decoding layers and classifier heads.

---

## 2. Experimental Scope Constraints

### 2.1 Task Permutation Explosion
With $T = 5$ tasks, there exist $5! = 120$ possible sequential orderings. Running 9 runs per ordering across all 120 permutations would require over 1,000 GPU-hours per method. We evaluate on 3 carefully pre-registered orderings that rotate the final task, which mitigates last-task plasticity bias but does not exhaustively cover all combinatorial transitions.

### 2.2 Probe Layer Selection
By default, SFP hooks activations at three pre-registered depths: $L/4$, $L/2$, and $3L/4$. While this avoids post-hoc layer cherry-picking ($p$-hacking), different capabilities (e.g., surface syntax vs. deep algorithmic reasoning) localize at different depths in different transformer architectures.

---

## 3. Unanswered Research Questions

1. **Sparse Autoencoders (SAEs) vs. PCA**: Can overcomplete dictionary learning (e.g., jump-ReLU SAEs) uncover monosemantic feature directions that yield tighter bounds than linear orthogonal PCA?
2. **Gradient-Informed Task Subspaces**: Can projecting activations onto the gradient subspace of the anchor task ($\text{span}(\nabla_a \mathcal{L}_{\text{old}})$) eliminate the need for heuristic ablation ranking?
3. **Beyond LoRA to Full In-Place Post-Training**: Does SFP scale gracefully to distributed full-parameter fine-tuning of 7B-70B parameter models?
