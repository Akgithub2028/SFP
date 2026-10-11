# Evaluation Metrics & Statistical Protocol

## 1. Primary Benchmark Metrics

Continual fine-tuning requires balancing two competing objectives:
1. **Stability (Retention)**: Preserving competence on previously acquired tasks.
2. **Plasticity**: Rapidly acquiring new capabilities on the current task.

Let $A_{i, j}$ denote the evaluation accuracy on task $\mathcal{T}_j$ after completing training on task $\mathcal{T}_i$ (where $1 \le j \le i \le T$). Let $A_{0, j}$ denote the pre-training zero-shot accuracy on task $\mathcal{T}_j$.

### 1.1 Retention Metric ($R$)
Retention measures the normalized performance retained across all previously trained tasks after training on all $T$ tasks completes:

$$R = \frac{1}{T - 1} \sum_{j=1}^{T-1} \frac{A_{T, j}}{\max(A_{j, j}, A_{0, j})}$$

### 1.2 Plasticity Metric ($P$)
Plasticity measures the model's ability to learn each new task upon completion of its training:

$$P = \frac{1}{T} \sum_{i=1}^T \frac{A_{i, i} - A_{0, i}}{1.0 - A_{0, i}}$$

### 1.3 Harmonic Evaluation Score ($H$)
Rather than using an arithmetic linear combination (e.g., $0.6 R + 0.4 P$) which allows trivial degenerate strategies (e.g., a frozen model scoring high by doing nothing), we adopt the **Harmonic Mean** of retention and plasticity, analogous to the $F_1$ score in classification:

$$H = \begin{cases} 
\frac{2 \cdot R \cdot P}{R + P} & \text{if } R \ge 0.1 \text{ and } P \ge 0.1 \\
0.0 & \text{otherwise}
\end{cases}$$

If either Retention or Plasticity drops below the floor threshold of $0.10$, the overall score is clamped to zero. This enforces that any competitive method must demonstrate non-trivial learning *and* non-trivial retention.

---

## 2. Standard Continual Learning Metrics

### 2.1 Average Accuracy (AA)
The overall accuracy across all tasks at the end of the sequential stream:
$$\text{AA} = \frac{1}{T} \sum_{j=1}^T A_{T, j}$$

### 2.2 Backward Transfer (BWT)
Measures the net influence of learning subsequent tasks on past task performance:
$$\text{BWT} = \frac{1}{T - 1} \sum_{j=1}^{T-1} (A_{T, j} - A_{j, j})$$
Negative values quantify catastrophic forgetting; positive values reflect backwards generalization.

### 2.3 Forward Transfer (FWT)
Measures zero-shot learning transferred from past tasks to future tasks prior to task-specific training:
$$\text{FWT} = \frac{1}{T - 1} \sum_{i=2}^T (A_{i-1, i} - A_{0, i})$$

---

## 3. Pareto Optimality

A method $M_A$ with retention and plasticity $(R_A, P_A)$ strictly dominates $M_B$ with $(R_B, P_B)$ if and only if:
$$(R_A \ge R_B \land P_A > P_B) \quad \lor \quad (R_A > R_B \land P_A \ge P_B)$$

A method is **Pareto Optimal** if it is not dominated by any other evaluated method under the benchmark protocol.

---

## 4. Hypothesis Testing Metrics for Low-Dimensional Drift (H1)

To validate whether activation drift predicts forgetting along low-dimensional subspaces:
1. **Univariate Linear Regression**:
   $$\text{Forgetting}(t) = \beta_0 + \beta_1 \cdot \|\Delta a_{\text{subspace}}(t)\| + \epsilon_t$$
   The coefficient of determination $R^2$ is computed across training checkpoint steps $t \in \{1, \dots, N\}$.
2. **Partial Correlation Control**:
   To ensure that high $R^2$ is not an artifact of total activation drift $\|\Delta a\|$, we compute the partial correlation:
   $$R^2(\text{top-}r \mid \text{total}) = 1 - \frac{1 - R^2(\text{top-}r, \text{total})}{1 - R^2(\text{total})}$$
3. **Causal Intervention Ratio**:
   At test time, Gaussian noise $\eta \sim \mathcal{N}(0, \sigma^2 I)$ is projected onto top-$r$, random-$r$, and bottom-$r$ subspaces. The ratio of perplexity increase:
   $$\text{Ratio} = \frac{\Delta \text{PPL}(\text{top-}r)}{\Delta \text{PPL}(\text{random-}r)}$$
   quantifies causal relevance.
