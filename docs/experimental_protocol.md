# Experimental Protocol

This document specifies the rigorous benchmarking and evaluation protocol designed to test catastrophic forgetting in sequential large language model post-training.

---

## 1. Task Suite and Benchmarks

The benchmark evaluates sequential adaptation across 5 distinct domains representing common production fine-tuning workloads:

| Task ID | Domain | Training Dataset | Evaluation Dataset | Primary Metric |
|---------|--------|------------------|-------------------|----------------|
| `math` | Mathematical Reasoning | GSM8K (train split) | GSM8K (test, 1319 samples) | Exact Match Accuracy (`math_accuracy`) |
| `code` | Code Generation | MBPP (train split) | HumanEval (164 problems) | pass@1 Sandboxed (`code_pass_at_1`) |
| `ifeval` | Instruction Following | Alpaca-cleaned | IFEval verifiable constraints | Strict Constraint Satisfaction (`ifeval_accuracy`) |
| `safety` | Alignment & Refusal | Anthropic HH-RLHF | Red-teaming prompts | Refusal Rate × (1 − Over-Refusal) (`safety_score`) |
| `domain` | Science / BioQA | PubMedQA & SciQ | SciQ test set | Accuracy / Perplexity (`domain_qa_accuracy`) |

---

## 2. Experimental Controls

### 2.1 Pre-Training Zero-Shot Baseline
Before initiating sequential training, the base model $\theta_0$ is evaluated across all 5 evaluation splits. This zero-shot anchor establishes the true baseline and distinguishes whether a model degraded from fine-tuning or had low competence initially.

### 2.2 Task Order Permutations
Task sequence order is known to be the largest source of variance in continual learning benchmarks. Evaluating on a single fixed order allows methods to inadvertently overfit sequence order.

We pre-register **3 deterministic orderings** rotating the final task:
1. **Order 1**: `math` $\to$ `code` $\to$ `ifeval` $\to$ `safety` $\to$ `domain`
2. **Order 2**: `domain` $\to$ `math` $\to$ `code` $\to$ `ifeval` $\to$ `safety`
3. **Order 3**: `safety` $\to$ `domain` $\to$ `ifeval` $\to$ `code` $\to$ `math`

### 2.3 Random Seeds and Statistical Confidence
Each method is evaluated across 3 random seeds ($s \in \{42, 43, 44\}$) per task ordering, resulting in **9 independent sequential training runs per method**. All reported metrics include sample mean and sample standard deviation ($\mu \pm \sigma$).

### 2.4 Setup Time Budget
At each task boundary, any setup computation (e.g., Fisher information computation, activation collection, PCA decomposition) is strictly capped at **120 seconds wall-clock time**. This prevents methods from smuggling unbounded offline computation or optimization into task transition phases.

---

## 3. Training Hyperparameters

All methods are trained under strictly matched parameter budgets:
- **Architecture**: Qwen/Qwen2.5-1.5B-Instruct (bfloat16) and SmolLM2-135M-Instruct (falsification control)
- **Fine-Tuning Method**: LoRA (Rank $r = 64$, $\alpha = 128$, target modules: `q_proj`, `v_proj`, `k_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
- **Optimizer**: AdamW ($\beta_1 = 0.9, \beta_2 = 0.999$, weight decay $= 0.01$)
- **Learning Rate**: $2 \times 10^{-4}$ with linear warmup (100 steps) and cosine decay
- **Batch Size**: 4 per device (gradient accumulation steps: 4, effective batch size: 16)
- **Steps per Task**: 2,000 steps per task
- **Memory Buffer ($\mathcal{M}$)**: 128 total examples (balanced across previously seen tasks, 256 tokens max length)

---

## 4. Execution Sandbox for Code Generation

To ensure evaluation safety and reproducibility during HumanEval pass@1 execution:
1. **Docker Container Sandbox**: Evaluated code runs inside an unprivileged Docker container with:
   - Network connectivity disabled (`--network none`)
   - Memory strictly capped at 256 MB (`--memory 256m`)
   - CPU quota limited to 1 vCPU
   - Wall-clock timeout of 10.0 seconds per problem
2. **OS-Level Fallback**: When Docker is unavailable, processes are spawned with `resource.setrlimit` enforcing identical CPU, virtual memory, and sub-process restrictions.
