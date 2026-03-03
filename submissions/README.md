# Submissions

Each file here is a leaderboard submission. To submit your own:

```bash
cp submissions/_example.py submissions/my_method.py
# edit it
python submit.py submissions/my_method.py
```

## File format

```python
"""One-line description of your method."""

CONTRIBUTOR = "Your Name"

def my_method_loss(model, batch, **kw):
    """Your loss function. Same signature as methods.py."""
    ...
    return loss
```

Rules:
- File must define exactly one function ending in `_loss`
- Function signature: `(model, batch, **kw) -> Tensor`
- Can use: `torch`, `torch.nn.functional`, `copy`, `math`
- Cannot use: `os`, `subprocess`, `sys`, `requests`, etc.
- Max 10KB
- Optional: `SETUP = "sfp"` to use SFP setup (PCA basis + anchors)
- Valid SETUP values: `none`, `distill`, `hidden_distill`, `orthogonal`, `sfp`

## Benchmark protocol

Your method is evaluated under the official leaderboard protocol:

- **Model**: Qwen2.5-1.5B-Instruct with LoRA (r=64)
- **Tasks**: math, code, ifeval, safety, domain (5 tasks)
- **Steps**: 2000 per task
- **Memory budget**: 128 examples
- **Task orderings**: 3 different permutations (task order is the #1 variance source in CL)
- **Seeds**: 3 per ordering (9 total runs)
- **Setup budget**: 120 seconds max per task boundary (prevents smuggling computation)

## Scoring

**Harmonic mean** of retention and plasticity (like F1 = harmonic mean of precision and recall):

```
score = 2 · retention · plasticity / (retention + plasticity)
```

This penalises methods that sacrifice either axis. A do-nothing method (retention=1, plasticity=0) scores **0**, not 0.6.

**Minimum threshold**: if either retention or plasticity falls below 0.1, score is clamped to 0.

Results also show **Pareto optimality** — whether your method is dominated on both axes by any other method.

## Sandboxing

Code generation tasks (HumanEval) run model-generated code in a sandbox:
- Docker isolation when available (no network, 256MB RAM, 10s timeout)
- OS-level resource limits as fallback

## Zero-shot baseline

Every run records zero-shot performance before any training. This is the reference point for measuring actual forgetting vs. "the model was already bad."
