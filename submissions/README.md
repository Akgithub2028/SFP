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
