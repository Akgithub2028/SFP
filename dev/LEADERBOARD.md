# Forgetting Leaderboard — How to Contribute

## How scores work

- **Retention** = mean accuracy on old tasks after all training
- **Plasticity** = mean accuracy on the most recently trained task
- **Score** = 0.6 x Retention + 0.4 x Plasticity

## What's fixed (you can't change these)

- Model: Qwen2.5-1.5B-Instruct
- Tasks: math -> code -> ifeval (3 sequential tasks)
- Steps: 1000 per task
- LoRA rank: 64
- Seeds: 42, 43, 44 (report mean +/- std)
- GPU: A10G (via Modal)

## How to submit

1. Fork the repo
2. Add your method to `methods.py`:
   - Write a loss function (see existing ones for the pattern)
   - Register it in the `METHODS` dict
   - If your method needs setup at task boundaries, add a case in `method_setup()`
3. Open a PR with your method code
4. A maintainer adds the `benchmark` label to your PR
5. GitHub Actions spins up a GPU on [Modal](https://modal.com), runs the full benchmark (3 seeds x 1000 steps x 3 tasks), and posts your scores as a PR comment
6. If your scores are good, we merge and update the leaderboard

That's it. You write the method, we run the benchmark. No GPU needed on your end.

## Running locally (optional)

If you want to test before submitting:

```bash
# Quick sanity check (CPU, ~5 min)
python train.py --method your_method --memory 128 \
    --model HuggingFaceTB/SmolLM2-135M-Instruct \
    --tasks math,code --steps_per_task 50

# Full local benchmark (1 GPU)
bash runs/leaderboard.sh your_method 128
```

## Running on Modal yourself

```bash
pip install modal
modal setup
modal run scripts/modal_bench.py --method your_method --memory 128
```

## Memory tiers

There are separate leaderboards for M=128 and M=512.

## Rules

- Same total tokens as baselines (enforced by the harness)
- Same memory buffer (provided, not custom)
- No access to eval data during training
- Must be reproducible across 3 seeds
- No changes to `train.py`, `evaluate.py`, `data.py`, or `model.py`
