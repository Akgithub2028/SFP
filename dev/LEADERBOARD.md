# Forgetting Leaderboard — How to Contribute

## How scores work

- **Retention** = mean accuracy on old tasks after all training
- **Plasticity** = mean accuracy on the most recently trained task
- **Score** = 0.6 × Retention + 0.4 × Plasticity

## What's fixed (you can't change these)

- Model: Qwen2.5-1.5B-Instruct
- Tasks: math → code → ifeval (3 sequential tasks)
- Steps: 1000 per task
- LoRA rank: 64
- Seeds: 42, 43, 44 (report mean ± std)

## How to submit

1. Fork the repo
2. Add your method to `methods.py` (or a new file that exports a loss function)
3. Run: `bash runs/leaderboard.sh your_method 128`
4. The script prints your scores in leaderboard format
5. Open a PR with your method code + results

## Memory tiers

There are separate leaderboards for M=128 and M=512.

## Rules

- Same total tokens as baselines (enforced by the harness)
- Same memory buffer (provided, not custom)
- No access to eval data during training
- Must be reproducible across 3 seeds
