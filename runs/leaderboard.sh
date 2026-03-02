#!/bin/bash
# Run a method through the leaderboard benchmark
# Usage: bash runs/leaderboard.sh sfp 128
set -e

METHOD=${1:-naive}
MEMORY=${2:-128}

echo "=== Leaderboard run: method=$METHOD memory=$MEMORY ==="

for SEED in 42 43 44; do
    python train.py --method $METHOD --memory $MEMORY --seed $SEED \
        --model Qwen/Qwen2.5-1.5B-Instruct --tasks math,code,ifeval \
        --steps_per_task 1000 --out_dir out/leaderboard/${METHOD}_m${MEMORY}_s${SEED}
done

echo "=== Results ==="
python evaluate.py --aggregate out/leaderboard/${METHOD}_m${MEMORY}_s* --format leaderboard
