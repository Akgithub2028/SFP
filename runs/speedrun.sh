#!/bin/bash
# Gate 1 + Gate 2 fast falsification (1-4 GPUs, 2-6 days)
set -e

echo "=== Stage 1: Gate 1 — Is forgetting low-dimensional? ==="
for METHOD in naive replay distill sfp; do
  for MEM in 32 128; do
    for SEED in 42 43; do
      python train.py --method $METHOD --memory $MEM --seed $SEED \
        --model Qwen/Qwen2.5-1.5B-Instruct --tasks math,code,ifeval \
        --steps_per_task 500 --out_dir out/gate1/${METHOD}_m${MEM}_s${SEED}
    done
  done
done

python analyze.py --gate1 out/gate1/

echo "=== Stage 2: Gate 2 — Does SFP win? ==="
for METHOD in naive replay distill hidden_distill orthogonal sfp; do
  for MEM in 128 512; do
    for SEED in 42 43 44; do
      python train.py --method $METHOD --memory $MEM --seed $SEED \
        --model Qwen/Qwen2.5-3B-Instruct --tasks math,code,ifeval,safety \
        --steps_per_task 1000 --out_dir out/gate2/${METHOD}_m${MEM}_s${SEED}
    done
  done
done

python analyze.py --gate2 out/gate2/
