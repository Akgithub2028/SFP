#!/bin/bash
# Full NeurIPS experiment suite (8×H100, ~7.5 days)
set -e

for METHOD in naive replay distill hidden_distill orthogonal sfp; do
  for MEM in 128 512; do
    for SEED in 42 43 44; do
      torchrun --standalone --nproc_per_node=8 train.py \
        --method $METHOD --memory $MEM --seed $SEED \
        --config configs/full.yaml \
        --out_dir out/full/${METHOD}_m${MEM}_s${SEED}
    done
  done
done

python analyze.py --paper out/full/
