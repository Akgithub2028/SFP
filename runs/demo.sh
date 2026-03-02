#!/bin/bash
# Smoke test: verify everything works end-to-end (~30 min, CPU or 1 GPU)
set -e

for METHOD in naive replay sfp; do
    echo "=== Running method: $METHOD ==="
    python train.py --method $METHOD --memory 32 \
        --model HuggingFaceTB/SmolLM2-135M-Instruct \
        --tasks math,code --steps_per_task 200 --seed 42 \
        --out_dir out/smoke/${METHOD}
done

echo "=== Comparing results ==="
python analyze.py --compare out/smoke/naive out/smoke/replay out/smoke/sfp
