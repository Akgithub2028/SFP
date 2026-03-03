#!/bin/bash
# Order sensitivity sweep: 10 random permutations × {naive, sfp}
# Uses SmolLM2-135M, 200 steps/task, memory=64 (~5 min/run on 1 GPU)
set -e

TASKS=(math code ifeval safety domain)
NUM_PERMS=10
METHODS="naive sfp"
OUT_BASE="out/order_sweep"

mkdir -p "$OUT_BASE"

# Generate and run permutations
for PERM_IDX in $(seq 0 $((NUM_PERMS - 1))); do
    # Shuffle tasks with a deterministic seed per permutation
    TASK_ORDER=$(python -c "
import random
random.seed($PERM_IDX)
tasks = list('${TASKS[*]}'.split())
random.shuffle(tasks)
print(','.join(tasks))
")
    echo "=== Permutation $PERM_IDX: $TASK_ORDER ==="

    for METHOD in $METHODS; do
        OUT_DIR="${OUT_BASE}/${METHOD}_perm${PERM_IDX}"
        if [ -f "$OUT_DIR/results.json" ]; then
            echo "  Skipping $METHOD perm$PERM_IDX (already done)"
            continue
        fi
        echo "  Running $METHOD ..."
        python train.py \
            --method "$METHOD" \
            --model HuggingFaceTB/SmolLM2-135M-Instruct \
            --tasks "$TASK_ORDER" \
            --steps_per_task 200 \
            --memory 64 \
            --seed 42 \
            --out_dir "$OUT_DIR"
    done
done

echo ""
echo "=== Analyzing order sensitivity ==="
python analyze.py --order-sensitivity "$OUT_BASE"/*
