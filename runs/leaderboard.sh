#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# Leaderboard benchmark — the official competition evaluation.
#
# Follows best-practice CL benchmark design (CLEAR, CTrL, Avalanche):
#   - 5 tasks (math, code, ifeval, safety, domain)
#   - 3 task orderings (task order is the #1 variance source in CL)
#   - 3 seeds per ordering (9 total runs)
#   - Results averaged across ALL runs, with std reported
#
# Usage:
#   bash runs/leaderboard.sh sfp 128          # run method with memory=128
#   bash runs/leaderboard.sh naive 0          # no memory baseline
#   bash runs/leaderboard.sh my_method 128 submissions/my_method.py  # submission
# ─────────────────────────────────────────────────────────────────
set -e

METHOD=${1:-naive}
MEMORY=${2:-128}
METHOD_FILE=${3:-}

# 3 task orderings — deterministic, covering different "which task comes last"
# so retention/plasticity isn't biased by one particular final task.
ORDERINGS=(
    "math,code,ifeval,safety,domain"
    "code,safety,math,domain,ifeval"
    "domain,ifeval,safety,code,math"
)
SEEDS=(42 43 44)

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  SFP Leaderboard Benchmark                              ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Method    : $METHOD"
echo "║  Memory    : $MEMORY"
echo "║  Orderings : ${#ORDERINGS[@]}"
echo "║  Seeds     : ${#SEEDS[@]}"
echo "║  Total runs: $(( ${#ORDERINGS[@]} * ${#SEEDS[@]} ))"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

METHOD_FLAG=""
if [ -n "$METHOD_FILE" ]; then
    METHOD_FLAG="--method_file $METHOD_FILE"
    echo "Using external method file: $METHOD_FILE"
    echo ""
fi

OUT_BASE="out/leaderboard/${METHOD}_m${MEMORY}"

for ORD_IDX in "${!ORDERINGS[@]}"; do
    ORDER="${ORDERINGS[$ORD_IDX]}"
    for SEED in "${SEEDS[@]}"; do
        RUN_DIR="${OUT_BASE}/ord${ORD_IDX}_s${SEED}"
        echo "=== Ordering ${ORD_IDX} (${ORDER}), Seed ${SEED} ==="
        python train.py \
            --config configs/leaderboard.yaml \
            --method "$METHOD" \
            --memory "$MEMORY" \
            --seed "$SEED" \
            --tasks "$ORDER" \
            --out_dir "$RUN_DIR" \
            $METHOD_FLAG
        echo ""
    done
done

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  AGGREGATE RESULTS                                       ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Aggregate across all orderings and seeds
python evaluate.py --aggregate ${OUT_BASE}/ord*_s* --format leaderboard

# Also compute summary statistics
echo ""
echo "--- Per-ordering breakdown ---"
for ORD_IDX in "${!ORDERINGS[@]}"; do
    ORDER="${ORDERINGS[$ORD_IDX]}"
    echo ""
    echo "Ordering ${ORD_IDX}: ${ORDER}"
    python evaluate.py --aggregate ${OUT_BASE}/ord${ORD_IDX}_s* --format leaderboard
done
