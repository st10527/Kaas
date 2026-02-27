#!/bin/bash
# parallel_run.sh — Run seeds in parallel on same GPU
# RTX 5070 Ti (16GB) can handle 2 concurrent CIFAR-100 experiments (~1GB each)
#
# Usage: bash parallel_run.sh [experiment] [device]
#   e.g.: bash parallel_run.sh main cuda:0
#         bash parallel_run.sh all cuda:0

EXP=${1:-main}
DEV=${2:-cuda:0}
SCRIPT="scripts/run_edge_experiments.py"

echo "══════════════════════════════════════════════"
echo " Parallel execution: $EXP on $DEV"
echo "══════════════════════════════════════════════"

if [ "$EXP" = "all" ]; then
    EXPS="main budget scale privacy"
else
    EXPS="$EXP"
fi

for exp in $EXPS; do
    echo ""
    echo "── Experiment: $exp ──"
    echo " Launching seed 42 & 123 in parallel..."

    # Wave 1: seeds 42 and 123 in parallel
    python $SCRIPT --exp $exp --seed 42  --device $DEV > logs/${exp}_s42.log  2>&1 &
    PID1=$!
    python $SCRIPT --exp $exp --seed 123 --device $DEV > logs/${exp}_s123.log 2>&1 &
    PID2=$!

    echo "  PIDs: $PID1 (seed42), $PID2 (seed123)"
    wait $PID1 $PID2
    echo "  Wave 1 done."

    # Wave 2: seed 456 alone
    echo "  Launching seed 456..."
    python $SCRIPT --exp $exp --seed 456 --device $DEV > logs/${exp}_s456.log 2>&1
    echo "  Wave 2 done."

    echo " ✓ $exp complete."
done

echo ""
echo "── Merging per-seed JSONs ──"
python merge_seed_results.py
echo "══════════════════════════════════════════════"
echo " All done! Check results/edge/*.json"
echo "══════════════════════════════════════════════"
