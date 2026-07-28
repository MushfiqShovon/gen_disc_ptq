#!/usr/bin/env bash
#
# Run the PTQ ablation grid and regenerate results/ptq_results.csv.
#
#   ./run_ptq.sh                            # plain PTQ, both models, all bit widths
#   ./run_ptq.sh --gpxq gpfq                # PTQ + GPFQ
#   ./run_ptq.sh --gpxq "none gpfq"         # both, in one pass
#   ./run_ptq.sh --model gen --bits "4 3" --gpxq gpfq
#   ./run_ptq.sh --collect-only             # just rebuild the CSV from existing logs
#
# Every cell writes its own row to results/ptq_results.csv keyed on
# (model, bit_width, gpxq), so re-running a cell replaces it and the table stays
# consistent no matter what order you run things in. Per-run logs and checkpoints
# are named after the bit width and algorithm, so nothing overwrites anything.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

MODELS="gen disc"
BITS="8 6 4 3 2"
GPXQ="none"
COLLECT_ONLY=0
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)        MODELS="$2"; shift 2 ;;
        --bits)         BITS="$2";   shift 2 ;;
        --gpxq)         GPXQ="$2";   shift 2 ;;
        --collect-only) COLLECT_ONLY=1; shift ;;
        -h|--help)      awk 'NR>1 && !/^#/{exit} NR>1{sub(/^# ?/, ""); print}' "$0"; exit 0 ;;
        *)              EXTRA+=("$1"); shift ;;   # anything else -> the python script
    esac
done

PYTHON="./nlp-env/bin/python"

if [[ $COLLECT_ONLY -eq 0 ]]; then
    total=0
    for model in $MODELS; do for bits in $BITS; do for algo in $GPXQ; do
        total=$((total + 1))
    done; done; done
    echo "==> running $total configuration(s): models=[$MODELS] bits=[$BITS] gpxq=[$GPXQ]"

    index=0
    for model in $MODELS; do
        for bits in $BITS; do
            for algo in $GPXQ; do
                index=$((index + 1))
                echo
                echo "--- [$index/$total] ${model} int${bits} gpxq=${algo} ---"
                ./quantize_"${model}".sh --bit-width "$bits" --gpxq "$algo" \
                    ${EXTRA[@]+"${EXTRA[@]}"}
            done
        done
    done
fi

echo
echo "==> regenerating results table"
"$PYTHON" -W ignore quantization/collect_results.py
