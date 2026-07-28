#!/usr/bin/env bash
#
# Calibration-sensitivity experiment: quantize with class-imbalanced calibration
# sets and measure the effect across bit widths.
#
#   ./run_calib_sensitivity.sh                          # disc, bits 8 6 4 3, all schemes
#   ./run_calib_sensitivity.sh --model gen
#   ./run_calib_sensitivity.sh --bits "4 3" --schemes "one_class two_class"
#   ./run_calib_sensitivity.sh --gpxq gpfq              # sensitivity of PTQ+GPFQ
#
# Schemes (fixed 8,192-doc budget; Class 1..4 = World, Sports, Business, Sci/Tech):
#   all_class 25/25/25/25 | three_class 33/33/33/0 | two_class 50/50/0/0 | one_class 100/0/0/0
#
# The sampled calibration files are a one-off job (make_calib_sets.py) -- created
# here if missing, then reused byte-identically. Every run records one row in
# results/calib_sensitivity.csv keyed on (model, bit_width, gpxq, calib_scheme);
# re-running a cell replaces its row, so any subset regenerates cleanly.
# No checkpoints are written.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON="./nlp-env/bin/python"

MODEL="disc"
BITS="8 6 4 3"
SCHEMES="all_class three_class two_class one_class"
GPXQ="none"
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)   MODEL="$2";   shift 2 ;;
        --bits)    BITS="$2";    shift 2 ;;
        --schemes) SCHEMES="$2"; shift 2 ;;
        --gpxq)    GPXQ="$2";    shift 2 ;;
        -h|--help) awk 'NR>1 && !/^#/{exit} NR>1{sub(/^# ?/, ""); print}' "$0"; exit 0 ;;
        *)         EXTRA+=("$1"); shift ;;
    esac
done

# One-off: build the sampled calibration sets if any are missing.
for scheme in $SCHEMES; do
    if [[ ! -f "data/ag_news/processed/calib/${scheme}.pt" ]]; then
        echo "==> generating calibration sets (one-off)"
        "$PYTHON" -W ignore quantization/make_calib_sets.py
        break
    fi
done

mkdir -p logs
total=0
for b in $BITS; do for s in $SCHEMES; do for a in $GPXQ; do total=$((total+1)); done; done; done
echo "==> $total run(s): model=$MODEL bits=[$BITS] schemes=[$SCHEMES] gpxq=[$GPXQ]"

index=0
for scheme in $SCHEMES; do
    for bits in $BITS; do
        for algo in $GPXQ; do
            index=$((index+1))
            suffix=""; [[ "$algo" != "none" ]] && suffix="_${algo}"
            LOG="logs/calib_${MODEL}_${scheme}_int${bits}${suffix}.log"
            echo
            echo "--- [$index/$total] $MODEL $scheme int$bits gpxq=$algo ---"
            "$PYTHON" -W ignore -u "quantization/ptq_${MODEL}.py" \
                --bit-width "$bits" \
                --gpxq "$algo" \
                --calib-file "data/ag_news/processed/calib/${scheme}.pt" \
                --no-save --skip-float-reference \
                ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee "$LOG" | grep -E "scheme|3\.|recorded"
        done
    done
done

echo
echo "==> results/calib_sensitivity.csv"
column -s, -t results/calib_sensitivity.csv
