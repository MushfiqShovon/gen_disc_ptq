#!/usr/bin/env bash
#
# Noise-robustness experiment: evaluate fp32 + quantized checkpoints on the
# QWERTY-typo test sets at all noise levels.
#
#   ./run_noise.sh                        # disc: fp32 + PTQ+GPFQ checkpoints, all levels
#   ./run_noise.sh --model gen
#   ./run_noise.sh --gpxq none            # plain-PTQ checkpoints instead
#   ./run_noise.sh --bits "32 4" --levels "0 5 10"
#
# Noisy sets are a one-off (robustness/make_noisy_sets.py) -- generated here if
# missing, then reused byte-identically by every model. Rows go to
# results/noise_robustness.csv keyed (model, bit_width, gpxq, noise_level);
# re-running a cell replaces it. bit_width=32 = float model.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON="./nlp-env/bin/python"

MODEL="disc"
GPXQ="gpfq"
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --gpxq)  GPXQ="$2";  shift 2 ;;
        -h|--help) awk 'NR>1 && !/^#/{exit} NR>1{sub(/^# ?/, ""); print}' "$0"; exit 0 ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done

if [[ ! -f data/ag_news/processed/noise/test_p30.pt ]]; then
    echo "==> generating noisy test sets (one-off)"
    "$PYTHON" -W ignore robustness/make_noisy_sets.py
fi

mkdir -p logs
LOG="logs/noise_${MODEL}_${GPXQ// /-}.log"
"$PYTHON" -W ignore -u robustness/noise_eval.py \
    --model "$MODEL" --gpxq "$GPXQ" \
    ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee "$LOG"

echo
echo "==> results/noise_robustness.csv"
column -s, -t results/noise_robustness.csv
