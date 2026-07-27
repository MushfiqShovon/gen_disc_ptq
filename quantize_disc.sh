#!/usr/bin/env bash
#
# Static post-training quantization (weights + activations) of the discriminative
# classifier, using Brevitas with calibration on the training set.
#
#   ./quantize_disc.sh                     # 8-bit
#   ./quantize_disc.sh --bit-width 4       # any extra flag overrides the defaults
#   for b in 8 6 4 3 2; do ./quantize_disc.sh --bit-width $b; done   # ablation sweep
#
# The checkpoint and the log are both named after the bit width, so a sweep never
# overwrites its own earlier runs:
#   checkpoints/disc_lstm_agnews_int<N>.pth
#   logs/ptq_disc_int<N>.log
#
# Requires checkpoints/disc_lstm_agnews.pth -- run ./train_disc.sh first.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON="./nlp-env/bin/python"
CKPT="checkpoints/disc_lstm_agnews.pth"

if [[ ! -x "$PYTHON" ]]; then
    echo "error: $PYTHON not found -- see README for environment setup" >&2
    exit 1
fi

if [[ ! -f "$CKPT" ]]; then
    echo "error: $CKPT not found. Train the float model first:" >&2
    echo "  ./train_disc.sh" >&2
    exit 1
fi

# Default bit width, overridden by a --bit-width in the passthrough args so that
# the log file name matches whatever actually runs.
BITS=8
GPXQ=none
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[i]}" in
        --bit-width)   BITS="${args[i + 1]}" ;;
        --bit-width=*) BITS="${args[i]#*=}" ;;
        --gpxq)        GPXQ="${args[i + 1]}" ;;
        --gpxq=*)      GPXQ="${args[i]#*=}" ;;
    esac
done
SUFFIX=""; [[ "$GPXQ" != "none" ]] && SUFFIX="_${GPXQ}"

mkdir -p logs
LOG="logs/ptq_disc_int${BITS}${SUFFIX}.log"

echo "==> static PTQ of the discriminative classifier at ${BITS} bits (gpxq=${GPXQ})"
"$PYTHON" -W ignore -u quantization/ptq_disc.py \
    --checkpoint    "$CKPT" \
    --bit-width     "$BITS" \
    --batch-size    64 \
    --calib-batches 128 \
    "$@" 2>&1 | tee "$LOG"

echo "==> done, log written to $LOG"
