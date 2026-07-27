#!/usr/bin/env bash
#
# Static post-training quantization (weights + activations) of the generative
# classifier, using Brevitas with calibration on the training set.
#
#   ./quantize_gen.sh                     # 8-bit
#   ./quantize_gen.sh --bit-width 4       # any extra flag overrides the defaults
#   for b in 8 6 4 3 2; do ./quantize_gen.sh --bit-width $b; done   # ablation sweep
#
# The checkpoint and the log are both named after the bit width, so a sweep never
# overwrites its own earlier runs:
#   checkpoints/gen_lstm_agnews_int<N>.pth
#   logs/ptq_gen_int<N>.log
#
# Requires checkpoints/gen_lstm_agnews.pth -- run ./train_gen.sh first.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON="./nlp-env/bin/python"
CKPT="checkpoints/gen_lstm_agnews.pth"

if [[ ! -x "$PYTHON" ]]; then
    echo "error: $PYTHON not found -- see README for environment setup" >&2
    exit 1
fi

if [[ ! -f "$CKPT" ]]; then
    echo "error: $CKPT not found. Train the float model first:" >&2
    echo "  ./train_gen.sh" >&2
    exit 1
fi

# Default bit width, overridden by a --bit-width in the passthrough args so that
# the log file name matches whatever actually runs.
BITS=8
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[i]}" in
        --bit-width)   BITS="${args[i + 1]}" ;;
        --bit-width=*) BITS="${args[i]#*=}" ;;
    esac
done

mkdir -p logs
LOG="logs/ptq_gen_int${BITS}.log"

echo "==> static PTQ of the generative classifier at ${BITS} bits"
"$PYTHON" -W ignore -u quantization/ptq_gen.py \
    --checkpoint    "$CKPT" \
    --bit-width     "$BITS" \
    --batch-size    64 \
    --calib-batches 128 \
    "$@" 2>&1 | tee "$LOG"

echo "==> done, log written to $LOG"
