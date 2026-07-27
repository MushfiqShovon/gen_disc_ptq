#!/usr/bin/env bash
#
# Train the discriminative classifier (LSTM + mean-pooled hidden states) on AG News.
#
#   ./train_disc.sh                    # best-known configuration
#   ./train_disc.sh --dropout 0        # any extra flag overrides the defaults below
#
# Best result reproduced by this script: 92.29 dev / 92.24 test accuracy
# (best epoch 10 of 15, ~22s per epoch on the GB10).
#
# Note: dropout 0.5 is worth +1.2 test points here. Without it the model peaks at
# epoch 3 and then overfits (97.9% train vs 90.9% dev). Pass --dropout 0 to
# reproduce the paper-faithful run (91.07 test).

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON="./nlp-env/bin/python"
CACHE="data/ag_news/processed/agnews.pt"

if [[ ! -x "$PYTHON" ]]; then
    echo "error: $PYTHON not found. Create the env first:" >&2
    echo "  conda create -y -p ./nlp-env python=3.11" >&2
    echo "  ./nlp-env/bin/pip install --index-url https://download.pytorch.org/whl/cu130 torch" >&2
    echo "  ./nlp-env/bin/pip install spacy pandas 'click<8.2'" >&2
    echo "  ./nlp-env/bin/python -m spacy download en_core_web_sm" >&2
    exit 1
fi

# One-off data preparation; skipped once the cache exists.
if [[ ! -f "$CACHE" ]]; then
    echo "==> $CACHE missing, running one-off data preparation"
    "$PYTHON" training/prepare_data.py
    "$PYTHON" training/preprocess.py
fi

mkdir -p logs
echo "==> training discriminative classifier"
"$PYTHON" -u training/DiscTrain.py \
    --data          "$CACHE" \
    --out           checkpoints/disc_lstm_agnews.pth \
    --epochs        100 \
    --batch-size    64 \
    --lr            1e-3 \
    --weight-decay  1e-5 \
    --word-emb-dim  100 \
    --hid-dim       100 \
    --layers        1 \
    --dropout       0.5 \
    --clip          1.0 \
    --log-interval  200 \
    "$@" 2>&1 | tee logs/train_disc.log

echo "==> done, log written to logs/train_disc.log"
