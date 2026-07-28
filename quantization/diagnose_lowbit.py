"""Locate which stage causes the discriminative model's low-bit collapse.

Holds one stage at high precision while crushing the others, so the accuracy drop
can be attributed to a stage rather than guessed at. Also prints the prediction
histogram, which distinguishes "lost some signal" from "collapsed onto one class".

    python quantization/diagnose_lowbit.py --low 3
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import agnews_data  # noqa: E402
from DiscTrain import collate  # noqa: E402
from ptq_disc import apply_gpxq, calibrate  # noqa: E402
from quant_models import QuantDiscModel, load_float_weights  # noqa: E402


@torch.no_grad()
def evaluate_detailed(loader, model, device, nclass):
    model.eval()
    correct, seen = 0, 0
    predicted = torch.zeros(nclass, dtype=torch.long)
    for text, lengths, labels in loader:
        logits = model(text.to(device), lengths)
        guess = logits.argmax(dim=1).cpu()
        correct += (guess == labels).sum().item()
        predicted += torch.bincount(guess, minlength=nclass)
        seen += len(labels)
    return 100.0 * correct / seen, predicted.tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=ROOT / 'checkpoints/disc_lstm_agnews.pth')
    p.add_argument('--data', type=Path, default=ROOT / 'data/ag_news/processed/agnews.pt')
    p.add_argument('--low', type=int, default=3, help='the bit width that collapses')
    p.add_argument('--high', type=int, default=8, help='precision to hold a stage at')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--calib-batches', type=int, default=128)
    p.add_argument('--gpxq', choices=('none', 'gpfq'), default='none')
    p.add_argument('--gpxq-batches', type=int, default=32)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, weights_only=False)
    state, vocab, classes = ckpt['state_dict'], ckpt['vocab'], ckpt['classes']
    trained = ckpt['args']
    nclass = len(classes)
    dims = dict(vocab_size=len(vocab), embedding_dim=trained['word_emb_dim'],
                hidden_dim=trained['hid_dim'], output_dim=nclass, n_layers=trained['layers'])

    low, high = args.low, args.high
    configs = [
        (f'all {high}-bit', high, high, high),
        (f'all {low}-bit', low, low, low),
        (f'embedding {high}, lstm+fc {low}', high, low, low),
        (f'lstm {high}, embedding+fc {low}', low, high, low),
        (f'fc {high}, embedding+lstm {low}', low, low, high),
    ]

    print(f'discriminative model | low={low} high={high} | gpxq={args.gpxq}')
    print(f'test-set class balance is uniform, so 25.00 is chance\n')
    print(f'  {"configuration":34s} {"acc":>7s}   prediction histogram')
    print(f'  {"-" * 34} {"-" * 7}   {"-" * 40}')

    for name, emb, lstm, fc in configs:
        torch.manual_seed(agnews_data.SEED)
        _, loaders = agnews_data.load(args.data, args.batch_size, collate)
        model = QuantDiscModel(**dims, bit_width=low, emb_bit_width=emb,
                               lstm_bit_width=lstm, fc_bit_width=fc)
        load_float_weights(model, state)
        model.to(device)

        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):   # calibration chatter
            calibrate(model, loaders['train'], device, args.calib_batches, False, classes)
            if args.gpxq != 'none':
                apply_gpxq(model, loaders['train'], device, args.gpxq_batches, args.gpxq)

        acc, histogram = evaluate_detailed(loaders['test'], model, device, nclass)
        spread = ' '.join(f'{c[:4]}={n:>4d}' for c, n in zip(classes, histogram))
        print(f'  {name:34s} {acc:7.2f}   {spread}')
        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
