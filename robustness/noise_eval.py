"""Noise-robustness evaluation over the saved QWERTY-typo test sets.

Evaluates one model family (disc or gen) -- fp32 plus the saved quantized
checkpoints -- on the clean test set and every noise level, recording one row
per (config, level) into results/noise_robustness.csv.

Design points:
- The *saved* quantized checkpoints are reused as-is: calibration stays clean,
  so this measures the robustness of the deployed artifact, not of recalibration.
- Level 0 (clean) is always evaluated first; it is the per-config baseline for
  delta_vs_clean and doubles as a checkpoint round-trip check against the
  accuracy recorded at quantization time.
- bit_width=32 denotes the float model.

    python robustness/noise_eval.py --model disc --gpxq "none gpfq"
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
sys.path.insert(0, str(ROOT / 'quantization'))

import agnews_data  # noqa: E402
import DiscTrain  # noqa: E402
import GenTrain  # noqa: E402
import ptq_gen  # noqa: E402
import results  # noqa: E402
from models import DiscModel, GenModel  # noqa: E402
from quant_models import (QuantDiscModel, QuantGenModel, load_float_weights,  # noqa: E402
                          load_gen_float_weights)

NOISE_FIELDS = ['model', 'bit_width', 'gpxq', 'noise_level', 'test_acc',
                'delta_vs_clean', 'test_metric', 'char_sub_rate', 'unk_rate',
                'checkpoint']
NOISE_KEY = ('model', 'bit_width', 'gpxq', 'noise_level')


def load_quant_state(model, state):
    """Load a saved quantized checkpoint, dropping the `*_orig` copies that
    gpfq_mode(create_weight_orig=True) leaves on corrected layers -- `weight`
    itself already holds the GPFQ-corrected values."""
    state = {k: v for k, v in state.items() if not k.endswith('_orig')}
    model.load_state_dict(state)
    return model


def level_sets(levels, cache):
    """[(level, split_dict, meta)] -- level 0 is the clean cached test split."""
    out = []
    for level in levels:
        if level == 0:
            out.append((0, cache['test'], {'char_sub_rate': 0.0, 'unk_rate': ''}))
        else:
            path = ROOT / f'data/ag_news/processed/noise/test_p{level:02d}.pt'
            split = torch.load(path, weights_only=False)
            out.append((level, split, split['meta']))
    return out


def build_model(family, bits, gpxq, device, cache):
    """Returns (model, eval_fn(split) -> (metric, acc), checkpoint_name)."""
    vocab, classes = cache['vocab'], cache['classes']
    nclass = len(classes)

    def loader(split, collate):
        return DataLoader(agnews_data.AGNews(split), batch_size=64,
                          shuffle=False, collate_fn=collate)

    if family == 'disc':
        criterion = nn.CrossEntropyLoss().to(device)
        if bits == 32:
            name = 'disc_lstm_agnews.pth'
            ckpt = torch.load(ROOT / 'checkpoints' / name, weights_only=False)
            model = DiscModel(len(vocab), 100, 100, nclass, 1, False, 0.0, 8)
            model.load_state_dict(ckpt['state_dict'])
        else:
            suffix = '' if gpxq == 'none' else f'_{gpxq}'
            name = f'disc_lstm_agnews_int{bits}{suffix}.pth'
            ckpt = torch.load(ROOT / 'checkpoints' / name, weights_only=False)
            d = ckpt['args']
            model = QuantDiscModel(d['vocab_size'], d['embedding_dim'], d['hidden_dim'],
                                   d['output_dim'], d['n_layers'], bit_width=bits)
            load_quant_state(model, ckpt['state_dict'])
        model = model.to(device).eval()

        def run(split):
            return DiscTrain.evaluate(loader(split, DiscTrain.collate), model,
                                      criterion, device)
        return model, run, name, ckpt.get('test_acc')

    # generative
    if bits == 32:
        name = 'gen_lstm_agnews.pth'
        ckpt = torch.load(ROOT / 'checkpoints' / name, weights_only=False)
        model = GenModel(len(vocab), 100, 100, 100, 1, nclass, 0.0, True,
                         False, False, 'hidden', False, False)
        model.load_state_dict(ckpt['state_dict'])
        model = model.to(device).eval()
        criterion = nn.CrossEntropyLoss(reduction='none').to(device)

        def run(split):
            return GenTrain.evaluate(loader(split, GenTrain.collate), model,
                                     criterion, device, nclass)
        return model, run, name, ckpt.get('test_acc')

    suffix = '' if gpxq == 'none' else f'_{gpxq}'
    name = f'gen_lstm_agnews_int{bits}{suffix}.pth'
    ckpt = torch.load(ROOT / 'checkpoints' / name, weights_only=False)
    d = ckpt['args']
    model = QuantGenModel(d['vocab_size'], d['embedding_dim'], d['label_emb_dim'],
                          d['hidden_dim'], d['nclass'], d['n_layers'], bit_width=bits)
    load_quant_state(model, ckpt['state_dict'])
    model = model.to(device).eval()

    def run(split):
        return ptq_gen.evaluate(loader(split, DiscTrain.collate), model, device, nclass)
    return model, run, name, ckpt.get('test_acc')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=('disc', 'gen'), default='disc')
    p.add_argument('--bits', default='32 8 6 4 3 2',
                   help='32 = float model; others load saved quantized checkpoints')
    p.add_argument('--gpxq', default='none', help='space-separated: none gpfq qronos')
    p.add_argument('--levels', default='0 1 2 5 10 15 20 30',
                   help='noise percent levels; 0 = clean (always evaluated as baseline)')
    p.add_argument('--data', type=Path, default=ROOT / 'data/ag_news/processed/agnews.pt')
    p.add_argument('--results', type=Path, default=ROOT / 'results/noise_robustness.csv')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(agnews_data.SEED)
    cache = torch.load(args.data, weights_only=False)

    levels = sorted({int(l) for l in args.levels.split()} | {0})
    sets = level_sets(levels, cache)
    bits_list = [int(b) for b in args.bits.split()]
    gpxq_list = args.gpxq.split()

    # fp32 is always the (single) baseline config regardless of --gpxq.
    configs = []
    for b in bits_list:
        if b == 32:
            configs.append((32, 'none'))
        else:
            configs.extend((b, g) for g in gpxq_list)
    print(f'model={args.model} | device={device} | {len(configs)} config(s) x {len(sets)} level(s)')

    for bits, gpxq in configs:
        tag = 'fp32' if bits == 32 else f'int{bits}' + ('' if gpxq == 'none' else f'+{gpxq}')
        try:
            model, run, name, recorded = build_model(args.model, bits, gpxq, device, cache)
        except FileNotFoundError as exc:
            print(f'\n== {tag}: checkpoint missing, skipping ({Path(str(exc)).name})')
            continue

        print(f'\n== {tag} ({name}) ==')
        clean_acc = None
        for level, split, meta in sets:
            start = time.time()
            metric, acc = run(split)
            if level == 0:
                clean_acc = acc
                check = '' if recorded is None else f' | recorded at quantization: {recorded:.2f}'
                print(f'  p=0.00  acc {acc:6.2f}  (clean baseline{check})   [{time.time() - start:.0f}s]')
            else:
                print(f'  p={level / 100:.2f}  acc {acc:6.2f}  ({acc - clean_acc:+6.2f} vs clean, '
                      f'unk {float(meta["unk_rate"]) * 100:.1f}%)   [{time.time() - start:.0f}s]')

            row = {'model': args.model, 'bit_width': bits, 'gpxq': gpxq,
                   'noise_level': level, 'test_acc': f'{acc:.2f}',
                   'delta_vs_clean': f'{acc - clean_acc:+.2f}',
                   'test_metric': f'{metric:.4f}',
                   'char_sub_rate': (f'{float(meta["char_sub_rate"]):.4f}'
                                     if meta['char_sub_rate'] != '' else '0'),
                   'unk_rate': (f'{float(meta["unk_rate"]):.4f}'
                                if meta['unk_rate'] != '' else ''),
                   'checkpoint': name}
            results.record(args.results, row, NOISE_FIELDS, NOISE_KEY)
        del model
        torch.cuda.empty_cache()

    print(f'\nrecorded -> {args.results}')


if __name__ == '__main__':
    main()
