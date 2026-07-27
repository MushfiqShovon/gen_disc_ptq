"""Static post-training quantization of the generative classifier (Brevitas).

Same recipe as ptq_disc.py -- weights and activations quantized to `--bit-width`,
activation scales calibrated on a random sample of the *training* set, evaluated
on dev and test.

The generative classifier scores a document under every candidate label and takes
the argmin of the summed token NLL, so quantization error enters at every token
and every label, and the decision rests on the margin between four accumulated
sums. That makes it a priori more fragile than the discriminative model, whose
decision is a single argmax over four logits.

    python quantization/ptq_gen.py --bit-width 8
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from brevitas.graph.calibrate import bias_correction_mode, calibration_mode
from brevitas.graph.gpfq import GPFQ, gpfq_mode
from brevitas.graph.qronos import Qronos

GPXQ_ALGORITHMS = {'gpfq': GPFQ, 'qronos': Qronos}

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import agnews_data  # noqa: E402
import GenTrain  # noqa: E402
from DiscTrain import collate as collate_padded  # noqa: E402  (plain pad + lengths)
from models import GenModel  # noqa: E402
from quant_models import QuantGenModel, load_gen_float_weights  # noqa: E402


def split_teacher_forced(text, lengths, device):
    """(padded sequence, lengths) -> LM inputs, targets, and a real-token mask."""
    text = text.to(device)
    inputs, targets = text[:, :-1], text[:, 1:]
    lengths = (lengths - 1).to(device)
    steps = torch.arange(inputs.shape[1], device=device)
    return inputs, targets, steps.unsqueeze(0) < lengths.unsqueeze(1)


def score_all_labels(model, inputs, targets, mask, nclass):
    """Summed -log p(x | y) per document for every label. Returns (nclass, batch)."""
    states = model.encode(inputs)          # label-independent: run the LSTM once
    scores = []
    for label in range(nclass):
        logits = model.decode(states, label)
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                              targets.reshape(-1), reduction='none').view_as(targets)
        scores.append((nll * mask).sum(dim=1))
    return torch.stack(scores)


@torch.no_grad()
def evaluate(loader, model, device, nclass):
    model.eval()
    total_nll, correct, seen = 0.0, 0, 0
    for text, lengths, labels in loader:
        labels = labels.to(device)
        scores = score_all_labels(model, *split_teacher_forced(text, lengths, device), nclass)
        correct += (scores.argmin(dim=0) == labels).sum().item()
        total_nll += scores.min(dim=0).values.sum().item()
        seen += len(labels)
    return total_nll / seen, 100.0 * correct / seen


def run_batches(model, loader, device, limit, nclass):
    seen = []
    for index, (text, lengths, labels) in enumerate(loader):
        if index >= limit:
            break
        score_all_labels(model, *split_teacher_forced(text, lengths, device), nclass)
        seen.append(labels)
    return torch.cat(seen) if seen else torch.empty(0, dtype=torch.long)


def calibrate(model, loader, device, batches, bias_correction, classes):
    model.eval()
    start = time.time()
    with torch.no_grad(), calibration_mode(model):
        labels = run_batches(model, loader, device, batches, len(classes))
    print(f'    activation calibration: {batches} batches in {time.time() - start:.1f}s')

    counts = torch.bincount(labels, minlength=len(classes)).tolist()
    spread = ' '.join(f'{c}={n}' for c, n in zip(classes, counts))
    print(f'    calibration sample:     {len(labels):,} training docs (random, unstratified)')
    print(f'                            {spread}')

    if bias_correction:
        # See ptq_disc.py: brevitas 0.13.0's bias_correction_mode does not support
        # QuantLSTM. Left opt-in so the failure is visible rather than silent.
        for module in model.modules():
            if hasattr(module, '_fast_cell'):
                module._fast_cell = None
        try:
            with torch.no_grad(), bias_correction_mode(model):
                run_batches(model, loader, device, batches, len(classes))
            print('    bias correction:        applied')
        except (AttributeError, RuntimeError) as exc:
            print(f'    bias correction:        SKIPPED -- unsupported for QuantLSTM '
                  f'({type(exc).__name__}: {exc})')
    return model


def apply_gpxq(model, loader, device, batches, nclass, algorithm):
    """Run GPFQ (or Qronos) on the decoder after activation calibration.

    GPxQ only handles nn.Linear/Conv, so the sole eligible module here is the
    decoder -- which is 66% of this model's weights, making it well worth doing.
    The embedding and the LSTM gates are untouched by construction.

    Note this must drive `model(...)`, not encode/decode directly: gpfq_mode
    replaces `forward` with a wrapper that runs the batch twice, once with
    quantization live to capture the layer's quantized input and once with it
    disabled to capture the float reference. Calling the halves separately would
    bypass that entirely.

    Every candidate label is fed, because at inference the decoder sees
    [h_t ; v_y] for all y -- that, not the training distribution, is the input
    distribution GPFQ should be minimising error over.
    """
    impl = GPXQ_ALGORITHMS[algorithm]
    start = time.time()
    with torch.no_grad(), gpfq_mode(model, algorithm_impl=impl, use_quant_activations=True,
                                    create_weight_orig=True) as gpfq:
        inner = gpfq.model
        print(f'    {algorithm}: {gpfq.num_layers} eligible layer(s)')
        for _ in range(gpfq.num_layers):
            for index, (text, lengths, _) in enumerate(loader):
                if index >= batches:
                    break
                inputs, _, _ = split_teacher_forced(text, lengths, device)
                for label in range(nclass):
                    inner(inputs, label)
            gpfq.update()
    print(f'    {algorithm}: {batches} batches x {nclass} labels in {time.time() - start:.1f}s')
    return model


def footprint(model, bit_width):
    """Weight footprint at fp32 vs at `bit_width`. Brevitas quantization is
    simulated, so this is what an int export *would* occupy."""
    quantized = sum(m.weight.numel() for m in model.modules()
                    if hasattr(m, 'weight_quant') and m.weight_quant.is_quant_enabled)
    total = sum(p.numel() for p in model.parameters())
    fp32_mb = total * 4 / 2 ** 20
    quant_mb = (quantized * bit_width / 8 + (total - quantized) * 4) / 2 ** 20
    print(f'  weights quantized: {quantized:,}/{total:,} ({100 * quantized / total:.1f}%)')
    print(f'  footprint: {fp32_mb:.2f} MB fp32 -> {quant_mb:.2f} MB at int{bit_width} '
          f'({fp32_mb / quant_mb:.2f}x, simulated)')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=ROOT / 'checkpoints/gen_lstm_agnews.pth')
    p.add_argument('--data', type=Path, default=ROOT / 'data/ag_news/processed/agnews.pt')
    p.add_argument('--out', type=Path, default=None,
                   help='default: checkpoints/gen_lstm_agnews_int<bit-width>.pth')
    p.add_argument('--bit-width', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--calib-batches', type=int, default=64)
    p.add_argument('--bias-correction', action='store_true',
                   help='opt-in; brevitas 0.13.0 does not support this for QuantLSTM')
    p.add_argument('--gpxq', choices=('none', 'gpfq', 'qronos'), default='none',
                   help='weight-error-correcting algorithm applied after calibration')
    p.add_argument('--gpxq-batches', type=int, default=32,
                   help='calibration batches for GPFQ/Qronos (each runs all labels, twice)')
    p.add_argument('--skip-float-reference', action='store_true')
    args = p.parse_args()

    if args.out is None:
        suffix = '' if args.gpxq == 'none' else f'_{args.gpxq}'
        args.out = ROOT / f'checkpoints/gen_lstm_agnews_int{args.bit_width}{suffix}.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(agnews_data.SEED)

    ckpt = torch.load(args.checkpoint, weights_only=False)
    state, vocab, classes = ckpt['state_dict'], ckpt['vocab'], ckpt['classes']
    trained = ckpt['args']
    nclass = len(classes)
    print(f'device: {device} | bit width: {args.bit_width} | vocab {len(vocab):,}')
    print(f'checkpoint: {args.checkpoint.name} (dev acc {ckpt["val_acc"]:.2f})')

    # Two views of the same data: packed for the original float model, padded for
    # the quantized one (QuantLSTM rejects PackedSequence).
    torch.manual_seed(agnews_data.SEED)
    _, packed = agnews_data.load(args.data, args.batch_size, GenTrain.collate)
    torch.manual_seed(agnews_data.SEED)
    _, padded = agnews_data.load(args.data, args.batch_size, collate_padded)

    dims = dict(vocab_size=len(vocab), embedding_dim=trained['word_emb_dim'],
                label_emb_dim=trained['label_emb_dim'], hidden_dim=trained['hid_dim'],
                nclass=nclass, n_layers=trained['layers'])

    print('\n== test set ==')
    start = time.time()
    float_model = GenModel(dims['vocab_size'], dims['embedding_dim'], dims['label_emb_dim'],
                           dims['hidden_dim'], dims['n_layers'], nclass, 0.0, True,
                           False, False, 'hidden', False, False).to(device)
    float_model.load_state_dict(state)
    criterion = nn.CrossEntropyLoss(reduction='none').to(device)
    fp32_nll, fp32_acc = GenTrain.evaluate(packed['test'], float_model, criterion, device, nclass)
    print(f'  1. float32             nll {fp32_nll:7.1f} | acc {fp32_acc:6.2f}   '
          f'[{time.time() - start:.0f}s]')

    if not args.skip_float_reference:
        start = time.time()
        reference = QuantGenModel(**dims, bit_width=args.bit_width, quant=False).to(device)
        load_gen_float_weights(reference, state)
        nll, acc = evaluate(padded['test'], reference, device, nclass)
        print(f'  2. float reference     nll {nll:7.1f} | acc {acc:6.2f}  '
              f'({acc - fp32_acc:+.2f} vs float32)   [{time.time() - start:.0f}s]')
        del reference

    print(f'\n== calibrating on {args.calib_batches} training batches ==')
    quantized = QuantGenModel(**dims, bit_width=args.bit_width, quant=True)
    load_gen_float_weights(quantized, state)
    quantized.to(device)
    calibrate(quantized, padded['train'], device, args.calib_batches,
              args.bias_correction, classes)

    if args.gpxq != 'none':
        apply_gpxq(quantized, padded['train'], device, args.gpxq_batches, nclass, args.gpxq)

    print('\n== quantized ==')
    start = time.time()
    dev_nll, dev_acc = evaluate(padded['valid'], quantized, device, nclass)
    print(f'     dev                 nll {dev_nll:7.1f} | acc {dev_acc:6.2f}   [{time.time() - start:.0f}s]')
    start = time.time()
    test_nll, test_acc = evaluate(padded['test'], quantized, device, nclass)
    tag = f'int{args.bit_width}' + ('' if args.gpxq == 'none' else f'+{args.gpxq}')
    print(f'  3. {tag:18s} nll {test_nll:7.1f} | acc {test_acc:6.2f}  '
          f'({test_acc - fp32_acc:+.2f} vs float32)   [{time.time() - start:.0f}s]')

    print()
    footprint(quantized, args.bit_width)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': quantized.state_dict(), 'vocab': vocab, 'classes': classes,
                'args': vars(args) | dims, 'test_acc': test_acc, 'dev_acc': dev_acc,
                'fp32_test_acc': fp32_acc}, args.out)
    print(f'\nsaved {args.out}')


if __name__ == '__main__':
    main()
