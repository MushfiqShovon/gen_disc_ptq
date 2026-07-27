"""Static post-training quantization of the discriminative classifier (Brevitas).

Quantizes weights *and* activations to `--bit-width` (default 8), calibrating the
activation scales on a sample of the training set.

Three numbers are reported so the quantization effect can be separated from the
harness change needed to get there:

  1. float32          the trained model exactly as evaluated in training
  2. float reference  same weights in the Brevitas graph, every quantizer off
                      (padded + masked instead of packed) -- isolates the port
  3. quantized        the same graph with weights and activations quantized

(3) vs (2) is the quantization effect proper; (3) vs (1) is end to end.

    python quantization/ptq_disc.py --bit-width 8
"""

import argparse
import copy
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from brevitas.graph.calibrate import bias_correction_mode, calibration_mode
from brevitas.graph.gpfq import GPFQ, gpfq_mode
from brevitas.graph.qronos import Qronos

GPXQ_ALGORITHMS = {'gpfq': GPFQ, 'qronos': Qronos}

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import agnews_data  # noqa: E402
from DiscTrain import collate, evaluate  # noqa: E402
from models import DiscModel  # noqa: E402
from quant_models import QuantDiscModel, load_float_weights  # noqa: E402


def run_batches(model, loader, device, limit):
    """Forward `limit` batches, discarding outputs. Used inside the Brevitas
    context managers, which collect statistics via hooks.

    Returns the labels seen, purely so the caller can report what the calibration
    sample actually contained.
    """
    seen = []
    for index, (text, lengths, labels) in enumerate(loader):
        if index >= limit:
            break
        model(text.to(device), lengths)
        seen.append(labels)
    return torch.cat(seen) if seen else torch.empty(0, dtype=torch.long)


def calibrate(model, loader, device, batches, bias_correction, classes):
    model.eval()
    start = time.time()
    with torch.no_grad(), calibration_mode(model):
        labels = run_batches(model, loader, device, batches)
    print(f'    activation calibration: {batches} batches in {time.time() - start:.1f}s')

    # The calibration sample is whatever the *shuffled* training loader yields
    # first -- drawn at random, not stratified by class. Reported so the actual
    # composition is on the record for each run.
    counts = torch.bincount(labels, minlength=len(classes)).tolist()
    spread = ' '.join(f'{c}={n}' for c, n in zip(classes, counts))
    print(f'    calibration sample:     {len(labels):,} training docs (random, unstratified)')
    print(f'                            {spread}')

    if bias_correction:
        # brevitas 0.13.0's bias_correction_mode is not compatible with QuantLSTM.
        # Two separate faults: its module scan trips over the lazily built
        # `_fast_cell` (whose fused act proxy has no `is_quant_enabled`), and once
        # past that, disabling output quant starves the LSTM's Int32Bias of the
        # input scale it needs. Verified that the same graph without the QuantLSTM
        # bias-corrects fine. Left opt-in so the failure is visible, not silent.
        for module in model.modules():
            if hasattr(module, '_fast_cell'):
                module._fast_cell = None
        start = time.time()
        try:
            with torch.no_grad(), bias_correction_mode(model):
                run_batches(model, loader, device, batches)
            print(f'    bias correction:        {batches} batches in {time.time() - start:.1f}s')
        except (AttributeError, RuntimeError) as exc:
            print(f'    bias correction:        SKIPPED -- unsupported for QuantLSTM '
                  f'in brevitas {__import__("brevitas").__version__} ({type(exc).__name__}: {exc})')
    return model


def apply_gpxq(model, loader, device, batches, algorithm):
    """Run GPFQ (or Qronos) after activation calibration.

    Expect almost nothing here. GPxQ only handles nn.Linear/Conv, and in this
    model that is the 400-weight classifier -- 0.014% of the parameters. The
    embedding (97%) and the LSTM gates (3%) are ineligible by construction, so
    there is essentially no weight error for it to redistribute. Contrast the
    generative model, where the decoder makes 66% of the weights eligible.
    """
    impl = GPXQ_ALGORITHMS[algorithm]
    start = time.time()
    with torch.no_grad(), gpfq_mode(model, algorithm_impl=impl, use_quant_activations=True,
                                    create_weight_orig=True) as gpfq:
        inner = gpfq.model
        eligible = sum(m.weight.numel() for m in model.modules()
                       if isinstance(m, nn.Linear) and hasattr(m, 'weight_quant'))
        total = sum(p.numel() for p in model.parameters())
        print(f'    {algorithm}: {gpfq.num_layers} eligible layer(s), '
              f'{eligible:,}/{total:,} weights ({100 * eligible / total:.3f}%)')
        for _ in range(gpfq.num_layers):
            for index, (text, lengths, _) in enumerate(loader):
                if index >= batches:
                    break
                inner(text.to(device), lengths)
            gpfq.update()
    print(f'    {algorithm}: {batches} batches in {time.time() - start:.1f}s')
    return model


def footprint(model, bit_width):
    """Weight footprint at fp32 vs at `bit_width`.

    Brevitas quantization is simulated -- the saved tensors are still float32, so
    this is what an int8 export *would* occupy, not the size of the .pth on disk.
    """
    quantized = sum(m.weight.numel() for m in model.modules()
                    if hasattr(m, 'weight_quant') and m.weight_quant.is_quant_enabled)
    total = sum(p.numel() for p in model.parameters())
    fp32_mb = total * 4 / 2 ** 20
    quant_mb = (quantized * bit_width / 8 + (total - quantized) * 4) / 2 ** 20
    print(f'  weights quantized: {quantized:,}/{total:,} '
          f'({100 * quantized / total:.1f}%)')
    print(f'  footprint: {fp32_mb:.2f} MB fp32 -> {quant_mb:.2f} MB at int{bit_width} '
          f'({fp32_mb / quant_mb:.2f}x, simulated)')


def report(name, loader, model, criterion, device, baseline=None):
    start = time.time()
    loss, acc = evaluate(loader, model, criterion, device)
    delta = '' if baseline is None else f'  ({acc - baseline:+.2f} vs float32)'
    print(f'  {name:22s} loss {loss:.4f} | acc {acc:6.2f}{delta}   [{time.time() - start:.0f}s]')
    return acc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=ROOT / 'checkpoints/disc_lstm_agnews.pth')
    p.add_argument('--data', type=Path, default=ROOT / 'data/ag_news/processed/agnews.pt')
    p.add_argument('--out', type=Path, default=None,
                   help='default: checkpoints/disc_lstm_agnews_int<bit-width>.pth')
    p.add_argument('--bit-width', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--calib-batches', type=int, default=64,
                   help='training batches used to calibrate activation scales')
    p.add_argument('--bias-correction', action='store_true',
                   help='opt-in; brevitas 0.13.0 does not support this for QuantLSTM')
    p.add_argument('--gpxq', choices=('none', 'gpfq', 'qronos'), default='none',
                   help='weight-error-correcting algorithm applied after calibration')
    p.add_argument('--gpxq-batches', type=int, default=32)
    p.add_argument('--skip-float-reference', action='store_true',
                   help='skip the quantizers-off sanity evaluation (it is slow)')
    args = p.parse_args()

    # Name the checkpoint after the bit width so an ablation sweep does not
    # overwrite its own earlier runs.
    if args.out is None:
        suffix = '' if args.gpxq == 'none' else f'_{args.gpxq}'
        args.out = ROOT / f'checkpoints/disc_lstm_agnews_int{args.bit_width}{suffix}.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(agnews_data.SEED)

    ckpt = torch.load(args.checkpoint, weights_only=False)
    state, vocab, classes = ckpt['state_dict'], ckpt['vocab'], ckpt['classes']
    trained = ckpt['args']
    print(f'device: {device} | bit width: {args.bit_width} | vocab {len(vocab):,}')
    print(f'checkpoint: {args.checkpoint.name} (dev acc {ckpt["val_acc"]:.2f})')

    cache, loaders = agnews_data.load(args.data, args.batch_size, collate)
    criterion = nn.CrossEntropyLoss().to(device)

    dims = dict(vocab_size=len(vocab), embedding_dim=trained['word_emb_dim'],
                hidden_dim=trained['hid_dim'], output_dim=len(classes),
                n_layers=trained['layers'])

    print('\n== test set ==')
    float_model = DiscModel(dims['vocab_size'], dims['embedding_dim'], dims['hidden_dim'],
                            dims['output_dim'], dims['n_layers'], False, 0.0, args.bit_width)
    float_model.load_state_dict(state)
    fp32_acc = report('1. float32', loaders['test'], float_model.to(device), criterion, device)

    if not args.skip_float_reference:
        reference = QuantDiscModel(**dims, bit_width=args.bit_width, quant=False)
        load_float_weights(reference, state)
        report('2. float reference', loaders['test'], reference.to(device), criterion, device, fp32_acc)

    print(f'\n== calibrating on {args.calib_batches} training batches ==')
    quantized = QuantDiscModel(**dims, bit_width=args.bit_width, quant=True)
    load_float_weights(quantized, state)
    quantized.to(device)
    calibrate(quantized, loaders['train'], device, args.calib_batches,
              bias_correction=args.bias_correction, classes=classes)

    if args.gpxq != 'none':
        apply_gpxq(quantized, loaders['train'], device, args.gpxq_batches, args.gpxq)

    print('\n== quantized ==')
    tag = f'int{args.bit_width}' + ('' if args.gpxq == 'none' else f'+{args.gpxq}')
    dev_acc = report('   dev', loaders['valid'], quantized, criterion, device)
    test_acc = report(f'3. {tag}', loaders['test'], quantized, criterion, device, fp32_acc)

    print()
    footprint(quantized, args.bit_width)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': quantized.state_dict(), 'vocab': vocab, 'classes': classes,
                'args': vars(args) | dims, 'test_acc': test_acc, 'dev_acc': dev_acc,
                'fp32_test_acc': fp32_acc}, args.out)
    print(f'\nsaved {args.out}')


if __name__ == '__main__':
    main()
