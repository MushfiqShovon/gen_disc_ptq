"""Rebuild the PTQ results table by parsing logs/ptq_*.log.

New runs write their own row directly (see results.py), so this is only needed to
back-fill runs that predate that, or to reconstruct the table if the CSV is lost.

    python quantization/collect_results.py            # merge into the CSV
    python quantization/collect_results.py --fresh    # discard the CSV and rebuild
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import results  # noqa: E402

# Log lines look like:
#   1. float32             loss 0.4090 | acc  91.99   [0s]        (disc)
#   1. float32             nll   209.2 | acc  90.67   [6s]        (gen)
#   3. int4+gpfq           loss 1.1255 | acc  73.16  (-18.83 ...)
NUMBER = r'(?:loss|nll)\s+([0-9.]+)\s*\|\s*acc\s+([0-9.]+)'
PATTERNS = {
    'fp32': re.compile(r'1\. float32\s+' + NUMBER),
    'float_ref': re.compile(r'2\. float reference\s+' + NUMBER),
    'dev': re.compile(r'^\s+dev\s+' + NUMBER, re.M),
    # Earlier runs tagged the line "int8 test"; once --gpxq arrived it became
    # "int8" / "int8+gpfq". Accept both so old logs still parse.
    'quant': re.compile(r'3\. int(\d+)(?:\+(\w+))?(?:\s+test)?\s+' + NUMBER),
    'calib': re.compile(r'calibration sample:\s+([0-9,]+) training docs'),
    'gpxq_batches': re.compile(r'^\s+(?:gpfq|qronos):\s+(\d+) batches', re.M),
    'footprint': re.compile(r'footprint:\s+([0-9.]+) MB fp32 -> ([0-9.]+) MB .*?\(([0-9.]+)x'),
}


def parse(path):
    text = path.read_text()
    quant = PATTERNS['quant'].search(text)
    fp32 = PATTERNS['fp32'].search(text)
    if not quant or not fp32:
        return None

    bits, gpxq, metric, acc = quant.groups()
    fp32_acc = float(fp32.group(2))
    row = {
        'model': 'gen' if '_gen_' in path.name else 'disc',
        'bit_width': int(bits),
        'gpxq': gpxq or 'none',
        'fp32_test_acc': f'{fp32_acc:.2f}',
        'test_acc': f'{float(acc):.2f}',
        'delta_vs_fp32': f'{float(acc) - fp32_acc:+.2f}',
        'test_metric': metric,
    }

    ref = PATTERNS['float_ref'].search(text)
    row['float_ref_test_acc'] = f'{float(ref.group(2)):.2f}' if ref else ''
    dev = PATTERNS['dev'].search(text)
    row['dev_acc'] = f'{float(dev.group(2)):.2f}' if dev else ''
    calib = PATTERNS['calib'].search(text)
    row['calib_docs'] = calib.group(1).replace(',', '') if calib else ''
    batches = PATTERNS['gpxq_batches'].search(text)
    row['gpxq_batches'] = batches.group(1) if batches else ''
    foot = PATTERNS['footprint'].search(text)
    if foot:
        row['fp32_mb'], row['quant_mb'], row['compression'] = foot.groups()
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--logs', type=Path, default=ROOT / 'logs')
    p.add_argument('--out', type=Path, default=ROOT / 'results/ptq_results.csv')
    p.add_argument('--fresh', action='store_true', help='ignore any existing CSV')
    args = p.parse_args()

    rows = [] if args.fresh else results.read(args.out)
    index = {tuple(str(r.get(k, '')) for k in results.KEY): r for r in rows}

    parsed, skipped = 0, []
    for log in sorted(args.logs.glob('ptq_*.log')):
        row = parse(log)
        if row is None:
            skipped.append(log.name)
            continue
        index[tuple(str(row.get(k, '')) for k in results.KEY)] = row
        parsed += 1

    results.write(args.out, list(index.values()))
    print(f'parsed {parsed} log(s) -> {args.out}')
    if skipped:
        print(f'skipped (no result line): {", ".join(skipped)}')

    table = results.read(args.out)
    print(f'\n{len(table)} rows:\n')
    header = ['model', 'bit_width', 'gpxq', 'fp32_test_acc', 'test_acc', 'delta_vs_fp32']
    print('  ' + ''.join(f'{h:>15s}' for h in header))
    for row in table:
        print('  ' + ''.join(f'{row.get(h, ""):>15s}' for h in header))


if __name__ == '__main__':
    main()
