"""Shared results table for the PTQ experiments.

Every PTQ run appends a row here, keyed on (model, bit_width, gpxq), so re-running
a configuration *replaces* its row rather than duplicating it. That makes the CSV
safe to regenerate incrementally: run whichever cells you want, in any order, and
the table stays consistent.
"""

import csv
from pathlib import Path

FIELDS = [
    'model',              # gen | disc
    'bit_width',
    'gpxq',               # none | gpfq | qronos
    'fp32_test_acc',
    'float_ref_test_acc',  # quantizers off -- isolates the Brevitas port
    'dev_acc',
    'test_acc',
    'delta_vs_fp32',
    'test_metric',        # cross-entropy for disc, summed NLL for gen
    'calib_docs',
    'gpxq_batches',
    'fp32_mb',
    'quant_mb',
    'compression',
]

KEY = ('model', 'bit_width', 'gpxq')


def _sort_key(row):
    try:
        bits = -int(row.get('bit_width') or 0)
    except ValueError:
        bits = 0
    return (row.get('model', ''), bits, row.get('gpxq', ''))


# The calibration-sensitivity experiment records the same measurements plus the
# scheme, keyed additionally on it, into its own CSV (results/calib_sensitivity.csv).
CALIB_FIELDS = FIELDS[:3] + ['calib_scheme'] + FIELDS[3:]
CALIB_KEY = KEY + ('calib_scheme',)


def read(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def write(path, rows, fields=FIELDS):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=_sort_key)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, '') for field in fields} for row in rows])
    return path


def record(path, row, fields=FIELDS, key=KEY):
    """Insert `row`, replacing any existing row with the same key tuple."""
    row_key = tuple(str(row.get(field, '')) for field in key)
    kept = [r for r in read(path)
            if tuple(str(r.get(field, '')) for field in key) != row_key]
    kept.append(row)
    return write(path, kept, fields)
