"""One-off sampler for the calibration-sensitivity experiment.

Builds class-imbalanced calibration sets from the *training* split and saves them
to data/ag_news/processed/calib/<scheme>.pt for reuse. Every scheme has the same
total budget -- only the class composition varies:

    all_class      25 / 25 / 25 / 25 %
    three_class    33 / 33 / 33 /  0 %
    two_class      50 / 50 /  0 /  0 %
    one_class     100 /  0 /  0 /  0 %

Class 1..4 = World, Sports, Business, Sci/Tech (label ids 0..3).

Design details that matter for the experiment:
- One seeded permutation per class; every scheme takes a *prefix* of it. Schemes
  therefore share documents wherever their classes overlap, so differences
  between schemes are composition, not resampling noise.
- The saved document order is shuffled (seeded), so any batch-prefix of a file
  reflects the scheme's mix -- required because GPFQ consumes a prefix.
- Idempotent: existing files are skipped unless --force. Fully deterministic:
  re-generation with --force produces byte-identical content.

    python quantization/make_calib_sets.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))

from agnews_data import SEED  # noqa: E402

SCHEMES = {
    'all_class': (0.25, 0.25, 0.25, 0.25),
    'three_class': (1 / 3, 1 / 3, 1 / 3, 0.0),
    'two_class': (0.5, 0.5, 0.0, 0.0),
    'one_class': (1.0, 0.0, 0.0, 0.0),
}


def counts_for(fractions, total):
    """Integer per-class counts that sum exactly to `total` (largest remainder)."""
    raw = [f * total for f in fractions]
    counts = [int(r) for r in raw]
    for index in sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True):
        if sum(counts) == total:
            break
        counts[index] += 1
    return counts


def subset(split, indices):
    """Extract `indices` from a flat/offsets/labels split into the same format."""
    offsets = split['offsets']
    sequences = [split['flat'][offsets[i]:offsets[i + 1]] for i in indices]
    lengths = np.fromiter((len(s) for s in sequences), dtype=np.int64, count=len(sequences))
    return {
        'flat': np.concatenate(sequences),
        'offsets': np.concatenate([[0], np.cumsum(lengths)]),
        'labels': split['labels'][indices],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT / 'data/ag_news/processed/agnews.pt')
    p.add_argument('--out-dir', type=Path, default=ROOT / 'data/ag_news/processed/calib')
    p.add_argument('--total', type=int, default=8192,
                   help='documents per scheme (fixed across schemes by design)')
    p.add_argument('--force', action='store_true', help='regenerate existing files')
    args = p.parse_args()

    cache = torch.load(args.data, weights_only=False)
    train, classes = cache['train'], cache['classes']
    labels = train['labels']
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # One permutation per class, shared by all schemes (prefix property).
    rng = np.random.RandomState(SEED)
    class_pools = [rng.permutation(np.flatnonzero(labels == c)) for c in range(len(classes))]

    for scheme, fractions in SCHEMES.items():
        out = args.out_dir / f'{scheme}.pt'
        if out.exists() and not args.force:
            print(f'{scheme:12s} exists, skipping ({out})')
            continue

        counts = counts_for(fractions, args.total)
        indices = np.concatenate([pool[:n] for pool, n in zip(class_pools, counts) if n])
        # Shuffle the saved order so any batch-prefix matches the scheme's mix.
        indices = indices[np.random.RandomState(SEED).permutation(len(indices))]

        data = subset(train, indices)
        data['meta'] = {'scheme': scheme, 'fractions': fractions, 'counts': counts,
                        'classes': classes, 'seed': SEED, 'source': 'train',
                        'source_indices': indices}
        torch.save(data, out)

        spread = ' '.join(f'{c}={n}' for c, n in zip(classes, counts))
        print(f'{scheme:12s} {len(indices):,} docs | {spread} -> {out}')


if __name__ == '__main__':
    main()
