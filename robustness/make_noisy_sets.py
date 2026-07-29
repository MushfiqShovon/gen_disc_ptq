"""One-off generator of QWERTY-typo-corrupted test sets.

Applies keyboard-adjacent character substitution to the *cleaned* test text at
several noise levels, runs the standard preprocessing (same spaCy tokenizer,
same training vocab, same 80-token truncation), and saves each level once to
data/ag_news/processed/noise/test_p<NN>.pt. Every model evaluates the identical
token ids for a given level.

Noise model:
- Each *letter* is independently substituted with probability p by a uniformly
  random QWERTY-adjacent letter. Letters only, letter -> letter: token
  boundaries are preserved, so this measures semantic corruption rather than
  tokenization/length artifacts. Digits and punctuation are untouched.
- Per-document seeding (SEED + doc index) makes generation order-independent,
  and reusing the same uniform draws across levels *couples* them: characters
  corrupted at p=0.02 are a subset of those corrupted at p=0.05, so curves
  across levels are monotone-comparable rather than independently resampled.

    python robustness/make_noisy_sets.py

Deterministic: regeneration with --force is byte-identical.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import spacy
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))

from agnews_data import SEED  # noqa: E402
from preprocess import UNK, encode, flatten, read_split, tokenize  # noqa: E402

LEVELS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30)

# Lowercase QWERTY adjacency (letters only, including diagonals).
QWERTY = {
    'q': 'wa', 'w': 'qeas', 'e': 'wrsd', 'r': 'etdf', 't': 'ryfg', 'y': 'tugh',
    'u': 'yihj', 'i': 'uojk', 'o': 'ipkl', 'p': 'ol',
    'a': 'qwsz', 's': 'awedzx', 'd': 'serfxc', 'f': 'drtgcv', 'g': 'ftyhvb',
    'h': 'gyujbn', 'j': 'huiknm', 'k': 'jiolm', 'l': 'kop',
    'z': 'asx', 'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb', 'b': 'vghn',
    'n': 'bhjm', 'm': 'njk',
}


def corrupt(text, p, doc_seed):
    """Substitute each letter with prob p by a random QWERTY neighbour.

    The rng is reset per document, and the same two draws (threshold, choice)
    are made for every character regardless of p -- that is what couples the
    noise levels: a character flips at every p above its threshold draw.
    """
    rng = np.random.RandomState(doc_seed)
    thresholds = rng.random_sample(len(text))
    choices = rng.random_sample(len(text))
    out = []
    changed = 0
    for ch, threshold, choice in zip(text, thresholds, choices):
        neighbours = QWERTY.get(ch)
        if neighbours is not None and threshold < p:
            out.append(neighbours[int(choice * len(neighbours))])
            changed += 1
        else:
            out.append(ch)
    return ''.join(out), changed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=ROOT / 'data/ag_news')
    p.add_argument('--cache', type=Path, default=ROOT / 'data/ag_news/processed/agnews.pt')
    p.add_argument('--out-dir', type=Path, default=ROOT / 'data/ag_news/processed/noise')
    p.add_argument('--max-len', type=int, default=80)
    p.add_argument('--force', action='store_true')
    args = p.parse_args()

    cache = torch.load(args.cache, weights_only=False)
    vocab = cache['vocab']
    stoi = {w: i for i, w in enumerate(vocab)}
    nlp = spacy.load('en_core_web_sm')

    # read_split returns *cleaned* text (lowercased, entities fixed) -- the noise
    # is applied to exactly what the model would otherwise see.
    texts, labels = read_split(args.data_dir / 'test.csv')
    total_letters = sum(sum(ch in QWERTY for ch in t) for t in texts)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f'test: {len(texts):,} docs | {total_letters:,} substitutable letters')
    for level in LEVELS:
        out = args.out_dir / f'test_p{round(level * 100):02d}.pt'
        if out.exists() and not args.force:
            print(f'p={level:.2f} exists, skipping ({out.name})')
            continue

        noisy, changed = [], 0
        for index, text in enumerate(texts):
            corrupted, n = corrupt(text, level, SEED + index)
            noisy.append(corrupted)
            changed += n

        tokenized = tokenize(noisy, nlp, args.max_len)
        sequences = encode(tokenized, stoi)
        split = flatten(sequences, labels)
        unk_rate = float(np.mean(split['flat'] == stoi[UNK]))
        char_rate = changed / total_letters
        split['meta'] = {'noise': 'qwerty_adjacent_substitution', 'level': level,
                         'seed': SEED, 'char_sub_rate': char_rate,
                         'unk_rate': unk_rate, 'source': 'test'}
        torch.save(split, out)
        lengths = np.diff(split['offsets'])
        print(f'p={level:.2f}  char-sub {char_rate * 100:5.2f}% | unk {unk_rate * 100:5.2f}% '
              f'| mean len {lengths.mean():5.1f} -> {out.name}')

    print('\nsample (p=0.10):')
    sample, _ = corrupt(texts[0], 0.10, SEED)
    print(f'  clean: {texts[0][:110]}')
    print(f'  noisy: {sample[:110]}')


if __name__ == '__main__':
    main()
