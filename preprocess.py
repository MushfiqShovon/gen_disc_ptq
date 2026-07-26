"""One-off preprocessing for AG News.

Cleans the raw CSVs, tokenises with spaCy, builds a vocabulary from the training
split and writes every split out as token ids so that training never has to
tokenise again.

    python preprocess.py

Output: data/ag_news/processed/agnews.pt
"""

import argparse
import html
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
import torch

UNK, BOS, EOS = '<unk>', '<bos>', '<eos>'
SPECIALS = [UNK, BOS, EOS]

# AG News uses "\\" as a paragraph break and is littered with half-escaped HTML
# entities ("#39;", "quot;") that lost their leading ampersand.
_BARE_NUMERIC = re.compile(r'(?<!&)#(\d+);')
_BARE_NAMED = re.compile(r'(?<!&)\b(quot|amp|lt|gt|nbsp|apos);')
_TAG = re.compile(r'<[^>]{1,40}>')
_SPACE = re.compile(r'\s+')


def clean(text):
    text = text.replace('\\', ' ')
    text = _BARE_NUMERIC.sub(r'&#\1;', text)
    text = _BARE_NAMED.sub(r'&\1;', text)
    text = html.unescape(html.unescape(text))
    text = _TAG.sub(' ', text)
    return _SPACE.sub(' ', text).strip().lower()


def read_split(path):
    """AG News CSVs are `label, title, description` with labels in 1..4."""
    df = pd.read_csv(path, header=None, names=['label', 'title', 'desc'])
    texts = (df.title.fillna('') + '. ' + df.desc.fillna('')).map(clean)
    return texts.tolist(), (df.label.to_numpy() - 1).astype(np.int64)


def tokenize(texts, nlp, max_len):
    return [[t.text for t in doc][:max_len] for doc in nlp.tokenizer.pipe(texts, batch_size=1000)]


def build_vocab(tokenized, max_size, min_freq):
    counts = Counter(tok for toks in tokenized for tok in toks)
    kept = [w for w, c in counts.most_common(max_size - len(SPECIALS)) if c >= min_freq]
    itos = SPECIALS + kept
    return itos, {w: i for i, w in enumerate(itos)}


def encode(tokenized, stoi):
    """Wrap each sentence in <bos>/<eos> so the LM predicts every real token."""
    unk, bos, eos = stoi[UNK], stoi[BOS], stoi[EOS]
    return [np.fromiter([bos] + [stoi.get(t, unk) for t in toks] + [eos],
                        dtype=np.int32) for toks in tokenized]


def flatten(sequences, labels):
    """Ragged sequences -> one flat buffer + offsets, so loading is a single read."""
    lengths = np.fromiter((len(s) for s in sequences), dtype=np.int64, count=len(sequences))
    return {
        'flat': np.concatenate(sequences),
        'offsets': np.concatenate([[0], np.cumsum(lengths)]),
        'labels': labels,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=Path('data/ag_news'))
    p.add_argument('--out', type=Path, default=Path('data/ag_news/processed/agnews.pt'))
    p.add_argument('--max-len', type=int, default=80, help='truncate sentences to this many tokens')
    p.add_argument('--max-vocab', type=int, default=40000)
    p.add_argument('--min-freq', type=int, default=5)
    args = p.parse_args()

    nlp = spacy.load('en_core_web_sm')
    classes = (args.data_dir / 'classes.txt').read_text().split()

    raw = {}
    for split in ('train', 'valid', 'test'):
        start = time.time()
        texts, labels = read_split(args.data_dir / f'{split}.csv')
        raw[split] = (tokenize(texts, nlp, args.max_len), labels)
        print(f'{split:5s} {len(texts):>7,} docs tokenized in {time.time() - start:5.1f}s')

    itos, stoi = build_vocab(raw['train'][0], args.max_vocab, args.min_freq)
    print(f'vocab: {len(itos):,} types (max {args.max_vocab:,}, min_freq {args.min_freq})')

    out = {'vocab': itos, 'classes': classes, 'max_len': args.max_len}
    for split, (tokenized, labels) in raw.items():
        sequences = encode(tokenized, stoi)
        out[split] = flatten(sequences, labels)
        lengths = np.diff(out[split]['offsets'])
        unk_rate = np.mean(out[split]['flat'] == stoi[UNK]) * 100
        print(f'{split:5s} {len(labels):>7,} seqs | mean len {lengths.mean():5.1f} '
              f'| max {lengths.max():3d} | unk {unk_rate:4.1f}%')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f'\nwrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
