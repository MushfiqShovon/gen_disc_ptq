"""One-off download + split for AG News.

Fetches the original AG News CSVs (`label, title, description`; labels 1..4) and
carves a stratified validation set out of the 120k training rows.

    python prepare_data.py

Output: data/ag_news/{train,valid,test}.csv + classes.txt
Run preprocess.py next.
"""

import argparse
import urllib.request
from pathlib import Path

import pandas as pd

BASE = 'https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv'
COLUMNS = ['label', 'title', 'desc']
SEED = 2021


def download(name, dest):
    if dest.exists():
        print(f'{dest} already present, skipping download')
        return
    print(f'downloading {name} -> {dest}')
    urllib.request.urlretrieve(f'{BASE}/{name}', dest)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=Path('data/ag_news'))
    p.add_argument('--valid-per-class', type=int, default=2500)
    args = p.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)

    download('train.csv', args.data_dir / 'train_full.csv')
    download('test.csv', args.data_dir / 'test.csv')
    download('classes.txt', args.data_dir / 'classes.txt')

    full = pd.read_csv(args.data_dir / 'train_full.csv', header=None, names=COLUMNS)
    valid = (full.groupby('label', group_keys=False)[COLUMNS]
                 .apply(lambda g: g.sample(n=args.valid_per_class, random_state=SEED)))
    train = full.drop(valid.index)

    for name, df in (('train', train), ('valid', valid)):
        df = df.sample(frac=1.0, random_state=SEED)  # shuffle
        df.to_csv(args.data_dir / f'{name}.csv', header=False, index=False)
        counts = df.label.value_counts().sort_index().tolist()
        print(f'{name:5s} {len(df):>7,} rows | per class {counts}')

    test = pd.read_csv(args.data_dir / 'test.csv', header=None, names=COLUMNS)
    print(f'test  {len(test):>7,} rows | per class {test.label.value_counts().sort_index().tolist()}')


if __name__ == '__main__':
    main()
