"""Shared AG News plumbing for the generative and discriminative classifiers.

Both training scripts read the same cache written by preprocess.py and build
their loaders here, so a head-to-head comparison differs only in the model and
the collate function -- never in the data, the split, or the shuffling.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

SEED = 2021


class AGNews(Dataset):
    """Ragged token-id sequences stored as one flat buffer plus offsets."""

    def __init__(self, split):
        self.flat = torch.from_numpy(split['flat'].astype(np.int64))
        self.offsets = split['offsets']
        self.labels = torch.from_numpy(split['labels'])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.flat[self.offsets[i]:self.offsets[i + 1]], self.labels[i]


def load(path, batch_size, collate_fn):
    """Returns (cache, {split: DataLoader})."""
    cache = torch.load(path, weights_only=False)
    loaders = {
        split: DataLoader(AGNews(cache[split]), batch_size=batch_size,
                          shuffle=(split == 'train'), collate_fn=collate_fn)
        for split in ('train', 'valid', 'test')
    }
    return cache, loaders
