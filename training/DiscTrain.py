"""Train the discriminative LSTM classifier (DiscModel) on AG News.

Follows Yogatama et al. (2017) section 2.1 and Ding & Gimpel (2019) section 2:
encode the document with a one-layer unidirectional LSTM, take the *average* of
its hidden states as the document representation, and put a softmax over labels
on top. Trained to maximise sum log p(y | x).

Shares preprocess.py's cache and agnews_data.py's loaders with GenTrain.py, so
the two classifiers are directly comparable.

    python DiscTrain.py --epochs 10
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

import agnews_data
from agnews_data import SEED
from models import DiscModel

# Paths default to the repo root, not the caller's cwd, so the scripts in
# training/ behave the same however they are invoked.
ROOT = Path(__file__).resolve().parents[1]


def collate(batch):
    """Pad to the longest document in the batch and keep the true lengths."""
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences])
    return pad_sequence(sequences, batch_first=True), lengths, torch.stack(labels)


@torch.no_grad()
def evaluate(loader, model, criterion, device):
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0

    for text, lengths, labels in loader:
        text, labels = text.to(device), labels.to(device)
        logits = model(text, lengths)

        total_loss += criterion(logits, labels).item() * len(labels)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += len(labels)

    return total_loss / seen, 100.0 * correct / seen


def train_epoch(loader, model, criterion, optimizer, device, epoch, args):
    model.train()
    total_loss, total_correct, seen = 0.0, 0, 0
    window = (0.0, 0, 0)
    start = time.time()

    for step, (text, lengths, labels) in enumerate(loader, 1):
        text, labels = text.to(device), labels.to(device)
        logits = model(text, lengths)
        loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += len(labels)

        if step % args.log_interval == 0:
            wl, wc, ws = window
            ms = (time.time() - start) * 1000 / args.log_interval
            print(f'| epoch {epoch:2d} | {step:5d}/{len(loader):5d} batches '
                  f'| {ms:6.1f} ms/batch | loss {(total_loss - wl) / (seen - ws):5.3f} '
                  f'| acc {100.0 * (total_correct - wc) / (seen - ws):5.2f} |')
            window = (total_loss, total_correct, seen)
            start = time.time()

    return total_loss / seen, 100.0 * total_correct / seen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT / 'data/ag_news/processed/agnews.pt')
    p.add_argument('--out', type=Path, default=ROOT / 'checkpoints/disc_lstm_agnews.pth')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--word-emb-dim', type=int, default=100)
    p.add_argument('--hid-dim', type=int, default=100)
    p.add_argument('--layers', type=int, default=1)
    p.add_argument('--bidirectional', action='store_true', help='papers use unidirectional')
    # The papers' reference implementation defaults to 0, but this model overfits
    # badly without it (best dev at epoch 3, then a widening gap). 0.5 is worth
    # +1.2 test points here; pass --dropout 0 to reproduce the paper-faithful run.
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--clip', type=float, default=1.0)
    p.add_argument('--log-interval', type=int, default=200)
    args = p.parse_args()

    torch.manual_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    cache, loaders = agnews_data.load(args.data, args.batch_size, collate)
    vocab, classes = cache['vocab'], cache['classes']
    print(f'vocab {len(vocab):,} | classes {classes}')
    for split, loader in loaders.items():
        print(f'{split:5s} {len(loader.dataset):>7,} examples | {len(loader):>5,} batches')

    model = DiscModel(len(vocab), args.word_emb_dim, args.hid_dim, len(classes),
                      args.layers, args.bidirectional, args.dropout, bit_witdh=8).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best_acc, best_epoch = 0.0, -1

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(loaders['train'], model, criterion,
                                            optimizer, device, epoch, args)
        val_loss, val_acc = evaluate(loaders['valid'], model, criterion, device)

        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save({'state_dict': model.state_dict(), 'vocab': vocab,
                        'classes': classes, 'args': vars(args), 'val_acc': val_acc},
                       args.out)

        print('-' * 92)
        print(f'| end of epoch {epoch:2d} | {time.time() - epoch_start:6.1f}s '
              f'| train loss {train_loss:5.3f} | train acc {train_acc:5.2f} '
              f'| valid loss {val_loss:5.3f} | valid acc {val_acc:5.2f} |')
        print('-' * 92)

    print(f'\nbest validation accuracy {best_acc:.2f} at epoch {best_epoch}')

    model.load_state_dict(torch.load(args.out, weights_only=False)['state_dict'])
    test_loss, test_acc = evaluate(loaders['test'], model, criterion, device)
    print('=' * 92)
    print(f'| test loss {test_loss:5.3f} | test acc {test_acc:5.2f} | checkpoint {args.out} |')
    print('=' * 92)


if __name__ == '__main__':
    main()


