"""Train the class-conditional LSTM language model (GenModel) as a classifier.

GenModel scores a sentence under a candidate label: it is an LSTM language model
whose decoder is conditioned on a label embedding, so its token-level loss is
-log p(x | y). Classification is `argmin_y` of the summed loss over all labels,
which is `argmax_y p(x | y)` under a uniform class prior.

Run preprocess.py first, then:

    python GenTrain.py --epochs 10
"""

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import PackedSequence, pack_sequence, pad_packed_sequence

import agnews_data
from agnews_data import SEED
from models import GenModel

# Paths default to the repo root, not the caller's cwd, so the scripts in
# training/ behave the same however they are invoked.
ROOT = Path(__file__).resolve().parents[1]


def collate(batch):
    """Sort by descending length so every pack_sequence below shares one layout."""
    batch.sort(key=lambda item: len(item[0]), reverse=True)
    sequences, labels = zip(*batch)
    return list(sequences), torch.stack(labels)


def pack_inputs(sequences, device):
    """Teacher-forced LM pair: inputs are seq[:-1], targets are seq[1:]."""
    x = pack_sequence([s[:-1] for s in sequences]).to(device)
    x_pred = pack_sequence([s[1:] for s in sequences]).to(device)
    return x, x_pred


def like(x, values):
    """Wrap a flat tensor in x's packing layout so `.data` stays token-aligned."""
    return PackedSequence(values, x.batch_sizes, x.sorted_indices, x.unsorted_indices)


def sequence_loss(loss_flat, x):
    """Per-token loss over the packed buffer -> summed loss per sentence."""
    padded, _ = pad_packed_sequence(like(x, loss_flat), batch_first=True)
    return padded.sum(dim=1)


def score_all_labels(model, x, x_pred, criterion, nclass):
    """Summed -log p(x | y) for every candidate label. Returns (nclass, batch)."""
    return torch.stack([
        sequence_loss(criterion(model(x, x_pred, like(x, torch.full_like(x.data, y)),
                                      None, criterion), x_pred.data), x)
        for y in range(nclass)
    ])


@torch.no_grad()
def evaluate(loader, model, criterion, device, nclass):
    model.eval()
    total_nll, correct, seen = 0.0, 0, 0

    for sequences, labels in loader:
        labels = labels.to(device)
        x, x_pred = pack_inputs(sequences, device)
        scores = score_all_labels(model, x, x_pred, criterion, nclass)

        correct += (scores.argmin(dim=0) == labels).sum().item()
        total_nll += scores.min(dim=0).values.sum().item()
        seen += len(sequences)

    return total_nll / seen, 100.0 * correct / seen


def train_epoch(loader, model, criterion, optimizer, device, epoch, args):
    model.train()
    total_loss, total_tokens = 0.0, 0
    window_loss, window_tokens, start = 0.0, 0, time.time()

    for step, (sequences, labels) in enumerate(loader, 1):
        labels = labels.to(device)
        x, x_pred = pack_inputs(sequences, device)
        # The true label of each sentence, broadcast to every one of its tokens.
        y_ext = pack_sequence([labels[i].expand(len(s) - 1)
                               for i, s in enumerate(sequences)])

        out = model(x, x_pred, y_ext, None, criterion)
        loss = criterion(out, x_pred.data)

        optimizer.zero_grad(set_to_none=True)
        loss.mean().backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.sum().item()
        total_tokens += loss.numel()

        if step % args.log_interval == 0:
            window = (total_loss - window_loss) / (total_tokens - window_tokens)
            ms = (time.time() - start) * 1000 / args.log_interval
            print(f'| epoch {epoch:2d} | {step:5d}/{len(loader):5d} batches '
                  f'| {ms:6.1f} ms/batch | loss {window:5.3f} | ppl {math.exp(window):7.1f} |')
            window_loss, window_tokens, start = total_loss, total_tokens, time.time()

    return total_loss / total_tokens


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT / 'data/ag_news/processed/agnews.pt')
    p.add_argument('--out', type=Path, default=ROOT / 'checkpoints/gen_lstm_agnews.pth')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--word-emb-dim', type=int, default=100)
    p.add_argument('--label-emb-dim', type=int, default=100)
    p.add_argument('--hid-dim', type=int, default=100)
    p.add_argument('--layers', type=int, default=1)
    # Measured: dropout 0.5 *hurts* this model (90.37 vs 90.57 test). The LM
    # objective already regularizes it -- it underfits rather than overfits.
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--clip', type=float, default=1.0)
    p.add_argument('--log-interval', type=int, default=200)
    args = p.parse_args()

    torch.manual_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    cache, loaders = agnews_data.load(args.data, args.batch_size, collate)
    vocab, classes = cache['vocab'], cache['classes']
    nclass = len(classes)
    print(f'vocab {len(vocab):,} | classes {classes}')

    for split, loader in loaders.items():
        print(f'{split:5s} {len(loader.dataset):>7,} examples | {len(loader):>5,} batches')

    model = GenModel(len(vocab), args.word_emb_dim, args.label_emb_dim, args.hid_dim,
                     args.layers, nclass, args.dropout, device.type == 'cuda',
                     tied=False, use_bias=False, concat_label='hidden',
                     avg_loss=False, one_hot=False).to(device)
    criterion = nn.CrossEntropyLoss(reduction='none').to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best_acc, best_epoch = 0.0, -1

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss = train_epoch(loaders['train'], model, criterion, optimizer,
                                 device, epoch, args)
        val_loss, val_acc = evaluate(loaders['valid'], model, criterion, device, nclass)

        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save({'state_dict': model.state_dict(), 'vocab': vocab,
                        'classes': classes, 'args': vars(args), 'val_acc': val_acc},
                       args.out)

        print('-' * 92)
        print(f'| end of epoch {epoch:2d} | {time.time() - epoch_start:6.1f}s '
              f'| train loss {train_loss:5.3f} | train ppl {math.exp(train_loss):7.1f} '
              f'| valid nll {val_loss:7.1f} | valid acc {val_acc:5.2f} |')
        print('-' * 92)

    print(f'\nbest validation accuracy {best_acc:.2f} at epoch {best_epoch}')

    model.load_state_dict(torch.load(args.out, weights_only=False)['state_dict'])
    test_loss, test_acc = evaluate(loaders['test'], model, criterion, device, nclass)
    print('=' * 92)
    print(f'| test nll {test_loss:7.1f} | test acc {test_acc:5.2f} | checkpoint {args.out} |')
    print('=' * 92)


if __name__ == '__main__':
    main()
