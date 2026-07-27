# Generative vs. Discriminative LSTM Text Classifiers on AG News

A clean, runnable replication of the two RNN text classifiers compared in
Yogatama et al. (2017) and Ding & Gimpel (2019):

- a **discriminative** classifier — an LSTM encoder, mean-pooled hidden states, softmax over labels, trained to maximise `log p(y | x)`;
- a **generative** classifier — a *class-conditional LSTM language model* trained to maximise `log p(x | y)`, which classifies by scoring a document under every label and picking the best (Bayes' rule).

Both models share one preprocessing pipeline and one data loader, so the comparison
between them differs only in the model.

## Papers replicated

| | Paper | Model replicated here |
|---|---|---|
| **DeepMind** | [Generative and Discriminative Text Classification with Recurrent Neural Networks](https://arxiv.org/pdf/1703.01898) — Yogatama, Dyer, Ling & Blunsom, 2017 | §2.1 discriminative; §2.2 generative ("Shared LSTM") |
| **UChicago / TTIC** | [Latent-Variable Generative Models for Data-Efficient Text Classification](https://arxiv.org/pdf/1910.00382) — Ding & Gimpel, EMNLP 2019 | §2 baselines (their generative baseline *is* DeepMind's "Shared LSTM") |

Ding & Gimpel's §2 baselines are re-implementations of Yogatama et al.'s models;
their footnote 1 lists the only intended differences. This repo follows **Ding &
Gimpel's formulation and default hyperparameters**, and the section
[Where this repo differs from the papers](#where-this-repo-differs-from-the-papers)
documents every point of divergence.

Ding & Gimpel's *latent-variable* models (their §3, the actual contribution of that
paper) are **not** implemented here — only the two baselines they compare against.

## Results

AG News, full training set, single seed, best-on-dev checkpoint evaluated on test.
Reproduced by the two scripts below.

| Model | Params | Dev acc | **Test acc** | Best epoch | Time |
|---|---|---|---|---|---|
| Generative (`GenModel`) | 8,232,500 | 91.40 | **90.67** | 12 / 100 | 66 s/epoch, 110 min |
| Discriminative (`DiscModel`) | 2,798,304 | 92.60 | **91.99** | 91 / 100 | 18 s/epoch, 30 min |

Measured on an NVIDIA GB10. The discriminative model wins by ~1.3 points at full
data, which is the direction and roughly the magnitude reported in Ding & Gimpel's
Figure 2(c).

The two models differ far more in *how* they train than in where they end up. The
discriminative model overfits — without dropout it peaks at epoch 3 and then decays
steadily. The generative model never overfits: its dev accuracy plateaus near 91.1
by epoch ~20 and stays flat for the remaining 80 epochs. The next-token language
modelling objective supplies hundreds of bits of signal per document, versus ~2 bits
for a 4-way label, and that acts as a powerful built-in regulariser. This is the
mechanism behind the data-efficiency claims in both papers.

### Dropout ablation

| Model | Dropout | Epoch budget | Dev acc | Test acc |
|---|---|---|---|---|
| Discriminative | 0.0 | 10 | 91.84 (ep 3) | 91.07 |
| Discriminative | 0.3 | 15 | 92.10 (ep 4) | 91.79 |
| Discriminative | **0.5** | 15 | 92.29 (ep 10) | **92.24** |
| Generative | **0.0** | 10 | 91.11 (ep 9) | **90.57** |
| Generative | 0.5 | 10 | 90.79 (ep 6) | 90.37 |

Dropout is worth **+1.2 test points** to the discriminative model and **costs** the
generative model 0.2 — the generative model underfits rather than overfits, so
regularising it only slows it down. Hence the asymmetric defaults: `DiscTrain.py`
uses 0.5, `GenTrain.py` uses 0.0.

Single seed per row, so 0.3 vs 0.5 is not a reliable separation; 0.0 vs 0.5 is.

## Setup

Requires a CUDA GPU and `conda`. The environment lives inside the repo at `./nlp-env`
and is gitignored.

```bash
conda create -y -p ./nlp-env python=3.11
./nlp-env/bin/pip install --index-url https://download.pytorch.org/whl/cu130 torch
./nlp-env/bin/pip install spacy pandas "click<8.2"
./nlp-env/bin/python -m spacy download en_core_web_sm
```

Verified with torch 2.13.0+cu130, spaCy 3.8.14, pandas 3.0.5 on Python 3.11.

Two notes that will cost you time otherwise:

- **The `cu130` index is not optional on Blackwell GPUs.** Earlier CUDA builds ship
  an `nvrtc` that does not recognise `sm_121`; matmuls work (they are precompiled)
  but any JIT-compiled kernel dies with `invalid value for --gpu-architecture`.
  Substitute the index URL that matches your own GPU.
- **`click<8.2` is pinned deliberately.** spaCy's `typer` dependency does not pull
  `click` in on all platforms, and spaCy fails at import without it.

## Reproducing the results

```bash
./train_disc.sh     # discriminative classifier  (~30 min)
./train_gen.sh      # generative classifier      (~110 min)
```

That is the whole flow. On the first run each script downloads AG News and builds
the preprocessing cache automatically; afterwards it goes straight to training.
Training is seeded, so you should reproduce the numbers above exactly.

Any extra flag is passed through to the Python trainer and overrides the script's
defaults:

```bash
./train_disc.sh --dropout 0        # paper-faithful, no dropout
./train_gen.sh  --epochs 20        # the generative model converges by ~epoch 20
```

Both scripts write their console output to `train_{gen,disc}.log` and save the
best-on-dev checkpoint to `checkpoints/`.

### Running the stages by hand

```bash
./nlp-env/bin/python training/prepare_data.py   # download AG News + stratified dev split
./nlp-env/bin/python training/preprocess.py     # clean, tokenise, build vocab, cache
./nlp-env/bin/python training/DiscTrain.py --epochs 15 --dropout 0.5
./nlp-env/bin/python training/GenTrain.py  --epochs 20
```

`--help` on any of them lists the full flag set.

## Repository layout

```
├── train_disc.sh, train_gen.sh   entry points — run these
├── README.md
├── training/                     all training code
├── checkpoints/                  best-on-dev weights
└── data/ag_news/                 downloaded + preprocessed data (gitignored)
```

| File | Purpose |
|---|---|
| `train_disc.sh`, `train_gen.sh` | Best-known configurations, end to end |
| `training/prepare_data.py` | One-off: download AG News, carve a stratified 10k dev split off the 120k train set |
| `training/preprocess.py` | One-off: clean, tokenise with spaCy, build the vocab, cache token ids |
| `training/agnews_data.py` | Shared `Dataset` + loader construction used by both trainers |
| `training/models.py` | `DiscModel`, `GenModel` (plus an unused `MLPFeatureExtractor`) |
| `training/DiscTrain.py` | Discriminative training / evaluation |
| `training/GenTrain.py` | Generative training / evaluation |

Paths inside `training/` resolve against the repo root rather than the working
directory, so the scripts behave identically whether you invoke them via the shell
wrappers, from the root, or from inside `training/`.

Preprocessing is deliberately split out from training: spaCy tokenisation of 127k
documents is a one-off job, and caching it means both models provably consume
byte-identical inputs.

## Data

AG News, from the [original release](https://arxiv.org/abs/1509.01626) (Zhang et al.,
2015): 4 balanced classes — World, Sports, Business, Sci/Tech.

| Split | Documents | Source |
|---|---|---|
| train | 110,000 | official 120k train, minus the dev split |
| dev | 10,000 | stratified sample, 2,500/class, seed 2021 |
| test | 7,600 | official test set |

Preprocessing: title and description are concatenated; AG News's `\\` paragraph
markers, half-escaped HTML entities (`#39;`, `quot;`) and stray `<b>` markup are
cleaned; text is lowercased and tokenised with spaCy; documents are truncated to 80
tokens and wrapped in `<bos>`/`<eos>`. The vocabulary is built from the training
split only — 40k cap, min frequency 5, yielding **27,171 types** and a 1.4% OOV rate.
Mean document length is 46.4 tokens.

The official AG News release has no dev split, so `prepare_data.py` creates one with
a fixed seed. It is byte-reproducible: rerunning the script regenerates the identical
split.

## Models

Both use a one-layer unidirectional LSTM, 100-dimensional word embeddings, and a
100-dimensional hidden state, per Ding & Gimpel §4.2.

### Discriminative (`DiscModel`)

Encode the document, **average** the hidden states, apply a softmax over labels:

```
p(y | x) ∝ exp( (1/T · Σₜ hₜ)ᵀ v_y + b_y )
```

Both papers specify the mean rather than the final hidden state; Yogatama et al. note
the average worked better in their preliminary experiments and is cheaper than
attention for long documents.

Sequences are packed, so the LSTM never consumes padding, and the average divides by
true length. The result is padding-invariant: a document's logits do not depend on
what else is in its batch.

### Generative (`GenModel`)

A class-conditional LSTM language model. The label embedding `v_y` is concatenated to
the hidden state at **every** timestep before the vocabulary softmax:

```
p(xₜ | x_<t, y) ∝ exp( u_{xₜ}ᵀ [hₜ ; v_y] )
log p(x | y) = Σₜ log p(xₜ | x_<t, y)
ŷ = argmin_y  −log p(x | y)
```

This is Yogatama et al.'s **"Shared LSTM"**: one model whose behaviour is modulated by
the label embedding, sharing word embeddings, LSTM and softmax parameters across
classes — as opposed to their "Independent LSTMs" variant, which trains a separate LM
per class and is not implemented here.

Classification runs the scoring forward pass once per candidate label and takes the
`argmin` of the summed negative log-likelihood, so evaluation costs `|Y|`× a training
forward pass.

## Where this repo differs from the papers

### Generative model

| Item | Papers | Here |
|---|---|---|
| Class-specific softmax bias `b_{y,xₜ}` | In Yogatama et al.'s equation. Ding & Gimpel's released code implements it as `--use_bias`, **default off**, and their §2 equations omit it | **Not implemented** — matches Ding & Gimpel's default |
| Label prior `p(y)` | Both papers state `argmax_y p(x\|y)p(y)` with `p(y)` from MLE | **Uniform** prior, i.e. plain `argmin` of the LM loss |
| Peephole connections | Yogatama et al. §2.1 use them; Ding & Gimpel's footnote 1 treats peepholes as a *discriminative-model-only* difference | Plain `nn.LSTM` — follows Ding & Gimpel |
| "Independent LSTMs" variant | Evaluated by Yogatama et al. | Not implemented |

On the missing bias: it costs nothing here. With `concat_label='hidden'` the label
embedding reaches the logits only through `U_label · v_y`, and since the label
dimension (100) exceeds the number of classes (4), those vectors can already realise
*any* per-class bias table exactly. The term is expressively redundant at this scale.

On the uniform prior: AG News is exactly balanced, and our dev split preserves that
(27,500 per class in train), so `log p(y)` is a constant that cancels in the `argmin`.
Predictions are identical. This would need fixing for an imbalanced corpus.

### Discriminative model

| Item | Papers | Here |
|---|---|---|
| Dropout | Reference implementation defaults to 0; neither paper's training details mention it | **0.5 by default** (worth +1.2 test points). `--dropout 0` reproduces the paper-faithful run |
| Peephole connections | Yogatama et al. §2.1 use them | Plain `nn.LSTM` — matches Ding & Gimpel, who state this as their one architectural departure |

### Training setup

| Item | Ding & Gimpel | Here | Effect |
|---|---|---|---|
| Batch size | 32 | 64 | Throughput |
| Gradient clipping | 0.25 | 1.0 | — |
| Weight decay | 1.2e-6 | 1e-5 | — |
| Generative backward loss | `sum` over tokens | `mean` over tokens | Decouples gradient scale from batch token count, so clipping behaves consistently |
| Datasets | 6 (Zhang et al., 2015) | AG News only | — |
| Training-set sizes | Swept 5 → 10k per class | Full data only | **The papers' central claim is about small-data regimes; this repo does not test it** |
| Dev split | Their own | 10k stratified from the 120k train set | Numbers not directly comparable to the papers' |

Matching both papers: 100-dim word embeddings, 100-dim label embeddings, 100-dim
hidden, 1 layer, unidirectional, Adam at lr 1e-3, 80-token truncation before adding
`<bos>`/`<eos>`, and early stopping by keeping the best-on-dev checkpoint.

### The most important caveat

Both papers argue that generative classifiers win in the **small-data** regime, and
that the advantage shrinks as training data grows. This repo only measures the
full-data endpoint, where both papers already report the discriminative model ahead.
**These results do not test either paper's actual claim.** Doing so means sweeping
training-set size (5 / 20 / 100 / 1k / 2k / 5k / 10k per class), which the shared
cache in `agnews_data.py` makes straightforward to add.

## Implementation notes

A few things worth knowing if you extend this code.

**`nn.LSTM(dropout=...)` is a silent no-op at `num_layers=1`.** PyTorch only inserts
dropout *between* stacked layers, so with a single layer the argument does nothing at
all (it warns, but the warning is easy to miss). `DiscModel` therefore applies
`nn.Dropout` explicitly to the embeddings and to the pooled document vector, and
passes `0.0` to the LSTM constructor when `n_layers == 1`.

**`GenModel.forward` operates on the flat `PackedSequence.data` buffer.** Any tensor
you hand it — label streams, per-token losses — must be wrapped in the *same* packing
layout, or token-to-label alignment breaks silently instead of raising. `GenTrain.py`
routes all of this through one `like()` helper, and its `collate` sorts by descending
length so every pack agrees.

**Selecting the best of 100 dev evaluations mildly overfits the dev set.** The
discriminative model's 100-epoch run reaches a *higher* dev score (92.60) but a
*lower* test score (91.99) than the 15-epoch run in the ablation table (92.29 dev /
92.24 test). If you care about the cleanest test estimate, a shorter budget is
actually better here.

## References

```bibtex
@article{yogatama2017generative,
  title   = {Generative and Discriminative Text Classification
             with Recurrent Neural Networks},
  author  = {Yogatama, Dani and Dyer, Chris and Ling, Wang and Blunsom, Phil},
  journal = {arXiv preprint arXiv:1703.01898},
  year    = {2017}
}

@inproceedings{ding2019latent,
  title     = {Latent-Variable Generative Models for Data-Efficient
               Text Classification},
  author    = {Ding, Xiaoan and Gimpel, Kevin},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in
               Natural Language Processing (EMNLP)},
  pages     = {507--517},
  year      = {2019},
  url       = {https://aclanthology.org/D19-1048/}
}

@inproceedings{zhang2015character,
  title     = {Character-level Convolutional Networks for Text Classification},
  author    = {Zhang, Xiang and Zhao, Junbo and LeCun, Yann},
  booktitle = {Advances in Neural Information Processing Systems (NIPS)},
  year      = {2015}
}
```

Ding & Gimpel's reference implementation: <https://github.com/AnnDing/Generative_classifier>
