# Quantization of the AG News Classifiers — Full Technical Record

Static post-training quantization (PTQ) of both trained classifiers — the
discriminative LSTM (`DiscModel`) and the generative class-conditional LSTM LM
(`GenModel`) — using **Brevitas 0.13.0**, with weights *and* activations
quantized, calibration on the training set, and GPFQ/Qronos weight-error
correction on top. This document records the whole process: decisions, numbers,
failed paths, bugs, and diagnoses.

Machine-readable results: [`results/ptq_results.csv`](results/ptq_results.csv).
Per-run logs: `logs/ptq_*.log`. Reproduction: [§11](#11-reproduction).

---

## TL;DR

| | Generative (fp32 90.67) | Discriminative (fp32 91.99) |
|---|---|---|
| int8 | 90.72 / **90.76** with GPFQ | 91.91 / 91.92 with GPFQ |
| int6 | 90.42 / 90.64 with GPFQ | 91.49 / 91.50 with GPFQ |
| int4 | 61.45 / **87.09** with GPFQ | 65.70 / 73.16 with GPFQ / **76.22** with Qronos |
| int3 | 23.08 / **89.01** with GPFQ | 36.36 / 27.22 with GPFQ (GPFQ *hurts*) |
| int2 | 25.00 (chance) | 25.00 (chance) |

- **8 and 6 bits are effectively lossless** for both models with plain PTQ.
- **Plain PTQ collapses at 4 bits**; the cause (verified by stage isolation) is
  the **LSTM**, not the 27k-row embedding, despite the embedding being 97% of the
  discriminative model's weights.
- **GPFQ rescues the generative model** (its 5.4M-weight decoder is
  GPxQ-eligible): −3.6 points at 4 bits, −1.7 at 3 bits.
- **GPFQ can hurt**: at disc int3 it fits the 400 classifier weights against an
  LSTM output that is pure noise, landing below plain rounding.
- **Mixed precision is the real compression play** for the discriminative model:
  LSTM@8 + everything-else@3 recovers 84.87 vs 36.54, for ~5% extra footprint.

---

## 1. Starting point

Trained float models (single seed 2021, best-on-dev checkpoints from the
100-epoch runs):

| Model | Test acc | Dev acc (best epoch) | Params | fp32 weights |
|---|---|---|---|---|
| `GenModel` | 90.67 | 91.40 (ep 12/100) | 8,232,500 | 31.40 MB |
| `DiscModel` | 91.99 | 92.60 (ep 91/100) | 2,798,304 | 10.67 MB |

Parameter composition — this drives everything downstream:

| Component | Disc | Gen |
|---|---|---|
| Embedding (27,171 × 100) | 2,717,100 (**97.1%**) | 2,717,100 (33.0%) |
| LSTM (1 layer, uni, 100 hid) | 80,800 (2.9%) | 80,800 (1.0%) |
| Output layer | fc 100→4: 404 (0.01%) | decoder 200→27,171: 5,434,200 (**66.0%**) |
| Label embedding (4 × 100) | — | 400 |

Hardware: NVIDIA GB10 (Grace-Blackwell, sm_121, **aarch64**), driver CUDA 13.0.
Environment: `./nlp-env` conda prefix, Python 3.11.

---

## 2. Framework selection (what was probed and why it failed or won)

Before Brevitas was chosen, PyTorch-native quantization was probed on this
machine. Findings, each verified by running code, not read from docs:

| Probe | Result |
|---|---|
| `fbgemm` / `x86` backends | `RuntimeError: unknown architecure` (sic) — x86-only kernels, machine is aarch64 |
| `qnnpack` backend | Works — the only usable native backend here |
| Quantized ops on CUDA | Not implemented — native PTQ is **CPU-only** |
| `aten::sum` on quantized tensors | `NotImplementedError` for `QuantizedCPU` → mean-pooling must dequantize first |
| `torch.ao.nn.quantizable.LSTM` | Exists; `from_float` needs an explicit `qconfig=` kwarg (its error message is misleading) |
| Working native toy pipeline | quant→LSTM→dequant→float pool→quant→fc: max output diff 0.0018 vs float |
| `torch.ao.quantization.quantize_pt2e` | Absent from this torch build |
| `torchao` | Not installed |
| FX graph mode | Untraceable here — `PackedSequence` + data-dependent control flow |

**Why Brevitas won** (user-directed, but also the right call): its quantization
is *simulated* (float tensors snapped to the integer grid), so it **runs on
GPU** — critical because the generative model's evaluation is 4 label passes ×
7,600 documents with a 27k softmax per token; supports **arbitrary bit widths**
(the whole ablation axis); and `QuantLSTM` quantizes the LSTM *internals* (gate
accumulators, sigmoid/tanh outputs, cell state), which native PTQ does not
expose. The trade-off: no real integer kernels, so speed/size wins are simulated
(a real deployment needs an export step, e.g. ONNX QCDQ).

### Environment side effect

`pip install brevitas` **downgraded torch 2.13.0+cu130 → 2.12.1+cu130** (brevitas
pins `torch<2.13`). CUDA and the nvrtc JIT path (the GB10's known failure mode,
see the cu130 requirement) were re-verified working, and the fp32 test baseline
was re-evaluated: **91.99, bit-identical to before the downgrade**. Rule: any
env mutation → re-run the float baseline before trusting new numbers.

---

## 3. Building the quantized models (`quantization/quant_models.py`)

### 3.1 Weight transfer: torch LSTM → Brevitas QuantLSTM

The layouts differ. torch packs all four gates in one matrix per direction with
**two** bias vectors summed at runtime; Brevitas keeps a separate module per
gate with **one** bias:

```
torch:    weight_ih_l0 (400×100)  = rows [i | f | g | o]   + bias_ih_l0, bias_hh_l0
brevitas: layers.0.0.{input,forget,cell,output}_gate_params.{input_weight,hidden_weight}.weight
          layers.0.0.<gate>_gate_params.bias  = bias_ih[rows] + bias_hh[rows]
```

Gate order is `[input, forget, cell, output]` (torch's `W_ii|W_if|W_ig|W_io`).
The remap lives in `_remap_lstm()`; `GenModel` additionally renames `rnn.` →
`lstm.`. **Validated numerically**: float QuantLSTM (all quantizers `None`)
loaded with remapped weights matches `nn.LSTM` to **8.9e-08** (fp32 rounding).
The loaders (`load_float_weights`, `load_gen_float_weights`) also verify that
every unmatched state-dict key is quantizer state — a leftover real parameter
raises.

The quantized graphs total 2,797,917 (disc) / 8,232,113 (gen) params — slightly
below the float models because the per-gate bias merge removes 400 duplicated
bias entries.

### 3.2 PackedSequence → pad + mask

`QuantLSTM` rejects `PackedSequence` outright
(`RuntimeError: PackedSequence input currently not supported.`). Both float
models used packing, so the quantized models pad and mask instead.

**Proof of equivalence** (this had to be proven, not assumed): for a
*unidirectional* LSTM, trailing padding cannot influence hidden states at real
positions, and both models discard pad positions (masked mean / masked NLL).
Measured: packed vs padded+masked on real data —

- float64, CPU: max diff **1.1e-16** (exact; machine epsilon)
- float32, GPU: ~6.4e-4 (cuDNN vs generic kernel accumulation order — not logic)
- nn.LSTM vs Brevitas-float LSTM, same padded input: 1.6e-4 (same reason)

### 3.3 Quantizer placement

**`QuantDiscModel`** (mirrors `DiscModel`):

```
QuantEmbedding ──> QuantIdentity ──> QuantLSTM ──> [float masked mean-pool] ──> QuantIdentity ──> QuantLinear(bias)
 (weights intN)     (acts intN)       (all intN)                                  (acts intN)      (weights intN, Int32Bias)
```

- `QuantEmbedding` has **no output activation quantizer** of its own → explicit
  `QuantIdentity` feeds the LSTM a quantized activation.
- The mean-pool runs in float (native lesson §2 carries over conceptually), then
  `QuantIdentity(return_quant_tensor=True)` supplies the input *scale* that
  `Int32Bias` needs to derive the bias scale.
- Default quantizers: `Int8WeightPerTensorFloat` (weights — **per-tensor**, one
  scale per layer), `Int8ActPerTensorFloat` (activations), `Int32Bias`.
  `QuantLSTM` internally also quantizes gate accumulators
  (`gate_acc_bit_width`), sigmoid outputs (`Uint8`), tanh outputs, and the cell
  state — all set to the same bit width in these experiments.
- Diagnostic extension: `emb_bit_width` / `lstm_bit_width` / `fc_bit_width`
  overrides (default to `bit_width`) for stage isolation (§9).

**`QuantGenModel`** (mirrors `GenModel`):

```
encode:  QuantEmbedding ──> QuantIdentity ──> QuantLSTM ──> h_t          (runs ONCE per batch)
decode:  cat[h_t ; v_y] ──> QuantIdentity ──> QuantLinear(200→27171)     (runs once per LABEL)
```

Two deliberate choices:

1. **encode/decode split.** Only the decoder depends on the candidate label, so
   the (expensive) QuantLSTM runs once per batch, the decoder 4×. QuantLSTM is
   **~370× slower than cuDNN** (223.1 vs 0.6 ms/batch at B=64, T=80, H=100), so
   fusing would quadruple the dominant cost for nothing.
2. **Decoder *output* left unquantized.** The logits feed the 27k-way
   log-softmax whose per-token NLLs, summed over ~46 tokens and compared across
   4 labels, *are* the classification decision. Snapping them to an int grid
   would inject error into exactly the compared quantity. (The decoder's
   weights and its input are still quantized.)
3. Known risk, flagged before running: `[h_t ; v_y]` is requantized to a
   **single shared scale**. The label embedding (4 rows) may occupy a very
   different range than the LSTM states; at very low bit widths this could
   crush the label signal. Splitting the scale is the untried fix.

### 3.4 The three-way report

Every PTQ run evaluates:

1. **float32** — the trained model exactly as in training (packed pipeline)
2. **float reference** — same weights inside the Brevitas graph, every quantizer
   `None` (pads + masks) — isolates the *port* from the *quantization*
3. **quantized** — the same graph, quantizers on

Float reference matched float32 **exactly** for both models (disc 91.99,
gen 90.67), so every reported delta is genuinely quantization.

---

## 4. Calibration protocol

- **Source:** training split only (110,000 docs). Test data is never seen before
  the final evaluation; dev is used only for reporting.
- **Sampling:** whatever the *shuffled* training loader yields first — random,
  **unstratified**. Measured composition at 4,096 docs: World 987 / Sports 1034 /
  Business 1011 / Sci-Tech 1064 (±4% of uniform; AG News train is exactly
  balanced so luck is on our side).
- **Reproducible:** `torch.manual_seed(2021)` precedes loader construction; two
  independent runs draw the identical document sequence. All cells of the
  ablation therefore calibrate on the same data — bit width is the only variable.
- **Budget:** initially 64 batches (4,096 docs); later standardized to **128
  batches = 8,192 docs** (the current wrapper default). All rows in the final
  CSV are at 8,192 except the two `int2` rows (4,096; both at chance anyway).
- **Mechanics:** `brevitas.graph.calibrate.calibration_mode` — disables
  quantization while observers collect true running averages; on exit,
  activation scales/zero-points are frozen. Activation calibration cost: ~10 s
  (disc, 64 batches) / ~20 s (disc, 128).
- Every run prints its sample's actual class composition into the log, so the
  provenance is on the record per run.

**Bias correction is NOT used** (see troubleshooting T3) — and at 8 bits there
was nothing left for it to recover.

---

## 5. Plain PTQ results

All numbers: test accuracy, calibration 8,192 docs (int2: 4,096), from
`results/ptq_results.csv`.

| Bits | Gen acc | Gen Δ | Gen NLL | Disc acc | Disc Δ | Disc CE | Compression |
|---|---|---|---|---|---|---|---|
| fp32 | 90.67 | — | 209.2 | 91.99 | — | 0.4090 | 1× |
| int8 | 90.72 | +0.05 | 209.7 | 91.91 | −0.08 | 0.4198 | 4.00× |
| int6 | 90.42 | −0.25 | 218.2 | 91.49 | −0.50 | 0.4482 | 5.33× |
| int4 | 61.45 | −29.22 | 435.1 | 65.70 | −26.29 | 2.0038 | ~8× |
| int3 | 23.08 | −67.59 | 456.1 | 36.36 | −55.63 | 1.5358 | ~10.7× |
| int2 | 25.00 | −65.67 | 460.7 | 25.00 | −66.99 | ~16× |

Simulated footprints: gen 31.40 → 7.85 / 5.89 / 3.93 / 2.95 / 1.96 MB;
disc 10.67 → 2.67 / 2.00 / 1.34 / 1.00 / 0.67 MB. ("Simulated" because Brevitas
stores float tensors snapped to the grid — the .pth files stay full size; these
are what an int export would occupy.)

Observations:

- A **cliff between 6 and 4 bits** for both models — initially read as an
  architectural property, later revised (§8: it is what round-to-nearest does,
  and much of it is recoverable; §9: the unrecoverable part lives in the LSTM).
- Disc int2 = exactly 25.00 with all predictions on one class — collapse, not
  degradation.
- Gen int3 = **23.08, below the 25% chance floor** — the label ranking is
  systematically *inverted*, not merely noisy. Consistent with the shared-scale
  concern in §3.3.
- Contrary to the a-priori prediction that the generative model (argmin over
  four sums of ~46 token NLLs, small margins) would be more fragile: at 8/6 bits
  it degrades *less* than the discriminative model. Accumulated token-level
  error averages out rather than compounding.

---

## 6. GPFQ / GPTQ / Qronos

### 6.1 What they are

All live in `brevitas.graph.*` and share the `GPxQ` base class
(`gpxq.py`), applied through the `gpfq_mode` context manager:

- **GPTQ** (`gptq.py`) — column-by-column weight quantization with an
  approximate inverse-Hessian of the input covariance; compensates remaining
  columns. Classically assumes float input.
- **GPFQ** (`gpfq.py`) — greedy path following; minimises
  ‖W_float·x_float − W_quant·**x_quant**‖, i.e. fits against the *actually
  quantized* input — the right objective when activations are quantized too
  (our setting).
- **Qronos** (`qronos.py`, AMD 2025, subclasses GPFQ) — interleaves explicit
  correction of weight error and input/activation error. Selected via
  `gpfq_mode(model, algorithm_impl=Qronos)`.

### 6.2 Eligibility — the decisive constraint

GPxQ handles **only `nn.Linear` + Conv1/2/3d** (ConvTranspose excluded). Not
LSTM cells (Brevitas `GateWeight` modules), not embeddings. Measured coverage:

| Model | Eligible module | Coverage |
|---|---|---|
| Generative | `decoder` (200→27,171) | **5,434,200 / 8,232,113 = 66.0%** |
| Discriminative | `fc` (100→4) | **400 / 2,797,917 = 0.014%** |

The initial prediction — "GPFQ will be near-useless for the discriminative
model" — was **wrong** (see §7). Coverage bounds what GPFQ can *rewrite*, not
what it can *compensate for*: because it fits against the quantized input, the
final layer can absorb error propagated from upstream stages it cannot touch.

### 6.3 Implementation notes (hard-won)

- `gpfq_mode` **replaces the model's `forward`** (`catch_stopfwd`): each
  calibration batch is run **twice** — once with quantization live (captures
  the layer's quantized input) and once with everything disabled (captures the
  float reference). Consequence: you must drive `model(...)` itself. Calling
  sub-methods (`encode()` / `decode()`) **silently bypasses GPFQ entirely**.
- Loop shape (per the Brevitas docstring):
  `with gpfq_mode(model, use_quant_activations=True) as gpfq:` →
  `for _ in range(gpfq.num_layers): for batch: gpfq.model(batch); gpfq.update()`.
- For the generative model, GPFQ is fed **all four candidate labels** per batch:
  at inference the decoder sees `[h_t ; v_y]` for every `y`, and that — not the
  training distribution (true labels only) — is the input distribution the
  reconstruction error should be minimised over.
- Cost is modest: gen 16 batches × 4 labels ≈ 29 s; disc is seconds.
- `--gpxq-batches 32` (2,048 docs) for all reported runs.

---

## 7. Results with GPFQ / Qronos

Test accuracy, all at 8,192 calibration docs, GPxQ on 32 batches:

**Generative** (fp32 90.67):

| Bits | Plain | +GPFQ | +Qronos | NLL plain → GPFQ |
|---|---|---|---|---|
| int8 | 90.72 | **90.76** | — | 209.7 → 209.8 |
| int6 | 90.42 | **90.64** | — | 218.2 → 218.1 |
| int4 | 61.45 | **87.09** | 26.05 | 435.1 → 368.1 |
| int3 | 23.08 | **89.01** | 31.38 | 456.1 → 436.1 |

**Discriminative** (fp32 91.99):

| Bits | Plain | +GPFQ | +Qronos |
|---|---|---|---|
| int8 | 91.91 | **91.92** | — |
| int6 | 91.49 | **91.50** | — |
| int4 | 65.70 | 73.16 | **76.22** |
| int3 | **36.36** | 27.22 | 25.43 |

Findings:

1. **≥6 bits: nothing to fix.** Plain PTQ is already ~lossless; GPFQ moves
   hundredths.
2. **Gen int4/int3: GPFQ is transformative** (+25.6 / +65.9). The low-bit
   collapse was mostly decoder weight error, which is exactly what GPFQ
   corrects — and 66% of the model is eligible.
3. **Gen int3+GPFQ (89.01) > gen int4+GPFQ (87.09).** Verified robust, not an
   artefact: int4 stable at 87.33/87.09/87.12 for gpxq-batches 16/32/64; int3 at
   89.01 for 16 and 32. The NLLs explain the direction of the paradox: int3 is
   the *worse language model* (NLL 436.1 vs 368.1) but the *better classifier* —
   classification depends only on the **ranking** of the four summed NLLs, not
   their absolute quality. (Why int3's ranking survives better remains
   unexplained — hypothesis, untested: coarser grids may push GPFQ toward
   solutions dominated by the label-conditional component.)
4. **Disc int4: both methods help** (+7.5 GPFQ, +10.5 Qronos) despite 0.014%
   coverage — the 400-weight decision layer compensates upstream error (§6.2).
5. **Disc int3: both methods *hurt*** (36.36 → 27.22 / 25.43). At this point the
   LSTM output is noise (§9); fitting the classifier against noise on 2,048
   calibration docs is fitting noise, and lands below plain rounding.
6. **Qronos does not generalise at default settings**: best-in-class for disc
   int4 (76.22), catastrophic for gen (26.05 at int4 — *worse than plain*).
   Recorded as "misapplied at defaults", not as a method verdict; its
   hyperparameters were not explored.

---

## 8. Prediction-vs-outcome ledger

Kept deliberately, because several a-priori arguments failed:

| Prediction | Outcome |
|---|---|
| Generative model more fragile under quantization (accumulating error, small margins) | **Wrong at 8/6 bits** (degrades less than disc); right-ish only below 6 |
| GPFQ near-useless for disc (0.014% coverage) | **Wrong** at int4 (+7.5); right at int3 for the wrong reason (it's *harmful*, not useless) |
| 6→4 bit cliff is an architectural property | **Half-wrong** — largely a round-to-nearest artefact; GPFQ recovers most of it where coverage exists |
| Embedding (97% of disc weights) would be the low-bit bottleneck | **Wrong** — it's the LSTM (§9) |
| Shared `[h_t ; v_y]` scale is a low-bit risk for gen | Consistent with the below-chance int3 result, but never isolated — still a hypothesis |

---

## 9. Root-cause diagnosis: stage isolation (`quantization/diagnose_lowbit.py`)

Question: disc int3+GPFQ = 27.22 while int4 ≈ 73–76 — why the huge drop?
Method: hold one stage at 8 bits while the others run at 3, measure test
accuracy and the **prediction histogram** (uniform test set → 25.00 is chance;
a skewed histogram = collapse onto a class, not graceful degradation).

Plain PTQ, low=3, high=8, calibration 128 batches:

| Configuration | Acc | Prediction histogram (World/Sports/Business/SciTech) |
|---|---|---|
| all 8-bit | 91.89 | 1894 / 1909 / 1903 / 1894 |
| all 3-bit | 36.54 | **5812** / 30 / 223 / 1535 |
| embedding@8, lstm+fc@3 | 53.42 | 3461 / 454 / 1988 / 1697 |
| **lstm@8, embedding+fc@3** | **84.87** | 2189 / 1813 / 1906 / 1692 |
| fc@8, embedding+lstm@3 | 37.54 | 4883 / 16 / 182 / 2519 |

Same grid with GPFQ:

| Configuration | Acc | Histogram |
|---|---|---|
| all 8-bit | 91.87 | ~uniform |
| all 3-bit | 26.26 | 585 / 14 / 293 / **6708** |
| embedding@8 | 30.96 | 887 / 43 / 142 / 6528 |
| **lstm@8** | **85.25** | 2186 / 1804 / 1727 / 1883 |
| fc@8 | 26.70 | 354 / 8 / 26 / 7212 |

Conclusions:

1. **The LSTM is the bottleneck** — protecting it alone (2.9% of weights)
   recovers 84.87; protecting the embedding (97.1% of weights) recovers only
   53.42. **Parameter count does not predict quantization sensitivity.**
2. **Mechanism:** recurrence re-quantizes per timestep. At 3 bits:
   `sigmoid_quant` = 8 levels on [0,1], `tanh_quant` = 8 levels on [−1,1],
   `cell_state_quant` re-snaps the accumulated cell state to 8 values at
   *every* of ~46–80 steps. A memory that must persist across 46 steps cannot
   survive 46 roundings to 8 levels. Feed-forward layers quantize once;
   recurrent state quantizes T times.
3. **GPFQ's sign flips with LSTM health**: hurts in every LSTM@3 configuration
   (36.54→26.26, 53.42→30.96, 37.54→26.70), helps in the LSTM@8 one
   (84.87→85.25). GPFQ's objective is only meaningful when its input still
   carries signal.
4. **Why the generative model survives 3 bits (89.01 with GPFQ)** — hypothesis,
   *not* verified: its label embedding `v_y` enters at the decoder, *bypassing*
   the LSTM. With `h_t` degraded, GPFQ can still tune the decoder to produce
   label-discriminative token distributions from `v_y` alone — degenerating
   toward a class-conditional unigram LM, which is already a strong AG News
   classifier. The discriminative model has no path around its LSTM. Testable
   by running the same stage isolation on `QuantGenModel`.
5. **Practical:** mixed precision LSTM@8 + rest@3 ≈ 1.05 MB vs 1.00 MB all-3
   (disc), i.e. **+49 accuracy points for ~5% footprint**. At aggressive
   compression, mixed precision is the whole game, not a refinement.

---

## 10. Troubleshooting log (chronological)

**T1 — PyTorch-native backend on aarch64.** `fbgemm`/`x86` →
`RuntimeError: unknown architecure` from `quantized::linear_prepack`. Fix:
`qnnpack`. Follow-on: `aten::sum.IntList_out` not implemented for QuantizedCPU
→ dequantize before mean-pool. Native path proven feasible (toy diff 0.0018)
but CPU-only → dropped in favour of Brevitas.

**T2 — Brevitas install downgraded torch** 2.13.0→2.12.1 (pin `torch<2.13`).
fp32 baseline re-verified bit-identical (91.99). No CUDA/nvrtc regression.

**T3 — `bias_correction_mode` × `QuantLSTM` (Brevitas 0.13.0): two-stage
failure.** (a) `AttributeError: 'FusedActivationQuantProxy' object has no
attribute 'is_quant_enabled'` — the module scan trips over `_fast_cell`, a
JIT-scripted cell that `QuantLSTM` builds *lazily on first forward*
(`quant_rnn.py` `fast_cell` property); a fresh model scans clean, a calibrated
one doesn't. (b) Clearing `_fast_cell` (it rebuilds next forward) gets past the
scan, but then `RuntimeError: Input scale required` — bias correction disables
output quantizers, starving the LSTM's `Int32Bias` of its input scale. Control
experiment: the same graph **minus** QuantLSTM bias-corrects fine → QuantLSTM
is definitively the blocker. Resolution: `--bias-correction` is opt-in and
fails with a loud `SKIPPED` message; headline results don't use it (at ≥6 bits
there is nothing for it to recover anyway).

**T4 — the mislabelled-runs bug (process failure, cost a redo).** `apply_gpxq`
was added to `ptq_disc.py` but the *call* was only wired into `ptq_gen.py`.
Result: four discriminative "GPFQ" runs that were plain PTQ with convincing
`_gpfq` filenames and logs. Detected by auditing logs for the algorithm's own
progress line (`grep -c '^\s+(gpfq|qronos):'` → 0) and the result tag (`int4
test` instead of `int4+gpfq`); a calibration-budget mismatch (4,096 vs 8,192)
between "plain" and "gpfq" runs was the second tell. Fixed, and the whole disc
grid re-run at a uniform 8,192-doc budget. **Lesson: verify a labelled run by
its execution evidence, not its filename.**

**T5 — gen int3 > int4 anomaly.** Robustness-checked across GPFQ budgets
(16/32/64 batches) before accepting: int4 87.33/87.09/87.12, int3 89.01/89.01.
Real. Explanation in §7.3.

**T6 — Qronos catastrophic on gen.** 26.05 at int4 — *below* plain PTQ.
Filed as misapplication-at-defaults, not a method verdict; the disc int4 result
(76.22, best method there) shows the integration itself works.

**T7 — results-CSV precision wart (open, minor).** Runs self-record rows with
full precision, but `run_ptq.sh` ends by invoking the log-parsing collector,
whose reconstructed rows *overwrite* live ones. The gen log prints NLL as
`%7.1f`, so gen `test_metric` in the CSV has 1 decimal where disc has 4.
Fix if wanted: make the collector fill only missing keys (preferred), or widen
the log precision.

**T8 — data hygiene notes.** (a) The two `int2` rows predate the calibration
budget change (4,096 docs vs 8,192 elsewhere); both models sit at exact chance
there, so it does not affect conclusions — re-run `./run_ptq.sh --bits 2` for a
fully uniform grid. (b) During the training phase, `logs/train_disc.log` was
truncated by an interrupted re-run; the fp32 baseline is nonetheless
corroborated independently (checkpoint metadata `val_acc: 92.60` + re-evaluated
test 91.99, reproduced in every PTQ log's line 1).

---

## 11. Reproduction

```bash
# one cell
./quantize_disc.sh --bit-width 4                 # plain PTQ
./quantize_gen.sh  --bit-width 4 --gpxq gpfq     # PTQ + GPFQ

# grids (each cell writes its own CSV row, keyed on model×bits×gpxq;
# re-running a cell REPLACES its row — safe to run piecemeal, any order)
./run_ptq.sh                                     # plain, both models, bits 8 6 4 3 2
./run_ptq.sh --gpxq "none gpfq"                  # plain + GPFQ in one pass
./run_ptq.sh --model gen --bits "4 3" --gpxq qronos
./run_ptq.sh --collect-only                      # rebuild CSV from logs, run nothing

# stage-isolation diagnostic (§9)
./nlp-env/bin/python quantization/diagnose_lowbit.py --low 3 --gpxq none
./nlp-env/bin/python quantization/diagnose_lowbit.py --low 3 --gpxq gpfq
```

Everything is seeded (2021): same calibration documents, same results, every
run. Naming scheme: `checkpoints/{disc,gen}_lstm_agnews_int<N>[_gpfq|_qronos].pth`,
`logs/ptq_{disc,gen}_int<N>[_gpfq|_qronos].log`.

Defaults that matter: batch 64, `--calib-batches 128` (8,192 docs, set in the
wrappers), `--gpxq-batches 32`, per-tensor weight scales, decoder/fc *output*
logits unquantized, bias correction off.

## 12. File map

| File | Role |
|---|---|
| `quantization/quant_models.py` | `QuantDiscModel` (+ per-stage bit overrides), `QuantGenModel` (encode/decode split), gate remap + verified weight loaders |
| `quantization/ptq_disc.py` / `ptq_gen.py` | Calibrate → (optional GPFQ/Qronos) → three-way evaluation → self-record CSV row |
| `quantization/results.py` | CSV schema; `record()` replaces rows keyed on (model, bits, gpxq) |
| `quantization/collect_results.py` | Rebuild CSV from `logs/ptq_*.log` (both old and new log tag formats); back-fill/recovery |
| `quantization/diagnose_lowbit.py` | Stage-isolation diagnostic with prediction histograms |
| `quantize_disc.sh` / `quantize_gen.sh` | Single-cell wrappers; bit-width/algorithm-derived names |
| `run_ptq.sh` | Grid driver (`--model`, `--bits`, `--gpxq`, `--collect-only`, passthrough) |
| `results/ptq_results.csv` | The results table (tracked in git; logs are not) |

## 13. Open threads

- Split the shared `[h_t ; v_y]` quantization scale in `QuantGenModel` — the
  prime suspect for the gen int3 below-chance inversion under plain PTQ.
- Stage isolation on the generative model — would test the "v_y bypasses the
  LSTM" survival hypothesis (§9.4).
- Mixed-precision run as a headline configuration (disc LSTM@8, rest@3/4).
- Qronos hyperparameters on the generative model.
- Per-channel weight quantization (`Int8WeightPerChannelFloat`) — everything
  here is per-tensor; the 27k-row decoder and embedding are the obvious
  beneficiaries at 4 bits.
- Real integer export (ONNX QCDQ) to turn simulated footprints into real ones.
- All results are single-seed; the fine distinctions (e.g. 73.16 vs 76.22)
  deserve seed replication before being leaned on.
