"""Brevitas quantized counterparts of the trained float models.

Weights *and* activations are quantized, so these are meant to be calibrated with
`brevitas.graph.calibrate.calibration_mode` before use -- the activation
quantizers have no scale until they have seen data.

Brevitas quantization is "fake": tensors stay float32 with their values snapped to
the quantization grid. That means these models still run on the GPU, unlike
PyTorch's native quantized backends which are CPU-only.
"""

import brevitas.nn as qnn
import torch
import torch.nn as nn
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat, Int32Bias

# torch packs the LSTM gates into one matrix as W_ii | W_if | W_ig | W_io.
# Brevitas splits them into four separately named submodules, in this order.
LSTM_GATES = ('input', 'forget', 'cell', 'output')


class QuantDiscModel(nn.Module):
    """`DiscModel` with a quantized embedding, LSTM and classifier.

    One structural difference from `DiscModel`: Brevitas' QuantLSTM rejects
    `PackedSequence` outright, so the batch stays padded and the mean-pool is
    masked instead. For a *unidirectional* LSTM the two are numerically
    equivalent -- trailing padding cannot reach hidden states at earlier
    positions, and the mask keeps pad positions out of the average. Pass
    `quant=False` to build the same graph with every quantizer disabled, which
    isolates that claim from any quantization error.
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 n_layers=1, bit_width=8, quant=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        if quant:
            emb_kw = dict(weight_quant=Int8WeightPerTensorFloat, weight_bit_width=bit_width)
            act_kw = dict(act_quant=Int8ActPerTensorFloat, bit_width=bit_width)
            lstm_kw = dict(weight_bit_width=bit_width, io_bit_width=bit_width,
                           gate_acc_bit_width=bit_width, sigmoid_bit_width=bit_width,
                           tanh_bit_width=bit_width, cell_state_bit_width=bit_width)
            fc_kw = dict(weight_quant=Int8WeightPerTensorFloat, weight_bit_width=bit_width,
                         bias_quant=Int32Bias)
        else:
            emb_kw = dict(weight_quant=None)
            act_kw = dict(act_quant=None)
            lstm_kw = dict(weight_quant=None, bias_quant=None, io_quant=None,
                           gate_acc_quant=None, sigmoid_quant=None, tanh_quant=None,
                           cell_state_quant=None)
            fc_kw = dict(weight_quant=None, bias_quant=None)

        self.embedding = qnn.QuantEmbedding(vocab_size, embedding_dim, **emb_kw)

        # QuantEmbedding has no output activation quantizer of its own, so the
        # LSTM's input activation is quantized explicitly here.
        self.emb_quant = qnn.QuantIdentity(return_quant_tensor=False, **act_kw)

        self.lstm = qnn.QuantLSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                                  batch_first=True, **lstm_kw)

        # The masked mean runs in float; its result is the classifier's input
        # activation. return_quant_tensor=True hands QuantLinear the input scale
        # that Int32Bias needs to derive the bias scale.
        self.pool_quant = qnn.QuantIdentity(return_quant_tensor=quant, **act_kw)

        self.fc = qnn.QuantLinear(hidden_dim, output_dim, bias=True, **fc_kw)

    def forward(self, text, text_lengths):
        embedded = self.emb_quant(self.embedding(text))

        output = self.lstm(embedded)
        if isinstance(output, tuple):
            output = output[0]

        # Mean over real tokens only. The LSTM does consume the pad positions
        # here, but their hidden states are masked out and cannot affect the
        # earlier real positions.
        lengths = text_lengths.to(output.device)
        steps = torch.arange(output.shape[1], device=output.device)
        mask = (steps.unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(-1)
        pooled = (output * mask).sum(dim=1) / lengths.unsqueeze(1)

        return self.fc(self.pool_quant(pooled))


class QuantGenModel(nn.Module):
    """`GenModel` with a quantized encoder, LSTM, label embedding and decoder.

    Same padding caveat as `QuantDiscModel`. The forward pass is split in two
    because only the decoder depends on the candidate label: `encode` runs the
    expensive QuantLSTM once per batch, and `decode` is called once per class.
    Fusing them would run the LSTM |Y| times for no reason.

    The decoder's output is deliberately left unquantized -- it feeds a 27k-way
    log-softmax, and snapping logits to an 8-bit grid would inject error into
    exactly the quantity the classifier compares across labels.
    """

    def __init__(self, vocab_size, embedding_dim, label_emb_dim, hidden_dim, nclass,
                 n_layers=1, bit_width=8, quant=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.nclass = nclass

        if quant:
            emb_kw = dict(weight_quant=Int8WeightPerTensorFloat, weight_bit_width=bit_width)
            act_kw = dict(act_quant=Int8ActPerTensorFloat, bit_width=bit_width)
            lstm_kw = dict(weight_bit_width=bit_width, io_bit_width=bit_width,
                           gate_acc_bit_width=bit_width, sigmoid_bit_width=bit_width,
                           tanh_bit_width=bit_width, cell_state_bit_width=bit_width)
            dec_kw = dict(weight_quant=Int8WeightPerTensorFloat, weight_bit_width=bit_width)
        else:
            emb_kw = dict(weight_quant=None)
            act_kw = dict(act_quant=None)
            lstm_kw = dict(weight_quant=None, bias_quant=None, io_quant=None,
                           gate_acc_quant=None, sigmoid_quant=None, tanh_quant=None,
                           cell_state_quant=None)
            dec_kw = dict(weight_quant=None)

        self.encoder = qnn.QuantEmbedding(vocab_size, embedding_dim, **emb_kw)
        self.emb_quant = qnn.QuantIdentity(return_quant_tensor=False, **act_kw)
        self.lstm = qnn.QuantLSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                                  batch_first=True, **lstm_kw)
        self.label_encoder = qnn.QuantEmbedding(nclass, label_emb_dim, **emb_kw)

        # [h_t ; v_y] is requantized to a single scale before the decoder, which
        # is what a real integer matmul over the concatenation would require.
        self.dec_quant = qnn.QuantIdentity(return_quant_tensor=False, **act_kw)
        self.decoder = qnn.QuantLinear(hidden_dim + label_emb_dim, vocab_size,
                                       bias=False, **dec_kw)

    def encode(self, text):
        """Label-independent hidden states, (batch, time, hidden)."""
        output = self.lstm(self.emb_quant(self.encoder(text)))
        return output[0] if isinstance(output, tuple) else output

    def decode(self, states, label):
        """Vocabulary logits for one candidate label, (batch, time, vocab)."""
        batch, steps, _ = states.shape
        index = torch.full((1,), label, dtype=torch.long, device=states.device)
        label_emb = self.label_encoder(index).view(1, 1, -1).expand(batch, steps, -1)
        return self.decoder(self.dec_quant(torch.cat([states, label_emb], dim=-1)))

    def forward(self, text, label):
        return self.decode(self.encode(text), label)


def _remap_lstm(state_dict, src, dst, hidden, n_layers):
    """torch packs the gates into one matrix with two summed biases; Brevitas
    keeps a module per gate with a single bias."""
    mapped = {}
    for layer in range(n_layers):
        for index, gate in enumerate(LSTM_GATES):
            rows = slice(index * hidden, (index + 1) * hidden)
            prefix = f'{dst}.layers.{layer}.0.{gate}_gate_params'
            mapped[f'{prefix}.input_weight.weight'] = state_dict[f'{src}.weight_ih_l{layer}'][rows]
            mapped[f'{prefix}.hidden_weight.weight'] = state_dict[f'{src}.weight_hh_l{layer}'][rows]
            mapped[f'{prefix}.bias'] = (state_dict[f'{src}.bias_ih_l{layer}'][rows]
                                        + state_dict[f'{src}.bias_hh_l{layer}'][rows])
    return mapped


def _load(qmodel, mapped):
    missing, unexpected = qmodel.load_state_dict(mapped, strict=False)
    if unexpected:
        raise RuntimeError(f'unexpected keys when loading float weights: {unexpected}')
    # Everything left over must be quantizer state (scales/zero-points), which
    # calibration fills in. A missing *parameter* means the mapping is wrong.
    stragglers = [k for k in missing if not any(
        t in k for t in ('quant', 'scaling', 'zero_point', 'bit_width', 'tensor_quant', 'act_impl'))]
    if stragglers:
        raise RuntimeError(f'these real parameters were not populated: {stragglers}')
    return qmodel


def load_float_weights(qmodel, state_dict):
    """Copy a trained `DiscModel` state_dict into a `QuantDiscModel`."""
    mapped = {
        'embedding.weight': state_dict['embedding.weight'],
        'fc.weight': state_dict['fc.weight'],
        'fc.bias': state_dict['fc.bias'],
    }
    mapped |= _remap_lstm(state_dict, 'lstm', 'lstm', qmodel.hidden_dim, qmodel.n_layers)
    return _load(qmodel, mapped)


def load_gen_float_weights(qmodel, state_dict):
    """Copy a trained `GenModel` state_dict into a `QuantGenModel`.

    `GenModel` names its recurrent module `rnn`; the quantized model calls it
    `lstm`, so the prefix is remapped as well.
    """
    mapped = {
        'encoder.weight': state_dict['encoder.weight'],
        'label_encoder.weight': state_dict['label_encoder.weight'],
        'decoder.weight': state_dict['decoder.weight'],
    }
    mapped |= _remap_lstm(state_dict, 'rnn', 'lstm', qmodel.hidden_dim, qmodel.n_layers)
    return _load(qmodel, mapped)
