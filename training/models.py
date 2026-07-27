import torch.nn as nn
import torch.nn.utils.rnn
#import brevitas.nn as qnn

#from brevitas.quant import Int8WeightPerTensorFixedPoint, Int8ActPerTensorFixedPoint, SignedBinaryWeightPerTensorConst, SignedBinaryActPerTensorConst, SignedTernaryWeightPerTensorConst, SignedTernaryActPerTensorConst, Uint8ActPerTensorFixedPoint

#Class for full precision Model
class DiscModel(nn.Module):
    
    def __init__(self,vocab_size,embedding_dim,hidden_dim,output_dim,n_layers,bidirectional,dropout, bit_witdh):

        super(DiscModel,self).__init__()

        self.embedding = nn.Embedding(vocab_size,embedding_dim)

        # nn.LSTM's own `dropout` only fires *between* stacked layers, so it is a
        # no-op at num_layers=1. Dropout on the non-recurrent connections (the
        # embeddings and the classifier input) is applied explicitly below.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=(dropout if n_layers > 1 else 0.0), batch_first=True)

        self.drop = nn.Dropout(dropout)

        # Softmax layer over labels: V in R^{E x |Y|} plus the bias b in R^{|Y|}.
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)



    def forward(self,text,text_lengths):
        embedded = self.drop(self.embedding(text))

        # Pack so the LSTM never consumes padding, otherwise the trailing <pad>
        # positions steer the hidden states and pollute the document average.
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths, batch_first=True, enforce_sorted=False)

        packed_output, (hidden_state, cell_state) = self.lstm(packed)

        # Document representation is the average of the hidden states, not the
        # final one: p(y | x) prop. exp((1/T sum_t h_t)^T v_y + b_y). Padded
        # positions come back as exact zeros, so summing then dividing by the
        # true length gives the mean over real tokens only.
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        hidden = output.sum(dim=1) / text_lengths.to(output.device).unsqueeze(1)

        dense_outputs=self.fc(self.drop(hidden))

        return dense_outputs


class GenModel(nn.Module):
    def __init__(self, ntoken, ninp, nlabelembed, nhid, nlayers, nclass, dropout, use_cuda, 
        tied, use_bias, concat_label, avg_loss, one_hot):
        
        super(GenModel, self).__init__()
        
        self.drop = nn.Dropout(dropout)
        
        self.encoder = nn.Embedding(ntoken, ninp)
          
        self.label_encoder = nn.Embedding(nclass, nlabelembed)

        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout, batch_first=False)

        self.decoder = nn.Linear((nhid + nlabelembed), ntoken, False)

        self.nlayers = nlayers
        self.nhid = nhid
        self.loss = nn.CrossEntropyLoss()
        self.use_cuda = use_cuda
        self.nclass = nclass
        self.avg_loss = avg_loss
        self.use_bias = use_bias
        self.concat_label = concat_label
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, x_pred, y_ext, hidden, criterion, model_state = 'fp'):
        
        #print("x.data.shape:", x.data.shape)
        embedded_sents = self.encoder(x.data)
        #print("embedded_sents.shape:", embedded_sents.shape)

        embedded_label = self.label_encoder(y_ext.data)
        #print("y_ext.data:", y_ext.data)
        #print("embedded_label.shape:", embedded_label.shape)

        # Re-wrap the embedded tokens as a PackedSequence so the LSTM recurs
        # within each sentence. Feeding the flat `.data` buffer directly makes
        # the LSTM run down the whole packed buffer as one unbatched sequence,
        # carrying state across sentence and timestep boundaries.
        packed_sents = nn.utils.rnn.PackedSequence(
            embedded_sents, x.batch_sizes, x.sorted_indices, x.unsorted_indices)

        packed_output, (h_0, _) = self.rnn(packed_sents, hidden)
        output = packed_output.data
        #print("output.shape:", output.shape)
        #print("h_0 shape:", h_0.shape)

        if model_state == 'quant': #calibration of quantized model
            output=output.mean(dim=1)
            hidden_data = torch.cat([output, embedded_label], dim=1)  # float cat
        else: #inference with fp model
            hidden_data = torch.cat((output, embedded_label), 1)
        #print("hidden_data.shape:", hidden_data.shape)

        hidden_data = self.drop(hidden_data)
        #print("hidden_data.shape:", hidden_data.shape)

        out = self.decoder(hidden_data)
        #print("out.shape:", out.shape)
        return out

class MLPFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPFeatureExtractor, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [batch_size, sequence_length]
        x = self.embedding(x)                   # [batch_size, sequence_length, hidden_dim]
        x = torch.mean(x, dim=1)                # Mean pooling: [batch_size, hidden_dim]
        x = self.fc1(x)                         # Fully connected: [batch_size, hidden_dim]
        x = self.relu(x)                        # ReLU activation
        logits = self.fc2(x)                    # Logits: [batch_size, output_dim]

        return logits