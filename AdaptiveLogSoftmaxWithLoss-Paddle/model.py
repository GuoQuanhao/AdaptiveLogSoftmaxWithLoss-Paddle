import paddle
import paddle.nn as nn
from adaptive import AdaptiveLogSoftmaxWithLoss

class RNNModel_with_adaptive(nn.Layer):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel_with_adaptive, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout, time_major=True)
        self.adaptive_loss = AdaptiveLogSoftmaxWithLoss(nhid, ntoken, cutoffs=[round(ntoken/15), 3*round(ntoken/15)], div_value=4)
        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight = paddle.create_parameter(shape=self.encoder.weight.shape, dtype='float32',
                              default_initializer=paddle.nn.initializer.Uniform(-initrange, initrange))

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        return output, hidden

    def init_hidden(self, bsz):
        return (paddle.zeros([self.nlayers, bsz, self.nhid]),
                paddle.zeros([self.nlayers, bsz, self.nhid]))
