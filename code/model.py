from functools import reduce
import math
import torch
import numpy
from torch import matmul, nn, softmax, tensor
from config import *

n_segments = 3  # ?


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model=MODEL_DIM, max_len=MAX_LEN):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class BERTEmbedding(nn.Module):
    def __init__(self):
        super(BERTEmbedding, self).__init__()
        # token embedding
        self.tok_embed = nn.Embedding(VOCAB_SIZE + 1, MODEL_DIM, padding_idx=6)
        # position embedding
        self.pos_embed = PositionalEmbedding()
        # segment(token type) embedding
        self.seg_embed = nn.Embedding(n_segments, MODEL_DIM, padding_idx=0)
        self.dropout = nn.Dropout(p=DROPOUT)

    def forward(self, x, y):
        _x = self.tok_embed(x) + self.pos_embed(x) + self.seg_embed(y)
        return self.dropout(_x)


class Attention(nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None):
        """Scaled dot product attention."""
        _x = matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        if mask is not None:
            _x = _x.masked_fill(mask == 0, -1e4)

        _x = softmax(_x, -1)

        if dropout is not None:
            _x = dropout(_x)

        return matmul(_x, value), _x


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int = 8, dropout: float = 0.1) -> None:
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_k = MODEL_DIM // self.num_heads
        self.linq = nn.Linear(MODEL_DIM, MODEL_DIM)
        self.link = nn.Linear(MODEL_DIM, MODEL_DIM)
        self.linv = nn.Linear(MODEL_DIM, MODEL_DIM)
        self.attention = Attention()
        self.out = nn.Linear(MODEL_DIM, MODEL_DIM)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        _q = (
            self.linq(query)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        _k = (
            self.link(key)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        _v = (
            self.linv(value)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )

        # batchSize -1 h d_k -> batch_size d_m
        _x, _ = self.attention(_q, _k, _v, mask, self.dropout)
        _x = (
            _x.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.d_k)
        )

        return self.out(_x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, dff=PFF_DIM, dropout: float = 0.1) -> None:
        super(PositionwiseFeedForward, self).__init__()

        self.linear_in = nn.Linear(MODEL_DIM, dff)
        self.linear_out = nn.Linear(dff, MODEL_DIM)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear_out(self.dropout(self.activation(self.linear_in(x))))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-4):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class TransformerBlock(nn.Module):
    def __init__(self, dropout) -> None:
        super(TransformerBlock, self).__init__()

        self.pff = PositionwiseFeedForward(dropout=dropout)
        self.mha = MultiHeadAttention(dropout=dropout)
        self.layer_norm_1 = LayerNorm(MODEL_DIM)
        self.layer_norm_2 = LayerNorm(MODEL_DIM)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        _x = self.layer_norm_1(x)
        s1 = x + self.dropout(self.mha(_x, _x, _x, mask))
        s2 = s1 + self.dropout(self.pff(self.layer_norm_2(s1)))
        return self.dropout(s2)


class BERT(nn.Module):
    def __init__(self, dropout=0.1, layers=12) -> None:
        super(BERT, self).__init__()
        self.dropout = dropout

        self.embedding = BERTEmbedding()
        self.layers = nn.ModuleList(
            [TransformerBlock(self.dropout) for _ in range(layers)]
        )

    def forward(self, x, y):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        _x = self.embedding(x, y)
        _x = reduce(lambda x, y: y(x, mask), self.layers, _x)

        return _x


class MaskedLanguageModel(nn.Module):
    def __init__(self) -> None:
        super(MaskedLanguageModel, self).__init__()
        self.linear = nn.Linear(MODEL_DIM, VOCAB_SIZE)
        self.softmax = nn.LogSoftmax(-1)

    def forward(self, x) -> tensor:
        return self.softmax(self.linear(x))


class NextSentencePrediction(nn.Module):
    def __init__(self) -> None:
        super(NextSentencePrediction, self).__init__()
        self.linear = nn.Linear(MODEL_DIM, 2)
        self.softmax = nn.LogSoftmax(-1)

    def forward(self, x) -> tensor:
        # Only grab info from [cls] token
        return self.softmax(self.linear(x[:, 0]))


class BERTLM(nn.Module):
    def __init__(self) -> None:
        super(BERTLM, self).__init__()

        self.bert = BERT()
        self.MLM = MaskedLanguageModel()
        self.NSP = NextSentencePrediction()

    def forward(self, x, y):
        b = self.bert(x, y)
        return self.NSP(b), self.MLM(b)


class ScheduledOptim:
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, n_warmup_steps=10000):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = numpy.power(MODEL_DIM, -0.5)

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return numpy.min(
            [
                numpy.power(self.n_current_steps, -0.5),
                numpy.power(self.n_warmup_steps, -1.5) * self.n_current_steps,
            ]
        )

    def _update_learning_rate(self):
        """Learning rate scheduling per step"""

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr
