import torch
from torch import nn

from .base import BaseAttention
from .scaled_dot_attention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module, BaseAttention):
    def __init__(
        self, d_model: int, sent_len: int, n_blocks: int, masked: bool = True
    ) -> None:
        super().__init__()
        assert d_model % n_blocks == 0, "Embedding sent_len / N blocks should == 0"
        self.sent_len = sent_len
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.masked = masked

        self.v = nn.Linear(self.d_model, self.d_model)
        self.q = nn.Linear(self.d_model, self.d_model)
        self.k = nn.Linear(self.d_model, self.d_model)
        self.o = nn.Linear(self.d_model, self.d_model)

        self.attention = ScaledDotProductAttention(
            self.d_model // self.n_blocks, self.sent_len, self.masked, "multi"
        )

    def forward(
        self, key: torch.Tensor, value: torch.Tensor, query: torch.Tensor
    ) -> torch.Tensor:
        key, value, query = list(
            map(lambda x: torch.unsqueeze(x, 0), [key, value, query])
        )  # add batch_size axis

        old_size = key.shape
        key = self.k(key)
        value = self.v(value)
        query = self.q(query)
        x = self.attention(list(map(lambda x: self.transpose(x), [key, value, query])))
        # print("x inner", x[0][0])
        x = x.reshape(*old_size)
        x = self.o(x)

        return x, [key, value, query]

    def transpose(self, x):
        sizes = x.shape
        return x.reshape(sizes[0], self.n_blocks, sizes[1], -1)
