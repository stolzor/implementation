import torch
from torch import nn

from models.base import BaseAttention
from models.scaled_dot_attention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module, BaseAttention):
    def __init__(self, d_model: int, size: int, n_blocks: int) -> None:
        super().__init__()
        assert d_model % n_blocks == 0, "Embedding size / N blocks should == 0"
        self.size = size
        self.d_model = d_model
        self.n_blocks = n_blocks

        self.v = nn.Linear(d_model, d_model)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(d_model // n_blocks, size, "multi")

    def forward(self, x):
        x = self.positional_encoding(x)
        x = torch.unsqueeze(x, 0) # add batch_size axis
        old_size = x.shape
        key = self.transpose(self.k(x))
        value = self.transpose(self.v(x))
        query = self.transpose(self.q(x))
        
        x = self.attention([key, value, query])
        x = x.reshape(*old_size)

        return self.o(x)
    
    def transpose(self, x):
        sizes = x.shape
        return x.reshape(sizes[0], self.n_blocks, sizes[1], -1)