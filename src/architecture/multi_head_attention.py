from typing import Optional

import torch
from torch import nn

from .scaled_dot_attention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, n_blocks: int, d_model: int, temperature: float = 0.8) -> None:
        super().__init__()
        assert d_model % n_blocks == 0, "Embedding sent_len / N blocks should == 0"
        self.temperature = temperature
        self.d_model = d_model
        self.n_blocks = n_blocks

        self.v = nn.Linear(self.d_model, self.d_model)
        self.q = nn.Linear(self.d_model, self.d_model)
        self.k = nn.Linear(self.d_model, self.d_model)
        self.o = nn.Linear(self.d_model, self.d_model)

        self.attention = ScaledDotProductAttention(self.temperature)

    def forward(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        query: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is not None:
            mask = mask.unsqueeze(1)

        old_size = key.shape
        key = self.k(key)
        value = self.v(value)
        query = self.q(query)
        attention = self.attention(
            list(map(lambda x: self.transpose(x), [key, value, query])), mask
        )

        x = attention.reshape(*old_size)
        x = self.o(x)

        return x

    def transpose(self, x):
        sizes = x.shape
        return x.reshape(
            sizes[0], -1, self.n_blocks, sizes[-1] // self.n_blocks
        ).transpose(1, 2)
