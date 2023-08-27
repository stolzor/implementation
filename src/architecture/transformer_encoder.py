from typing import Optional

import torch
from torch import nn

from . import *


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        n_blocks: int,
        d_model: int,
        enter_shape: int,
        inner_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.multi_head = MultiHeadAttention(n_blocks, d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.add_norm_1 = AddNorm(enter_shape)

        self.feed_forward = PositionWiseFFN(d_model, inner_dim)
        self.dropout_2 = nn.Dropout(dropout)
        self.add_norm_2 = AddNorm(enter_shape)

    def forward(
        self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        old_x = x.clone()
        x = self.multi_head(x, x, x, x_mask)
        x = self.dropout_1(x)
        x = self.add_norm_1(old_x, x)

        old_x = x.clone()
        x = self.feed_forward(x)
        x = self.dropout_2(x)
        x = self.add_norm_2(old_x, x)
        return x
