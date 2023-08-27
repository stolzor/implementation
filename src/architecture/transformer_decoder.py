from typing import List

import torch
from torch import nn

from . import *


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        n_blocks: int,
        d_model: int,
        enter_shape: int,
        temperature: float = 0.8,
        inner_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.multi_head_1 = MultiHeadAttention(n_blocks, d_model, temperature)
        self.dropout_1 = nn.Dropout(dropout)
        self.add_norm_1 = AddNorm(enter_shape)

        self.multi_head_2 = MultiHeadAttention(n_blocks, d_model, temperature)
        self.dropout_2 = nn.Dropout(dropout)
        self.add_norm_2 = AddNorm(enter_shape)

        self.feed_forward = PositionWiseFFN(d_model, inner_dim)
        self.dropout_3 = nn.Dropout(dropout)
        self.add_norm_3 = AddNorm(enter_shape)

    def forward(
        self,
        encoder_output: List[torch.Tensor],
        x: torch.Tensor,
        enc_mask: torch.Tensor,
        x_mask: torch.Tensor,
    ) -> torch.Tensor:
        old_x = x.clone()
        x = self.multi_head_1(x, x, x, x_mask)
        x = self.dropout_1(x)
        x = self.add_norm_1(old_x, x)

        old_x = x.clone()

        x = self.multi_head_2(encoder_output[0], encoder_output[1], x)
        x = self.dropout_2(x)
        x = self.add_norm_2(old_x, x)

        old_x = x.clone()
        x = self.feed_forward(x)
        x = self.dropout_3(x)
        x = self.add_norm_3(old_x, x)

        return x
