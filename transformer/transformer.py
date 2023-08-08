from typing import List

import torch
from torch import nn

from encoder.transformer_encoder import TransformerEncoder
from decoder.transformer_decoder import TransformerDecoder


class Transformer(nn.Module):
    def __init__(
        self,
        n_layers: int,
        enter_shape: List[int],
        n_blocks: int,
        d_model: int,
        sen_len: int,
        inner_dim: int = 2048,
    ) -> None:
        super().__init__()
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.enter_shape = enter_shape
        self.d_model = d_model
        self.sen_len = sen_len
        self.inner_dim = inner_dim

        self.encoders = nn.ModuleList(
            [
                TransformerEncoder(
                    self.n_blocks,
                    self.d_model,
                    self.sen_len,
                    self.enter_shape,
                    self.inner_dim,
                )
                for _ in range(n_layers)
            ]
        )
        self.decoders = nn.ModuleList(
            [
                TransformerDecoder(
                    self.n_blocks,
                    self.d_model,
                    self.sen_len,
                    self.enter_shape,
                    self.inner_dim,
                )
                for _ in range(n_layers)
            ]
        )

        self.linear = nn.Linear(self.d_model, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        for i in range(self.n_layers):
            encoder_output, attention = self.encoders[i](x)

        for i in range(self.n_layers):
            x = self.decoders[i](encoder_output, x)

        x = self.linear(x)
        x = self.softmax(x).reshape(x.shape[0], self.sen_len)

        return x
