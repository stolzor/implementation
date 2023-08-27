import torch
from torch import nn

from . import *


def get_pad_mask(sequence: torch.Tensor, pad_idx: int | float):
    return (sequence != pad_idx).unsqueeze(-2)


def get_mask(size: int) -> torch.Tensor:
    mask = (1 - torch.triu(torch.ones((1, size, size)), diagonal=1)).bool()
    return mask


class Transformer(nn.Module):
    def __init__(
        self,
        n_layers: int,
        enter_shape: int,
        n_blocks: int,
        d_model: int,
        n_vocab: int,
        sent_len: int,
        inner_dim: int = 2048,
        dropout: float = 0.1,
        pad_idx: int = 0,
        temperature: float = 0.8,
    ) -> None:
        super().__init__()
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.enter_shape = enter_shape
        self.d_model = d_model
        self.n_vocab = n_vocab
        self.sent_len = sent_len
        self.inner_dim = inner_dim
        self.dropout = dropout
        self.pad_idx = pad_idx
        self.temperature = temperature

        self.device = torch.device("cuda:0")

        self.src_word_emb = nn.Embedding(
            self.n_vocab, self.d_model, padding_idx=pad_idx
        )
        self.src_pos_encoder = PositionalEncoder(self.d_model, self.sent_len, dropout)
        self.src_dropout = nn.Dropout(self.dropout)
        self.encoders = nn.ModuleList(
            [
                TransformerEncoder(
                    self.n_blocks,
                    self.d_model,
                    self.enter_shape,
                    self.inner_dim,
                    self.dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.trg_word_emb = nn.Embedding(n_vocab, d_model, padding_idx=pad_idx)
        self.trg_pos_decoder = PositionalEncoder(self.d_model, self.sent_len, dropout)
        self.trg_dropout = nn.Dropout(self.dropout)
        self.decoders = nn.ModuleList(
            [
                TransformerDecoder(
                    self.n_blocks,
                    self.d_model,
                    self.enter_shape,
                    self.temperature,
                    self.inner_dim,
                    self.dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.linear = nn.Linear(self.d_model, self.n_vocab)
        self.softmax = nn.Softmax(dim=-1)

    def add_axis(self, x) -> torch.Tensor:
        if isinstance(x, list):
            x = torch.unsqueeze(self.pos_encoder(x), 0)
        return x

    def preprocessing_seq(
        self, seq: torch.Tensor, target: bool = False
    ) -> torch.Tensor:
        if target:
            return get_pad_mask(seq, self.pad_idx).to(self.device) & get_mask(
                seq.size(-1)
            ).to(self.device)
        return get_pad_mask(seq, self.pad_idx)

    def forward(self, src_sen: torch.Tensor, trg_sen: torch.Tensor) -> torch.Tensor:
        if not all(isinstance(e, torch.Tensor) for e in [src_sen, trg_sen]):
            raise TypeError("All input values should be consists type torch.Tensor")

        src_mask = self.preprocessing_seq(src_sen).to(self.device)
        trg_mask = self.preprocessing_seq(trg_sen, True).to(self.device)

        src_word_emb = self.src_dropout(
            self.src_pos_encoder(self.src_word_emb(src_sen))
        )
        for i in range(self.n_layers):
            src_word_emb = self.encoders[i](src_word_emb, src_mask)

        trg_word_emb = self.trg_dropout(
            self.trg_pos_decoder(self.trg_word_emb(trg_sen))
        )
        for i in range(self.n_layers):
            trg_word_emb = self.decoders[i](
                [src_word_emb, src_word_emb], trg_word_emb, src_mask, trg_mask
            )

        x = self.linear(trg_word_emb)
        x = self.softmax(x)

        return x.view(-1, x.size(2))
