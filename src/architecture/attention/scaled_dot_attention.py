from typing import Literal, Union, List

import torch
import torch.nn as nn
from torch.nn import functional as fn

from .base import BaseAttention


class ScaledDotProductAttention(nn.Module, BaseAttention):
    def __init__(
        self,
        d_model: int,
        sent_len: int | None = None,
        mask: bool = False,
        usage: Union[Literal["single"], Literal["multi"]] = "single",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.usage = usage
        self.sent_len = sent_len
        self.d_model = d_model
        self.mask = mask
        if self.sent_len is None:
            self.sent_len = 100

        self.dropout = nn.Dropout(dropout)
        self.key = nn.Linear(self.d_model, self.sent_len)
        self.query = nn.Linear(self.d_model, self.sent_len)
        self.value = nn.Linear(self.d_model, self.sent_len)

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        if self.usage == "single":
            assert len(x) == 1, (
                f"You are using {self.usage}, but the list does not contain 1 "
                "items (he contain {len(x)})"
            )
            x = x[0]
            key = self.key(x)
            query = self.query(x)
            value = self.value(x)
            permutes = (1, 0)
        else:
            assert len(x) == 3, (
                f"You are using {self.usage}, but the list does not contain 3 "
                "items (he contain {len(x)})"
            )
            key, value, query = x
            permutes = (0, 1, 3, 2)

        x = (query @ key.permute(permutes)) / self.sent_len**0.5
        if self.mask:
            mask = torch.triu(torch.ones(*x.shape[-2:]), diagonal=1)
            mask[mask.bool()] = -float("inf")
            x += mask

        result = self.dropout((fn.softmax(x, dim=-1))) @ value
        return result


if __name__ == "__main__":
    model = ScaledDotProductAttention()
