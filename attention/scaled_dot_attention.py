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
        usage: Union[Literal["single"], Literal["multi"]] = "single",
    ) -> None:
        super().__init__()
        self.usage = usage
        self.sent_len = sent_len
        self.d_model = d_model
        if self.size is None:
            self.size = 100

        self.key = nn.Linear(self.d_model, self.sent_len)
        self.query = nn.Linear(self.d_model, self.sent_len)
        self.value = nn.Linear(self.d_model, self.sent_len)

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        if self.usage == "single":
            assert (
                len(x) == 1
            ), f"You are using {self.usage}, but the list does not contain 1 \
                items (he contain {len(x)})"
            x = self.positional_encoding(x[0])
            key = self.key(x)
            query = self.query(x)
            value = self.value(x)
            permutes = (1, 0)
        else:
            assert (
                len(x) == 3
            ), f"You are using {self.usage}, but the list does not contain 3 \
                items (he contain {len(x)})"
            key, value, query = x
            permutes = (0, 1, 3, 2)

        x = (query @ key.permute(permutes)) / self.sent_len**0.5
        result = (fn.softmax(x, dim=1)) @ value
        return result


if __name__ == "__main__":
    model = ScaledDotProductAttention()
