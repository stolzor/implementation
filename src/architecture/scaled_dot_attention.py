from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn import functional as fn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature: float = 0.8, dropout: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: List[torch.Tensor], mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        assert (
            len(x) == 3
        ), "Enter values should consists key, value and query parameters"
        key, value, query = x

        x = (query @ key.transpose(-2, -1)) / self.temperature

        if mask is not None:
            x = x.masked_fill(mask == 0, -1e-9)

        result = self.dropout((fn.softmax(x, dim=-1))) @ value
        return result


if __name__ == "__main__":
    model = ScaledDotProductAttention()
