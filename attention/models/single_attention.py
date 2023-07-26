from typing import List

import torch.nn as nn

from .base import AttentionBase
from settings import SettingAttention


class Attention(nn.Module, AttentionBase):
    def __init__(self, d_models, max_len, sizes: List[int]) -> None:
        super().__init__()
        self.key = nn.Linear(d_models, 100)

    def forward(self, x):
        print('TEST')
        pass


if __name__ == "__main__":
    model = Attention()
