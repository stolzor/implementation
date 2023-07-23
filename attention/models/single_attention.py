import torch.nn as nn

from base import AttentionBase


class Attention(AttentionBase, nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, key, value, query):
        pass


if __name__ == "__main__":
    model = Attention()
