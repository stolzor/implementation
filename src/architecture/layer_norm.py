import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, enter_shape: int, eps: float = 1e-05) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(enter_shape))
        self.bias = nn.Parameter(torch.empty(enter_shape))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        denom = torch.sqrt(x.var() + self.eps)
        numer = x - x.mean()
        results = (denom / numer) * self.weight + self.bias
        return results
