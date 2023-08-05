from torch import nn


class PositionWiseFFN(nn.Module):
    def __init__(self, d_model: int, inner_dim: int) -> None:
        super().__init__()
        self.weight_1 = nn.Linear(d_model, inner_dim)
        self.weight_2 = nn.Linear(inner_dim, d_model)

    def forward(self, x):
        x = self.weight_2(self.weight_1(x))
        return x
