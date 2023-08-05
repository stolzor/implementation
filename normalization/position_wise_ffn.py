from torch import nn


class PositionWiseFFN(nn.Module):
    def __init__(self, d_models: int, inner_dim: int) -> None:
        super().__init__()
        self.weight_1 = nn.Linear(d_models, inner_dim)
        self.weight_2 = nn.Linear(inner_dim, d_models)
