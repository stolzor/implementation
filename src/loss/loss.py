import torch
from torch import nn


class CrossEntropyLoss(nn.Module):
    def __init__(self, n_class: int, eps: float = 0.1) -> None:
        super().__init__()
        self.n_class = n_class
        self.eps = eps if eps is not None else 0.1

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        y_smooth = self.label_smoothing(y_true, self.n_class, self.eps)
        loss = -torch.sum(y_smooth * torch.log(y_pred))
        return loss

    def label_smoothing(
        self, y: torch.Tensor, n_class: int, eps: float
    ) -> torch.Tensor:
        result = y * (1 - eps) + eps / n_class
        return result
