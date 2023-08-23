import torch

from src.machine_translation.machine_translation import MachineTranslation
from src.optimization.optimization import Adam
from src.loss.loss import CrossEntropyLoss
from .dataset import CustomTranslationDataset


def calc_performance(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss: CrossEntropyLoss,
) -> ...:

    loss = loss(pred, target)


def train_epoch(
    model: MachineTranslation,
    train_data: CustomTranslationDataset,
    optimizer: Adam,
    loss: CrossEntropyLoss,
    device: torch.device,
    smoothing: bool = True,
) -> ...:
    ...
