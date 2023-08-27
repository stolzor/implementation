import torch

from src.architecture import Transformer
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
    model: Transformer,
    train_data: CustomTranslationDataset,
    optimizer: Adam,
    loss_func: CrossEntropyLoss,
    device: torch.device,
) -> ...:
    model.train()

    # results = {"accuracy": 0.0, "loss": 0.0}

    for i in range(len(train_data)):
        data = train_data[i]
        src_sentences = data["src"].to(device)
        trg_sentences = data["trg"].to(device)
        # attn_masks = data["attn_masks"].to(device)

        model.zero_grad()

        pred = model(src_sentences, trg_sentences)
        loss = loss_func(pred, trg_sentences)
        loss.backward()
        optimizer.step()


def eval_epoch(
    model,
    test_data: CustomTranslationDataset,
    optimizer: Adam,
    loss: CrossEntropyLoss,
    device: torch.device,
) -> ...:
    model.eval()
