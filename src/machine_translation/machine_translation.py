from typing import List

from torch import nn

from src.architecture.transformer import Transformer


class MachineTranslation(nn.Module):
    def __init__(
        self,
        n_layers: int,
        enter_shape: List[int],
        n_blocks: int,
        d_model: int,
        sent_len: int,
        inner_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.transfomer = Transformer(
            n_layers=n_layers,
            enter_shape=enter_shape,
            n_blocks=n_blocks,
            d_model=d_model,
            sent_len=sent_len,
            inner_dim=inner_dim,
            dropout=dropout,
        )
