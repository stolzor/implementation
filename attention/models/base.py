from abc import ABC


class AttentionBase(ABC):
    def __init__(self) -> None:
        ...

    def forward(self, key, query, values):
        ...
