from abc import ABC, abstractmethod


class BaseAttention(ABC):
    def __init__(self) -> None:
        ...

    @abstractmethod
    def forward(self, key, query, values):
        ...
