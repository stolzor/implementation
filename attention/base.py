from abc import ABC, abstractmethod
from typing import List
import math

import torch


class BaseAttention(ABC):
    def __init__(self) -> None:
        ...

    @abstractmethod
    def forward(self, key, query, values):
        ...
