from dataclasses import dataclass
from typing import List


@dataclass
class SettingAttention:
    MAX_LEN: int = 128
    D_MODELS: int = 512
    SIZES: List[int] = []
