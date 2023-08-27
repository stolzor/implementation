from dataclasses import dataclass


@dataclass
class SettingAttention:
    MAX_LEN: int = 128
    D_MODELS: int = 512
    SIZES: int = 100
    VOCAB_SIZE: int = 1024
