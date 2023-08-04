from typing import List

from for_test import get_example
from models.scaled_dot_attention import ScaledDotProductAttention
from models.multi_head_attention import MultiHeadAttention
from settings import SettingAttention


example_ids: List[int] = get_example(None)

attention = ScaledDotProductAttention(SettingAttention.D_MODELS, SettingAttention.MAX_LEN)
attention = MultiHeadAttention(SettingAttention.D_MODELS, SettingAttention.MAX_LEN, 4)


print("Scaled dot product attention: ", attention([example_ids]).shape)
print("MultiHead attention: ", attention(example_ids).shape)