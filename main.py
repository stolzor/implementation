from typing import List

from for_test import get_example
from attention.scaled_dot_attention import ScaledDotProductAttention
from attention.multi_head_attention import MultiHeadAttention
from attention.settings import SettingAttention


example_ids: List[int] = get_example(None)

scaled_attention = ScaledDotProductAttention(
    SettingAttention.D_MODELS, SettingAttention.MAX_LEN, True
)
multi_attention = MultiHeadAttention(
    SettingAttention.D_MODELS, SettingAttention.MAX_LEN, 4
)


print("Scaled dot product attention: ", scaled_attention([example_ids]).shape)
print("MultiHead attention: ", multi_attention(example_ids).shape)
