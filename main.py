from typing import List

from for_test import get_example
from attention.scaled_dot_attention import ScaledDotProductAttention
from attention.multi_head_attention import MultiHeadAttention
from attention.settings import SettingAttention
from encoder.transformer_encoder import TransformerEncoder
from decoder.transformer_decoder import TransformerDecoder


example_ids: List[int] = get_example(None)

scaled_attention = ScaledDotProductAttention(
    SettingAttention.D_MODELS, SettingAttention.MAX_LEN, True
)
multi_attention = MultiHeadAttention(
    SettingAttention.D_MODELS, SettingAttention.MAX_LEN, 4
)
transformer_encoder = TransformerEncoder(
    4, SettingAttention.D_MODELS, SettingAttention.MAX_LEN, [128, 512]
)
transformer_decoder = TransformerDecoder(
    4, SettingAttention.D_MODELS, SettingAttention.MAX_LEN, [128, 512]
)


pos_encod = scaled_attention.positional_encoding(example_ids)
print("Scaled dot product attention: ", scaled_attention([pos_encod]).shape)
print(
    "MultiHead attention: ", multi_attention(pos_encod, pos_encod, pos_encod)[0].shape
)
out_encoder = transformer_encoder(multi_attention.positional_encoding(example_ids))
print("Transformer encoder: ", out_encoder[0].shape)
# print(len(out_encoder[1]))
print(
    "Transformer decoder: ", transformer_decoder(out_encoder[1][:2], example_ids).shape
)
