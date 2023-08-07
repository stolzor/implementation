from typing import List

import torch

from for_test import get_example
from attention.scaled_dot_attention import ScaledDotProductAttention
from attention.multi_head_attention import MultiHeadAttention
from attention.settings import SettingAttention
from encoder.transformer_encoder import TransformerEncoder
from decoder.transformer_decoder import TransformerDecoder
from transformer.transformer import Transformer


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
transformer = Transformer(
    5, [128, 512], 4, SettingAttention.D_MODELS, SettingAttention.MAX_LEN
)


pos_encod = torch.unsqueeze(scaled_attention.positional_encoding(example_ids), 0)

out_scaled = scaled_attention(pos_encod)
print("Scaled dot product attention: ", out_scaled.shape)

out_multi = multi_attention(pos_encod, pos_encod, pos_encod)
print("MultiHead attention: ", out_multi[0].shape)

out_encoder = transformer_encoder(pos_encod)
print("Transformer encoder: ", out_encoder[0].shape)

out_decoder = transformer_decoder(out_encoder[0], pos_encod)
print("Transformer decoder: ", out_decoder.shape)

out_transformer = transformer(pos_encod)
print("Transformer: ", out_transformer.shape)
