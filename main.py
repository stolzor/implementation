from typing import List

import torch

from for_test import get_example
from src.architecture.positional_encoder import PositionalEncoder
from src.architecture.scaled_dot_attention import ScaledDotProductAttention
from src.architecture.multi_head_attention import MultiHeadAttention
from src.architecture.settings import SettingAttention
from src.architecture.transformer_encoder import TransformerEncoder
from src.architecture.transformer_decoder import TransformerDecoder
from src.architecture.transformer import Transformer


example_ids: List[int] = get_example(None)
pos_encoder = PositionalEncoder(SettingAttention.D_MODELS, SettingAttention.MAX_LEN)
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


pos_encod = torch.unsqueeze(pos_encoder(example_ids), 0)

out_scaled = scaled_attention(pos_encod)
print("Scaled dot product attention: ", out_scaled.shape)

out_multi = multi_attention(pos_encod, pos_encod, pos_encod)
print("MultiHead attention: ", out_multi[0].shape)

out_encoder = transformer_encoder(pos_encod)
print("Transformer encoder: ", out_encoder[0].shape)

out_decoder = transformer_decoder(out_encoder[0], pos_encod)
print("Transformer decoder: ", out_decoder.shape)

out_transformer = transformer(example_ids, example_ids)
print("Transformer: ", out_transformer.shape)
