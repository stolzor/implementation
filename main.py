import torch

from src.architecture.transformer import Transformer
from src.loss.loss import CrossEntropyLoss
from src.optimization.optimization import Adam


device = torch.device("cuda")
transformer = Transformer(2, 512, 4, 512, 1024, 512).to(device)

pad_ids = torch.ones((128, 512), dtype=torch.long).to(device)
out_transformer = transformer(pad_ids, pad_ids)
print("Transformer: ", out_transformer.shape)

optim = Adam(params=transformer.parameters())
loss = CrossEntropyLoss(2)
