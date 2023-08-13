import os
from typing import Literal, Iterable

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

from .tokenizer import get_bpe


class CustomTranslationDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        vocab_size: int,
    ) -> None:
        super().__init__()
        self.data = data
        self.vocab_sie = vocab_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> ...:
        print(index)
        src_sentence, trg_sentence = self.data["en"][index], self.data["de"][index]
        return {"src": src_sentence, "trg": trg_sentence}


def preprocesser_data(data: pd.DataFrame, tokenizer: Tokenizer) -> ...:
    if os.path.isfile("data/wmt14-train-preprocessed.parquet"):
        print("Preprocessed dataframe exists")
        return pd.read_parquet("data/wmt14-train-preprocessed.parquet")

    print("Start tokenized...")
    data["en"] = data["en"].apply(lambda x: tokenizer.encode(x).ids)
    data["de"] = data["de"].apply(lambda x: tokenizer.encode(x).ids)
    print("End tokenized")

    print("Start sort by len str")
    data.sort_values(by="en", inplace=True, key=lambda x: x.str.len(), ascending=False)
    data.reset_index(inplace=True, drop=True)
    print("End sort")

    print("Save dataframe")
    data.to_parquet("data/wmt14-train-preprocessed.parquet", index=False)
    return data


def get_dataloader(
    data: pd.DataFrame,
    vocab_size: int,
    batch_size: int,
    train_valid_test_sizes: Iterable[float] = [0.7, 0.2, 0.1],
    max_seq_len: Literal["auto"] | int = "auto",
    tokenizer_path: str = "data/tokenizer",
) -> ...:
    tokenizer = get_bpe(data, vocab_size)
    tokenizer.save(tokenizer_path)

    data = preprocesser_data(data, tokenizer)

    train_size, valid_size, _ = train_valid_test_sizes
    train, valid, test = np.split(
        data.sample(frac=1, random_state=42),
        [int(train_size * len(data)), int((train_size + valid_size) * len(data))],
    )
    train.reset_index(drop=True, inplace=True)
    valid.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    train = CustomTranslationDataset(train, vocab_size)
    valid = CustomTranslationDataset(valid, vocab_size)
    test = CustomTranslationDataset(test, vocab_size)

    train_loader = DataLoader(train, batch_size, shuffle=True)
    valid_loader = DataLoader(valid, batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    df = pd.read_parquet("data/wmt14-train.parquet")
    vocab_size = 60000

    train, valid, test = get_dataloader(df, vocab_size)
    print(next(iter(train)))
