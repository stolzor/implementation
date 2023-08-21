import os
import random
from typing import Iterable, List, Dict, Tuple

import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from tokenizers import Tokenizer

from .tokenizer import get_bpe


class CustomTranslationDataset(Dataset):
    def __init__(
        self,
        data: List[List[torch.Tensor]],
        vocab_size: int,
    ) -> None:
        super().__init__()
        self.inputs, self.attn_masks, self.labels = data
        self.vocab_sie = vocab_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> ...:
        src_sentence, trg_sentence = self.inputs[index], self.labels[index]
        attn_masks = self.attn_masks[index]
        return {"src": src_sentence, "trg": trg_sentence, "attn_masks": attn_masks}


def preprocesser_data(data: pd.DataFrame, tokenizer: Tokenizer) -> ...:
    if os.path.isfile("data/wmt14-train-preprocessed.parquet"):
        print("Preprocessed dataframe exists")
        return pd.read_parquet("data/wmt14-train-preprocessed.parquet").iloc[:1000, :]

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


def random_batcher(
    sentences: List[str], labels: List[str], batch_size: int
) -> List[List[int]]:
    batch_ordered_sentences = []
    batch_ordered_labels = []

    while len(sentences) > 0:
        if (len(batch_ordered_sentences) % 1000) == 0:
            print("Selected {} batches".format(len(batch_ordered_sentences)))

        to_take = min(batch_size, len(sentences))
        select = random.randint(0, len(sentences) - to_take)
        batch_sent = sentences[select : select + to_take]
        batch_lab = labels[select : select + to_take]

        batch_ordered_sentences.append([i for i in batch_sent])
        batch_ordered_labels.append([i for i in batch_lab])

        del sentences[select : select + to_take]
        del labels[select : select + to_take]

    return [batch_ordered_sentences, batch_ordered_labels]


def add_padding(
    sentnences: List[str], labels: List[str], tokenizer: Tokenizer
) -> List[List[torch.Tensor]]:
    arr_inputs = []
    arr_attn_masks = []
    arr_labels = []
    pad_token_id = tokenizer.get_vocab()["[PAD]"]
    for batch_inputs, batch_labels in zip(sentnences, labels):
        batch_padded_inputs = []
        batch_attn_masks = []
        batch_padded_target = []

        inp_max_size = max([len(s) for s in batch_inputs])
        trg_max_size = max([len(s) for s in batch_labels])

        for i, s in enumerate(batch_inputs):
            inp_num_pads = inp_max_size - len(s)
            trg_num_pads = trg_max_size - len(batch_labels[i])

            padded_input = list(s) + [pad_token_id] * inp_num_pads
            padded_target = list(batch_labels[i]) + [pad_token_id] * trg_num_pads

            attn_masks = [1] * len(s) + [0] * inp_num_pads
            batch_padded_inputs.append(padded_input)
            batch_attn_masks.append(attn_masks)
            batch_padded_target.append(padded_target)

        arr_inputs.append(torch.tensor(batch_padded_inputs))
        arr_attn_masks.append(torch.tensor(batch_attn_masks))
        arr_labels.append(torch.tensor(batch_padded_target))

    return [arr_inputs, arr_attn_masks, arr_labels]


def smart_batchers(
    data: pd.DataFrame, batch_size: int, tokenizer: Tokenizer
) -> List[List[torch.tensor]]:
    sentences = data["en"].tolist()
    labels = data["de"].tolist()

    ordered_sentnces, ordered_labels = random_batcher(sentences, labels, batch_size)
    inputs, attn_masks, labels = add_padding(
        ordered_sentnces, ordered_labels, tokenizer
    )
    return inputs, attn_masks, labels


def split_train_valid_test(
    data: pd.DataFrame, len_data: int, sizes: List[int]
) -> Tuple[pd.DataFrame]:
    train_size, valid_size = sizes

    train, valid, test = np.split(
        data.sample(frac=1),
        [int(train_size * len_data), int((train_size + valid_size) * len_data)],
    )

    return train, valid, test


def get_dataset(
    data: pd.DataFrame,
    vocab_size: int,
    batch_size: int,
    train_valid_test_sizes: Iterable[float] = [0.7, 0.2, 0.1],
    tokenizer_path: str = "data/tokenizer",
) -> Dict[str, CustomTranslationDataset]:
    tokenizer = get_bpe(data, vocab_size)
    tokenizer.save(tokenizer_path)

    data = preprocesser_data(data, tokenizer)
    data.to_parquet("data/wmt14-train-preprocessing.parquet", index=False)
    train_size, valid_size, _ = train_valid_test_sizes

    train, valid, test = split_train_valid_test(
        data, len(data), [train_size, valid_size]
    )

    print("Train data preprocessing...")
    train = smart_batchers(train, batch_size, tokenizer)

    print("Valid data preprocessing...")
    valid = smart_batchers(valid, batch_size, tokenizer)

    print("Test data preprocessing...")
    test = smart_batchers(test, batch_size, tokenizer)

    train = CustomTranslationDataset(train, vocab_size)
    valid = CustomTranslationDataset(valid, vocab_size)
    test = CustomTranslationDataset(test, vocab_size)

    return {"train": train, "valid": valid, "test": test}


if __name__ == "__main__":
    df = pd.read_parquet("data/wmt14-train.parquet")
    vocab_size = 60000

    dataset = get_dataset(df, vocab_size, 128)
    print(next(iter(dataset["train"])))
