import pickle
from typing import Dict, List

from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase, NFKC, Sequence, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


def get_bpe(data, vocab_size: int) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.normalizer = Sequence([NFKC(), StripAccents(), Lowercase()])
    tokenizer.decoder = ByteLevelDecoder()
    trainer_src = BpeTrainer(
        vocab_size=vocab_size, special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
    )

    def _batch_generator_src(batch_size: int = 10000):
        for i in range(0, len(data), batch_size):
            yield data["en"][i : i + batch_size]

        for i in range(0, len(data), batch_size):
            yield data["de"][i : i + batch_size]

    tokenizer.train_from_iterator(
        _batch_generator_src(), trainer=trainer_src, length=len(data)
    )

    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[
            ("[BOS]", 2),
            ("[EOS]", 3),
        ],
    )
    tokenizer = Tokenizer.from_file("data/tokenizer")

    return tokenizer


if __name__ == "__main__":
    with open("data/wmt14-train.pkl", "rb") as f:
        data: Dict[str, List[str]] = pickle.load(f)

    tokenizer = get_bpe(data, 60000)
    print(tokenizer.decode(tokenizer.encode("Hello!").ids))
