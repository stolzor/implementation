from typing import List

from src.utils.tokenizer import get_bpe


def get_example(example: str | None = None) -> List[int]:
    tokenizer = get_bpe(None, None)
    if example is None:
        example = "Hello world! Go play games in the street!"
    words = tokenizer.encode(example).ids
    return words


if __name__ == "__main__":
    result = get_example()
    print(result)
