import json
from typing import List, Dict

# import nltk
from nltk.tokenize import RegexpTokenizer


# nltk.download('punkt') # use it if you needed


def load_dict() -> Dict[str, int]:
    with open("words_dictionary.json", "r") as f:
        voc = json.load(f)
    return voc


def tokenize(example: str) -> List[str]:
    tokenizer = RegexpTokenizer(r"\w+")
    words = [i.lower() for i in tokenizer.tokenize(example)]
    return words


def get_example(example: str | None = None) -> List[int]:
    voc: Dict = load_dict()
    if example is None:
        example = "Hello world! Go play games in the street!"
    words = tokenize(example)
    indexs = [voc[word] for word in words]
    return indexs


if __name__ == "__main__":
    result = get_example()
    print(result)
