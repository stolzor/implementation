import json
from typing import List, Dict

# import nltk
from nltk.tokenize import RegexpTokenizer


# nltk.download('punkt') # use it if you needed


def load_dict() -> Dict:
    with open("words_dictionary.json", "r") as f:
        voc = json.load(f)
    return voc


def tokenize(example: str) -> List[str]:
    tokenizer = RegexpTokenizer(r"\w+")
    words = [i.lower() for i in tokenizer.tokenize(example)]
    return words


def get_example(example: str | None = None) -> Dict[str, int]:
    voc: Dict = load_dict()
    if example is None:
        example = "Hello, world! I can calculate!"
    words = tokenize(example)
    word2index = {word: voc[word] for word in words}
    return word2index


if __name__ == "__main__":
    result = get_example()
    print(result)
