import json
import argparse
from typing import List
from nltk.tokenize.util import align_tokens
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
    Regex,
    NormalizedString,
    PreTokenizedString
)
from tokenizers.pre_tokenizers import PreTokenizer

from tiny_tokenizer import pre_tokenize as pt

def get_tokens_and_offsets(text):
    tokens = pt(text, add_bos=False)
    token_positions = align_tokens(tokens, text)

    assert len(tokens) == len(token_positions)
    tokpos = [(tokens[ii], token_positions[ii][0], token_positions[ii][1]) for ii in range(len(tokens))]
    return tokpos


class CustomPreTokenizer:
    def _split(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:
        splits = []
        tokpos = get_tokens_and_offsets(str(normalized_string))

        for token, start, stop in tokpos:
            splits.append(normalized_string[start:stop])

        return splits

    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(self._split)


def make_tokenizer():
    with open('tiny_tokenizer/tokens.json', 'r') as f:
        vocab = json.load(f)
    #NOTE: first two are [bos], and [unk], will skip since they will be added later.
    vocab = vocab[2:]
    vd = dict([(i,ii+1) for ii,i in enumerate(vocab)])
    vd['[UNK]'] = 0

    tokenizer = Tokenizer(models.WordLevel(vocab = vd, unk_token = "[UNK]" ) )

    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.Replace("``", '"'),
            normalizers.Replace("''", '"'),
            normalizers.Replace("“", '"'),
            normalizers.Replace("”", '"'),
            normalizers.Replace("’", "'"),
            normalizers.NFKD(),
            normalizers.Lowercase(),
            normalizers.StripAccents(),
            normalizers.Replace(Regex(" {2,}"), " "),
        ]
    )

    tokenizer.pre_tokenizer = PreTokenizer.custom(CustomPreTokenizer())
    return tokenizer


def save_tokenizer(tokenizer, filename='tokenizer.json'):
    #note only way to save it since I used a custom pre-tokenizer:
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(filename)

if __name__ == "__main__":

    filename = 'tinytokenizer.json'
    tokenizer = make_tokenizer()
    save_tokenizer(tokenizer, filename=filename)

"""
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import PreTokenizer
tokenizer = Tokenizer.from_file("tinytokenizer.json")
tokenizer.pre_tokenizer = PreTokenizer.custom(CustomPreTokenizer())
# can't save custom pre-tokenizer (https://github.com/huggingface/tokenizers/issues/581)

example = "Once upon a time, there was a big pumpkin. It was hot outside, and the pumpkin was sad."

print(" ".join(tokenizer.encode(example).tokens))
"""

