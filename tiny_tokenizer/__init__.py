import json
import os
import numpy as np
import regex as re

current_dir = os.path.dirname(os.path.abspath(__file__))
tokens_path = os.path.join(current_dir, 'tokens.json')

def normalize(text):
    text = text.lower().strip()
    return text

with open(tokens_path, 'r') as f:
    _tokens = json.load(f)

    

# exclude documents with these characters
excluded_characters_pat = re.compile(r'[:-;`\\<>\[\]=_@~/+&*%#{}$\t|-]')

# after removing documents with excluded characters, make sure that the document doesn't have a match with this pattern
unallowed_characters = re.compile('[^a-zA-Z.?!()0-9" ,\n\']')

single_character_toks = re.compile(r'[ ,."\'0-9?!()\n-]')

def normalize(text):
    text = text.lower().strip()
    return text

def pre_tokenize(text, known_toks=None, add_bos=True):
    matches = list(single_character_toks.finditer(text))

    i = 0
    res = []
    for match in matches:
        span = match.span()
        assert span[0] >= i
        assert span[1] > span[0]

        if span[0] > i:
            res.append(text[i:span[0]])
        res.append(text[span[0]:span[1]])
        i = span[1]
    if i < len(text):
        res.append(text[i:])

    res = [substr for substr in res if substr != ' ']
    if known_toks is not None:
        res = [tok if tok in known_toks else f'[unk]' for tok in res]

    if add_bos:
        res = ['[bos]'] + res
    return res


def get_tok_strs(text, known_toks=None, add_bos=True):
    text = normalize(text)
    tok_strs = pre_tokenize(text, known_toks=known_toks, add_bos=add_bos)

    return tok_strs

def encode(text):
    global _tokens
    tok_strs = get_tok_strs(text)
    tok_ids = [_tokens.index(tok_str) if tok_str in _tokens else 1 for tok_str in tok_strs]  # [unk] token id is 1
    return tok_ids

enc = encode

def tok_split(text):
    tok_ids = encode(text)
    tok_strs = [_tokens[tok_id] for tok_id in tok_ids]
    return tok_strs


def decode(tok_ids):
    return ' '.join([_tokens[tok_id] for tok_id in tok_ids])

dec = decode

class Tokenizer:
    def __init__(self):
        pass
    def encode(self, text):
        return encode(text)
    def decode(self, tok_ids):
        return decode(tok_ids)


    
