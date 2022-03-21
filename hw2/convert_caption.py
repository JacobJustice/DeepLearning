from gen_json import generate_reverse
import json
import torch
from torch import nn

def convert_caption(caption_str, word_toks, reverse_word_toks=generate_reverse(word_toks)):

    return torch.Tensor()

if __name__ == "__main__":
    with open('./data/training_label.json') as fp:
        word_toks = json.load(fp)

    convert_caption('hello world',word_toks)
