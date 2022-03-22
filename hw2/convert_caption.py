from gen_json import generate_reverse
import re
import json
import torch
from torch import nn
import random
import numpy as np

punctuation = ['.',',','!']

def convert_caption(caption_str, word_tokens, reverse_word_tokens, num_words=80):
    print('input:',caption_str)
#    print(word_tokens['0'])
#    print(reverse_word_tokens['girl'])

    caption_matrix = [ [0 for j in range(len(word_tokens))] for i in range(num_words) ]

    split_caption = re.split('(\.|!|, )', caption_str)
    split_caption = caption_str.split()
    for i, word in enumerate(split_caption):
        if len(word) > 1:
            if ',' in word or '.' in word or '!' in word:
                #print(word)
                del split_caption[i]
                split_word = re.split('(\.|!|,)', word)
                split_caption.insert(i, split_word[0])
                split_caption.insert(i+1, split_word[1])

    #print(len(split_caption),split_caption)
    for i, word in enumerate(split_caption):
        try:
            # find index of this word in word_vec
            index = reverse_word_tokens[word]
        except KeyError:
            # if word isn't in known word_tokens give index of unknown
            print(i,word,'key error!')
            index = "0"
        #print(i, index, word)
        caption_matrix[i][int(index)] = 1

    for word_vec in caption_matrix:
        # if nothing in this word_vec
        if sum(word_vec) == 0:
            word_vec[2] = 1

    return torch.Tensor(caption_matrix)

def construct_caption(caption_matrix, word_tokens):
    out_sentence = ""
    for i, word_vec in enumerate(caption_matrix):
        word_vec_arr = np.array(word_vec)
        choice_index = np.random.choice(list(range(len(word_vec))), p=word_vec)
        if choice_index == 2:
            break
        choice_word = word_tokens[str(choice_index)]
        if choice_word in punctuation:
            out_sentence = out_sentence[:-1]
        out_sentence += word_tokens[str(choice_index)] + ' '

    return out_sentence

if __name__ == "__main__":
    with open('./word_tokens.json') as fp:
        word_tokens = json.load(fp)
        reverse_word_tokens = generate_reverse(word_tokens)

    caption_matrix = convert_caption('A chef, prepares raw poultry.',word_tokens,reverse_word_tokens)
    print('output',construct_caption(caption_matrix,word_tokens))
    caption_matrix = convert_caption("The parakeet pecked at the phone's numbers and made a call!",word_tokens,reverse_word_tokens)
    print('output',construct_caption(caption_matrix,word_tokens))
    caption_matrix = convert_caption("A man and two women walk across the beach.",word_tokens,reverse_word_tokens) 
    print('output',construct_caption(caption_matrix,word_tokens))

