from gen_json import generate_reverse
import re
import json
import torch
from torch import nn
import random
import numpy as np

punctuation = ['.',',','!']

def make_dataset(data_dir, label):
    with open(label) as fp:
        training_labels = json.load(fp)
    training_data = []
    for i, vid_data in enumerate(training_labels):
        x_data = np.load(data_dir + vid_data['id']+'.npy')
        for caption in vid_data["caption"]:
            y_data = (vid_data['id'], caption)
            training_data.append((x_data, y_data))
            break

    return training_data

def convert_caption(caption_str, word_tokens, reverse_word_tokens, num_words=80):
    #print('input:',caption_str)
#    print(word_tokens['0'])
#    print(reverse_word_tokens['girl'])

#    caption_matrix = [ [0 for j in range(len(word_tokens))] for i in range(num_words) ]
    caption_matrix = np.zeros((num_words, len(word_tokens)))
    caption_matrix[:, 2] = 1
    #print(caption_matrix)

    #split_caption = re.split('(\.|!|, )', caption_str)
    #print(split_caption)
    #split_caption = caption_str.replace('.','').replace('!','').replace(', ','').split()
    split_caption = caption_str.split()
    #print(split_caption)
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
            #print(i,word,'key error!')
            index = "0"
        #print(i, index, word)
        caption_matrix[i][2] = 0
        caption_matrix[i][int(index)] = 1

    return np.array(list(caption_matrix))

def construct_caption(caption_matrix, word_tokens):
    out_sentence = ""
    for i, word_vec in enumerate(caption_matrix):
        word_vec_arr = np.array(word_vec)
        #print('vec',word_vec)
        #print('shape',word_vec.shape)
        #print('sum',sum(word_vec))
        choice_index = int(np.argmax(word_vec))
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

    for i in range(100):
        caption_matrix = convert_caption('A chef, prepares raw poultry.',word_tokens,reverse_word_tokens)
        #print('output',construct_caption(caption_matrix,word_tokens))
        caption_matrix = convert_caption("The parakeet pecked at the phone's numbers and made a call!",word_tokens,reverse_word_tokens)
        #print('output',construct_caption(caption_matrix,word_tokens))
        caption_matrix = convert_caption("A man and two women walk across the beach.",word_tokens,reverse_word_tokens) 
        #print('output',construct_caption(caption_matrix,word_tokens))

#    training_dataset = make_dataset('./data/training_data/feat/', './data/training_label.json')
#    with open('./data/training_data_x.npy','wb') as fp:
#        np.save(fp, np.array([x for x,y in training_dataset]))
#    with open('./data/training_data_y.npy','wb') as fp:
#        np.save(fp, np.array([y for x,y in training_dataset]))
