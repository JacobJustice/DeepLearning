import json

def generate_reverse(dictionary):
    return {v: k for k, v in dictionary.items()}

if __name__ == "__main__":
    # generate dictionary of words
    with open('./data/training_label.json') as fp:
        training_content = json.load(fp)

    # 0: Unknown token
    # 1: Beginning of Sentence
    # 2: End of Sentence
    word_tokens = { 0:'<UNK>',
                    1:'<BOS>',
                    2:'<EOS>',
                    3:'<PAD>',
                    3:'.',
                    4:',',
                    5:'!'
                    }

    # for every video 
    for vid_data in training_content:
        # list of all captions
        caption_list = (vid_data['caption'])
        for caption in caption_list:
            clean_caption = caption.replace('.','').replace(',','').replace('!','')
            split_caption = clean_caption.split(" ")
            for word in split_caption:
                if word not in word_tokens.values():
    #                print(len(word_tokens),word)
                    word_tokens.update({len(word_tokens):word})

    with open('word_tokens.json','w') as fp:
        json.dump(word_tokens, fp)
