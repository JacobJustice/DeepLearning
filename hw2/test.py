from module import *
#from utils import *

def print_caption_from_tensor(caption_tens_list):
    pass

vocab_size = len(word_tokens)
BATCH_SIZE = 10

testing_data = make_dataset('./data/testing_data/feat/','./data/testing_label.json')
testing_X = [x for x, y in testing_data]
testing_Y = [y for x, y in testing_data]
testing_set = VideoDataset(testing_X,testing_Y)

learning_rate = 0.01
test_loader = torch.utils.data.DataLoader(dataset=testing_set
                                            , batch_size=BATCH_SIZE
                                            , shuffle=True
                                            , pin_memory=True)


if __name__ == "__main__":
    s2vt = S2VT(vocab_size=vocab_size, batch_size=BATCH_SIZE)
    s2vt = s2vt.cuda()
    s2vt.load_state_dict(torch.load("./s2vt_params_final.pkl"))
    s2vt.eval()
    for i, (x, y_list) in enumerate(test_loader):
        #video, caption, cap_mask = fetch_val_data(batch_size=BATCH_SIZE)
        video = torch.Tensor(x.float()).cuda()
        caption = torch.LongTensor(np.array([convert_caption_ind(y_,word_tokens, reverse_word_tokens) for y_ in y_list[1]])).cuda()

        cap_out = s2vt(video)

        captions = []
        for tensor in cap_out:
            captions.append(tensor.tolist())
        # size of captions : [79, batch_size]

        # transform captions to [batch_size, 79]
        captions = [[row[i] for row in captions] for i in range(len(captions[0]))]
        caption = caption.tolist()

        for i, in enumerate(captions)


        for i, (cap_y, cap_yhat) in enumerate(zip(captions, caption)):
            #print(cap_y,cap_yhat)
            print()
            print('............................\nModel Caption:\n')
            print(construct_caption_ind(cap_y, word_tokens))
            print('............................\nCorrect Caption:\n')
            print(construct_caption_ind(cap_yhat, word_tokens))
