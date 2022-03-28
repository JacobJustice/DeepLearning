#from tkinter import HIDDEN
import torch
from torch import nn
import json
from convert_caption import *

# load word_tokens
with open('./word_tokens.json') as fp:
    word_tokens = json.load(fp)
    reverse_word_tokens = generate_reverse(word_tokens)

class VideoDataset(Dataset):
    def __init__(self,X,Y):
        assert len(X) == len(Y)
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        _x = self.X[index]
        _y = self.Y[index]
        return _x, _y

class S2VT(nn.Module):
    def __init__(self, vocab_size, batch_size, frame_dim=4096, hidden=500, dropout=.5, n_step=80):
        super().__init__()
        self.batch_size=batch_size
        self.frame_dim = frame_dim # 4096
        self.hidden = hidden # whatever you want it to be
        self.n_step = n_step # sequence length (80)

        self.drop = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(frame_dim, hidden)
        self.linear2 = nn.Linear(hidden, vocab_size)

        # encoder
        self.lstm1 = nn.LSTM(hidden, hidden, batch_first=True, dropout=dropout)

        # decoder
        self.lstm2 = nn.LSTM(hidden, hidden, batch_first=True, dropout=dropout)

        self.embedding = nn.Embedding(vocab_size, hidden)

    #
    # video: extracted video features
    def forward(self, video, caption=None):
        # encode the video data
        video = video.contiguous().view(-1, self.frame_dim)
        video = self.drop(video)
        video = self.linear1(video) # embed video 
        video = video.view(-1, self.n_step) # reshape video tensor
        padding = torch.zeros([self.batch_size, self.n_step-1, self.hidden]).cuda()
        video = torch.cat((video, padding), 1) # pad video
        vid_out, _ = self.lstm1(video)
        
        # if in training output raw values
        if self.training:
            caption = self.embedding(caption[:, 0:self.n_step-1])
            padding = torch.zeros([self.batch_size, self.n_step-1, self.hidden]).cuda()
            caption = torch.cat((padding, caption), 1) # caption padding
            caption = torch.cat((caption, vid_out), 2) # caption input


            cap_out, state_cap = self.lstm2(caption)
            # padding input of the second layer of LSTM, 80 time steps

            # shape is (batch_size, 2*n_step-1, hidden)
            cap_out = cap_out[:, self.n_step:, :] # cut out padding
            cap_out = cap_out.contiguous().view(-1,self.hidden) # reshape cap_out
            cap_out = self.drop(cap_out) # dropout
            cap_out = self.linear2(cap_out)

            return cap_out

        #if not training output a string
        else:
            padding = torch.zeros([self.batch_size, self.n_step, self.hidden]).cuda()
            cap_input = torch.cat((padding, vid_out[:, 0:self.n_step, :]), 2)
            cap_out, state_cap = self.lstm2(cap_input)
            # padding input of the second layer of LSTM, 80 time steps

            bos_id = word_tokens['<BOS>']*torch.ones(self.batch_size, dtype=torch.long)
            bos_id = bos_id.cuda()
            cap_input = self.embedding(bos_id)
            cap_input = torch.cat((cap_input, vid_out[:, self.n_step, :]), 1)
            cap_input = cap_input.view(self.batch_size, 1, 2*self.hidden)

            cap_out, state_cap = self.lstm2(cap_input, state_cap)
            cap_out = cap_out.contiguous().view(-1, self.hidden)
            cap_out = self.drop(cap_out)
            cap_out = self.linear2(cap_out)
            cap_out = torch.argmax(cap_out, 1)
            # input ["<BOS>"] to let the generate start

            caption = []
            caption.append(cap_out)
            # put the generate word index in caption list, generate one word at one time step for each batch
            for i in range(self.n_step-2):
                cap_input = self.embedding(cap_out)
                cap_input = torch.cat((cap_input, vid_out[:, self.n_step+1+i, :]), 1)
                cap_input = cap_input.view(self.batch_size, 1, 2 * self.hidden)

                cap_out, state_cap = self.lstm2(cap_input, state_cap)
                cap_out = cap_out.contiguous().view(-1, self.hidden)
                cap_out = self.drop(cap_out)
                cap_out = self.linear2(cap_out)
                cap_out = torch.argmax(cap_out, 1)
                # get the index of each word in vocabulary
                caption.append(cap_out)
            return caption
            # size of caption is [79, batch_size]
