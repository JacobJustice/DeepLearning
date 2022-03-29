
import torch
from torch import nn
from torch.utils.data import Dataset
import json
from convert_caption import *

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
    def __init__(self, vocab_size, batch_size=10, frame_dim=4096, hidden=256, dropout=0.5, n_step=80):
        super(S2VT, self).__init__()
        self.batch_size = batch_size
        self.frame_dim = frame_dim
        self.hidden = hidden
        self.n_step = n_step

        self.drop = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(frame_dim, hidden)
        self.linear2 = nn.Linear(hidden, vocab_size)

        self.lstm1 = nn.LSTM(hidden, hidden, batch_first=True)
        self.lstm2 = nn.LSTM(2*hidden, hidden, batch_first=True)

        self.embedding = nn.Embedding(vocab_size, hidden)

    def forward(self, video, caption=None):
        video = video.contiguous().view(-1, self.frame_dim)
        video = self.drop(video)
        video = self.linear1(video)                   
        video = video.view(-1, self.n_step, self.hidden)
        padding = torch.zeros([self.batch_size, self.n_step-1, self.hidden]).cuda()
        video = torch.cat((video, padding), 1)        
        vid_out, state_vid = self.lstm1(video)

        if self.training:
            caption = self.embedding(caption[:, 0:self.n_step-1])
            padding = torch.zeros([self.batch_size, self.n_step, self.hidden]).cuda()
            caption = torch.cat((padding, caption), 1)        
            caption = torch.cat((caption, vid_out), 2)        

            cap_out, state_cap = self.lstm2(caption)
            
            cap_out = cap_out[:, self.n_step:, :]
            cap_out = cap_out.contiguous().view(-1, self.hidden)
            cap_out = self.drop(cap_out)
            cap_out = self.linear2(cap_out)
            return cap_out
            
        else:
            padding = torch.zeros([self.batch_size, self.n_step, self.hidden]).cuda()
            cap_input = torch.cat((padding, vid_out[:, 0:self.n_step, :]), 2)
            cap_out, state_cap = self.lstm2(cap_input)
            

            bos_id = int(reverse_word_tokens['<BOS>'])*torch.ones(self.batch_size, dtype=torch.long)
            bos_id = bos_id.cuda()
            cap_input = self.embedding(bos_id)
            cap_input = torch.cat((cap_input, vid_out[:, self.n_step, :]), 1)
            cap_input = cap_input.view(self.batch_size, 1, 2*self.hidden)

            cap_out, state_cap = self.lstm2(cap_input, state_cap)
            cap_out = cap_out.contiguous().view(-1, self.hidden)
            cap_out = self.drop(cap_out)
            cap_out = self.linear2(cap_out)
            cap_out = torch.argmax(cap_out, 1)
            

            caption = []
            caption.append(cap_out)
            
            for i in range(self.n_step-2):
                cap_input = self.embedding(cap_out)
                cap_input = torch.cat((cap_input, vid_out[:, self.n_step+1+i, :]), 1)
                cap_input = cap_input.view(self.batch_size, 1, 2 * self.hidden)

                cap_out, state_cap = self.lstm2(cap_input, state_cap)
                cap_out = cap_out.contiguous().view(-1, self.hidden)
                cap_out = self.drop(cap_out)
                cap_out = self.linear2(cap_out)
                cap_out = torch.argmax(cap_out, 1)
                
                caption.append(cap_out)
            return caption
            
