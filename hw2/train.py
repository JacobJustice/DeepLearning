from module import *
import torch
from torch import nn
#from utils import *


EPOCH = 30
BATCH_SIZE = 10
LEARNING_RATE = 0.0001
vocab_size = len(word_tokens)

training_data = make_dataset('./data/training_data/feat/','./data/training_label.json')
training_X = [x for x, y in training_data]
training_Y = [y for x, y in training_data]
training_set = VideoDataset(training_X,training_Y)
train_loader = torch.utils.data.DataLoader(dataset=training_set
                                            , batch_size=BATCH_SIZE
                                            , shuffle=True
                                            , pin_memory=True)
                                            
if __name__ == "__main__":
    s2vt = S2VT(vocab_size=vocab_size, batch_size=BATCH_SIZE)
    s2vt = s2vt.cuda()
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(s2vt.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCH):
        for i, (x, y_list) in enumerate(train_loader):
#            video, caption, cap_mask = fetch_train_data(BATCH_SIZE)
#            video, caption, cap_mask = torch.FloatTensor(video).cuda(), torch.LongTensor(caption).cuda(), \
#                                       torch.FloatTensor(cap_mask).cuda()
            video = x.cuda()
            caption = torch.Tensor(np.array([convert_caption(y_,word_tokens, reverse_word_tokens) for y_ in y_list[1]])).cuda()

            cap_out = s2vt(video, caption)
            cap_labels = caption[:, 1:].contiguous().view(-1)       # size [batch_size, 79]
            #cap_mask = cap_mask[:, 1:].contiguous().view(-1)        # size [batch_size, 79]

            logit_loss = loss_func(cap_out, cap_labels)
            #masked_loss = logit_loss*cap_mask
            #loss = torch.sum(masked_loss)/torch.sum(cap_mask)

            optimizer.zero_grad()
            logit_loss.backward()
            optimizer.step()

            if i%20 == 0:
                print("Epoch: %d  iteration: %d , loss: %f" % (epoch, i, loss))
                #write_txt(epoch, i, loss)
            if i%2000 == 0:
                torch.save(s2vt.state_dict(), "/data/video-captioning/Data/s2vt_params.pkl")
                print("Epoch: %d iter: %d save successed!" % (epoch, i))
