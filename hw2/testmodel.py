import torch
from torch import nn
import numpy as np
from lstm_objs import LSTM_Softmax, VideoDataset
from convert_caption import construct_caption, convert_caption, make_dataset
from gen_json import generate_reverse
import json

# Sequential ( LSTM -> Softmax )
torch.set_default_dtype(torch.double)
# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# load word_tokens
with open('./word_tokens.json') as fp:
    word_tokens = json.load(fp)
    reverse_word_tokens = generate_reverse(word_tokens)

model = torch.load('./model1648146987.model').to(device)

testing_data = make_dataset('./data/testing_data/feat/','./data/testing_label.json')
testing_X = [x for x, y in testing_data]
testing_Y = [y for x, y in testing_data]
testing_set = VideoDataset(testing_X,testing_Y)

batch_size = 150
learning_rate = 0.01
test_loader = torch.utils.data.DataLoader(dataset=testing_set
                                            , batch_size=batch_size
                                            , shuffle=True
                                            , pin_memory=True)

loss_function = nn.CrossEntropyLoss()

n_total_steps = len(test_loader)
for i, (x, y_list) in enumerate(test_loader):
    x = x.to(device)
    y = torch.Tensor(np.array([convert_caption(y_,word_tokens, reverse_word_tokens) for y_ in y_list]))
    y = y.to(device)
    model.zero_grad()

    caption_out = model(x)

    print(y.shape,caption_out.shape)
    loss = loss_function(caption_out, y)
    out_loss = loss.item()
    loss.backward()
    #optimizer.step()
    print ()
    print (f'Step [{i+1}/{n_total_steps}], Loss: {out_loss:.4f}')
    print ('y    :',y_list[0])
    print ('y_hat:',construct_caption(caption_out[0].detach().cpu(),word_tokens))
