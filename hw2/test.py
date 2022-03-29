from module import *
from math import log
import torch.nn.functional as F
import sys
#from utils import *

data_dir = sys.argv[1]
output_file = sys.argv[2]

# beam search
def beam_search_decoder(data, k):
	sequences = [[list(), 0.0]]
	# walk over each step in sequence
	for row in data:
		all_candidates = list()
		# expand each current candidate
		for i in range(len(sequences)):
			seq, score = sequences[i]
			for j in range(len(row)):
				candidate = [seq + [j], score - log(row[j])]
				all_candidates.append(candidate)
		# order all candidates by score
		ordered = sorted(all_candidates, key=lambda tup:tup[1])
		# select k best
		sequences = ordered[:k]
	return sequences

vocab_size = len(word_tokens)
BATCH_SIZE = 10

testing_data = make_dataset(data_dir+'/testing_data/feat/',data_dir+'/testing_label.json')
testing_X = [x for x, y in testing_data]
testing_Y = [y for x, y in testing_data]
testing_set = VideoDataset(testing_X,testing_Y)

test_loader = torch.utils.data.DataLoader(dataset=testing_set
                                            , batch_size=BATCH_SIZE
                                            , pin_memory=True)

#testing_labels = json.load('./data/testing_label.json')

if __name__ == "__main__":
    s2vt = S2VT(vocab_size=vocab_size, batch_size=BATCH_SIZE, hidden=500)
    s2vt = s2vt.cuda()
    s2vt.load_state_dict(torch.load("./s2vt_params.pkl"))
    s2vt.eval()
    for i, (x, y_list) in enumerate(test_loader):
        video = torch.Tensor(x.float()).cuda()
        caption = torch.LongTensor(np.array([convert_caption_ind(y_,word_tokens, reverse_word_tokens) for y_ in y_list[1]])).cuda()
        caption = caption.tolist()

        cap_out = s2vt(video)

        captions = []
        for tensor in cap_out:
            captions.append(tensor.tolist())
        captions = [[row[i] for row in captions] for i in range(len(captions[0]))]

        with open(output_file,'w') as fp:
            for j, j_cap in enumerate(captions):
                caption_string = construct_caption_ind(j_cap, word_tokens)
                fp.write(y_list[0][j]+','+caption_string+'\n')
                print(y_list[0][j], caption_string,sep=',')
