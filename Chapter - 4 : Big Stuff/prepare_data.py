import torch
import pickle
import json
from torch.utils.data import Dataset
import config
import helper
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm 

class CustomDataset(Dataset):
    def __init__(self,data_type = 'train'):
        super(CustomDataset,self).__init__()
        self.max_seq_len = config.max_seq_len
        data = pickle.load(open(config.base_dir + config.train_test_data_file,'rb'))
        vocab = json.load(open(config.base_dir + config.vocab_file_name,"r"))
        self.word2index = vocab['word2index']
        self.pad_str  = 0

        if data_type == 'train':
            dataset = list()
            for inp,lbl in zip(data['X_train'],data['y_train']):
                dataset.append((inp,lbl))
            self.data = dataset
    
        elif "val" in data_type or "test" in data_type:
            dataset = list()
            for inp,lbl in zip(data['X_test'],data['y_test']):
                dataset.append((inp,lbl))
            self.data = dataset
        
        else:
            dataset = list()
            count = 0
            for inp,lbl in zip(data['X_test'],data['y_test']):
                dataset.append((inp,lbl))
                count += 1
                if count == 100:
                    break
            self.data = dataset

    def __len__(self):
        return len(self.data[:10000])

    def __getitem__(self,idx):
        seq,label = self.data[idx]

        # seq = [self.word2index[s] for s in seq]
        seq_padded = self.padded(seq)

        return {
            "seq_padded" : torch.tensor(seq_padded,dtype = torch.long),
            "label" : torch.tensor(label,dtype = torch.float)
        }

    def padded(self,x):
        x = x[:self.max_seq_len]
        x = x +[self.pad_str] * (self.max_seq_len - len(x))
        return x


# if __name__ == "__main__":
#     data = helper.read_data_file(config.base_dir,config.preprocessed_dataset_file)
#     # samples,word2index,index2word,uniq_words = build_vocabulary(data[config.input_column[0]])
    
#     # json.dump(
#     # {
#     #   "word2index" : word2index,
#     #   "index2word" : index2word
#     # },
#     vocabs = json.load(open(config.base_dir + config.vocab_file_name,"r"))
#     tokens = list()
#     for sentence in tqdm(data['trimmed_review']):
#         token_sentence = list()
#         for word in sentence.split():
#             token_sentence.append(vocabs['word2index'][word])
#         tokens.append(token_sentence)
#     samples = tokens
#     samples_lbl = data[config.target_columns[0]]
#     assert len(samples) == len(samples_lbl),f"input data {len(samples)} doesn't match with target {len(samples_lbl)}"
#     X_train, X_test, y_train, y_test = train_test_split(samples,samples_lbl,test_size=0.2,shuffle=True)

#     pickle.dump({
#         "X_train" : X_train,
#         "X_test" : X_test,
#         "y_train" : y_train.values.tolist(),
#         "y_test" : y_test.values.tolist(),
#     },open(config.base_dir + config.train_test_data_file,'wb'))
#     print("Data prepared.")