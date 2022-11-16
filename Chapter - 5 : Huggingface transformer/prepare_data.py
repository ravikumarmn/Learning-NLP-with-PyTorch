import torch
from torch import nn
import pickle 
import config 
import json 
from transformers import AutoTokenizer
from tqdm.auto import tqdm

             

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self,data_type = "train"):
        super().__init__()

        data = pickle.load(open(config.train_test_data_file,mode = "rb"))
        self.vocab = json.load(open(config.base_folder + config.base_dir + config.vocab_file_name,"r"))
        self.tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)   
        if data_type == 'train':
            # for inp,lbl in zip(data['X_train'],data['y_train']):
            sentences = [self.decode_word(ids) for ids in data['X_train']]
            self.tokens = self.batch_encode(sentences)
            self.labels = torch.tensor(data['y_train'],dtype=torch.float)
            

        elif "val" in data_type or "test" in data_type:
            sentences = [self.decode_word(ids) for ids in data['X_test']]
            self.tokens = self.batch_encode(sentences)
            self.labels = torch.tensor(data['y_test'],dtype=torch.float)

        else:
            sentences = [self.decode_word(ids) for ids in data['X_test'][:100]]
            self.tokens = self.batch_encode(sentences)
            self.labels = torch.tensor(data['y_test'][:100],dtype=torch.float)
        self.n_examples = len(self.tokens)
    def __len__(self):
        return self.n_examples

    def __getitem__(self,idx):
        items = {"input" : self.tokens[idx],
                "label" : self.labels[idx]}

        return items

    def batch_encode(self,total_data,batch_size=10):
        tokens_list = []
        for i in tqdm( range(0,len(total_data),batch_size) ):
            batch = total_data[i:i+batch_size]
            tokens = self.tokenizer.batch_encode_plus(batch,
                                padding = 'max_length',
                                truncation =True,
                                max_length = 300,
                                return_tensors = "pt",
                                )

            for i in range(len(tokens["input_ids"])):
                local_dict = {}
                for k,v in tokens.items():
                    local_dict[k] = v[i]
                tokens_list.append(local_dict)
        ##{input_ids:[[seq],[seq],[seq],[seq]],"attenmask":[[seq],[seq],[seq],[seq]]}
        return tokens_list



    def decode_word(self,word_list):
        sentence = [self.vocab['index2word'][str(index)] for index in word_list]
        return " ".join(sentence)
