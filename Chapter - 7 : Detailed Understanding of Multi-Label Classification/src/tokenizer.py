import json
import config
import torch

dataset_to_path_mapping = {
    "topic-modelling-research-articles" : config.VOCAB_DIR
}

class Tokenizer:
    def __init__(self,dataset_name):
        dataset_path = dataset_to_path_mapping.get(dataset_name)
        if dataset_path:
            vocab = json.load(open(dataset_path,"r"))
            self.word_to_token = {k:int(v) for k,v in vocab["vocabs"].items()}
            self.token_to_word = {int(v):k for k,v in vocab["vocabs"].items()}
        else:
            raise KeyError

    def encode(self,text,padding = False,max_length = None,return_tensors = ""):
        if isinstance(text,str):
            tokens = [self.word_to_token[x] for x in text.split()]
        elif isinstance(text,(tuple,list)):
            tokens = list()
            tokens.append([self.word_to_token[word] for word in " ".join(text).split()])
        if padding:
            tokens = [self.padding_seq(token,max_length) for token in tokens]
        if padding == "max_length":
            max_length = len(self.word_to_token)
            tokens = self.padding_seq(tokens,max_length)

        if return_tensors == "pt":
            tokens = torch.tensor(tokens,dtype = torch.long)
            return tokens
        else:
            return tokens

    def __call__(self,text,padding = False,max_length = None,return_tensors = ""):
        return self.encode(text,padding,max_length,return_tensors)

    def decode(self,x) -> str:
        if isinstance(x,torch.Tensor):
            x = x.flatten()
        words = [self.token_to_word[t] for t in x]
        return " ".join(words)

    def padding_seq(self,seq:list,max_length) -> list: 
        if not max_length:
            max_length = 250
        seq = seq[:max_length]
        padded = seq+[0]*(max_length-len(seq))
        return padded