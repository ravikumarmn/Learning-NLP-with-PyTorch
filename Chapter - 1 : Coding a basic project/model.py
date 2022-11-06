import torch
import torch.nn as nn 
import pickle
from model1 import LSTM_cell

class BCModel(nn.Module):
    def __init__(self,args_dict,num_dim,tuple_in,pretrained_embd = None):
        """
        Given embedding_matrix: numpy array with vector for all words
        return prediction ( in torch tensor format)
        """
        super(BCModel, self).__init__()

        self.embedding = nn.Embedding(num_dim,
                        embedding_dim=args_dict['EMBED_SIZE'])
        if pretrained_embd != None:
            pickle_data = pickle.load(open(pretrained_embd,'rb'))
            pickle_data = torch.tensor(pickle_data["embedding_vector"],requires_grad = True).float()
            emb_matrix = nn.Parameter(pickle_data)
            assert self.embedding.weight.shape == emb_matrix.shape
            self.embedding.weight = emb_matrix


        # Embedding matrix actually is collection of parameter
        # self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype =))
        # Because we use pretrained embedding (GLove, Fastext,etc) so we turn off requires_grad-meaning we do not train gradient on embedding weight
        # self.embedding.weight.requires_grad = False
        # LSTM with hidden_size = 64
        self.lstm = LSTM_cell(args_dict['EMBED_SIZE'],args_dict["HIDDEN_SIZE"])
        # Input(512) because we use bi-directional LSTM ==> hidden_size*2 + maxpooling **2  = 64*4 = 256, will be explained more on forward method
        self.before_out = nn.Linear(128,64)

        self.out = nn.Linear(args_dict["HIDDEN_SIZE"], args_dict["n_labels"])
        self.sigmoid = nn.Sigmoid()
    def forward(self, x,tuple_in):
        # pass input (tokens) through embedding layer
        x = self.embedding(x)
        # fit embedding to LSTM
        hidden, _ = self.lstm(x,tuple_in)
        # apply mean and max pooling on lstm output
        avg_pool= torch.mean(hidden, 1)
        max_pool, index_max_pool = torch.max(hidden, 1)
        # concat avg_pool and max_pool ( so we have 256 size, also because this is bidirectional ==> 128*2 = 256)
        out = torch.cat((avg_pool, max_pool), 1) # ( 32,128)
        out = self.before_out(out)

        # fit out to self.out to conduct dimensionality reduction from 512 to 1
        out = self.out(out)
        out = self.sigmoid(out)
        # return output
        return out