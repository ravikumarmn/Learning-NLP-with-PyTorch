import torch
import torch.nn as nn 
import json 

class ClassifierModel(nn.Module):
    def __init__(self,vocab_len,params):
        super().__init__()
        self.embs = nn.Embedding(vocab_len,params["EMBED_SIZE"])
        self.linear1 = nn.Linear(in_features=params["EMBED_SIZE"],out_features=params["HIDDEN_SIZE"]//2)
        self.linear2 = nn.Linear(in_features=params["HIDDEN_SIZE"]//2,out_features=params["HIDDEN_SIZE"])
        self.dropout1 = nn.Dropout(params["DROP_OUT"])
        self.dropout2 = nn.Dropout(params["DROP_OUT"])
        self.relu = nn.LeakyReLU()
        self.transformer = nn.TransformerEncoderLayer( 
            d_model=params["HIDDEN_SIZE"],
            nhead = params["N_HEAD"],
            dim_feedforward= params["DIM_FORWARD"],
            dropout= params["DROP_OUT"],
            activation= "relu",
            batch_first=True,
        )
        self.fc = nn.Linear(in_features=params["HIDDEN_SIZE"],out_features=params["N_LAYERS"])

    def forward(self,x): 
        mask = (x!=0) # (8,45)
        emb = self.embs(x)
        out = self.relu(self.dropout1(self.linear1(emb)))
        out = self.relu(self.dropout2(self.linear2(out)))
        tras_out = self.transformer(out,src_key_padding_mask = (x == 0)) # (8,450,512)
        mask = mask.unsqueeze(-1) # (8,64,1)
        
        tras_mask = tras_out*mask
        sum_out = tras_mask.sum(dim = 1)/mask.sum(dim = 1) # (8,512)
        out = self.fc(sum_out) # (8,6)
        return out





