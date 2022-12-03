import torch
import torch.nn as nn 
import pickle

class ClassifierModel(nn.Module):
    def __init__(self,):
        super().__init__()
        self.embs = nn.Embedding(800000,32)
        self.linear1 = nn.Linear(in_features=32,out_features=512)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead = 8,
            dim_feedforward= 2048,
            dropout= 0.1,
            activation= "relu",
            batch_first=True,
        )
        self.fc = nn.Linear(in_features=512,out_features=6)

    def forward(self,x): 
        batch_sz,seq_len = x.size()
        src_m = x == 0
        mask = x != 0
        src_m = src_m.unsqueeze(1).repeat(1,seq_len,1)
        src_m = src_m.repeat(8,1,1)
        emb = self.embs(x)
        out = self.linear1(emb)
        tras_out = self.transformer(out,src_mask = src_m)
        mask = mask.unsqueeze(-1)
        out = self.fc(tras_out)
        tras_mask = out*mask
        mean_out_logits = torch.sum(tras_mask,dim = 1)/mask.sum(dim = 1)
        return mean_out_logits





