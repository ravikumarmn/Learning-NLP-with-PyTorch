from torch import nn
import torch

class ClassifierModel(nn.Module):
    def __init__(self,args_dict):
        super().__init__()
        self.args_dict = args_dict
        self.embs = nn.Embedding(args_dict["N_WORDS"],args_dict['EMBED_SIZE'])
        self.expand_dim = nn.Linear(args_dict["EMBED_SIZE"],args_dict["HIDDEN_SIZE"])
        encoder_layer = nn.TransformerEncoderLayer(d_model=args_dict["HIDDEN_SIZE"],nhead=args_dict["N_HEAD"]) 
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=args_dict["N_LAYERS"])        
        self.final_layer = nn.Linear(args_dict["HIDDEN_SIZE"], args_dict["NUM_LABELS"])

    def forward(self,input_ids):
        emb = self.embs(input_ids) # [16, 450, 32]
        emb = self.expand_dim(emb)
        src = self.transformer_encoder(emb)
        avg_pool= torch.sum(src, 1) # 16,512
        logits = self.final_layer(avg_pool) 
        return logits 


