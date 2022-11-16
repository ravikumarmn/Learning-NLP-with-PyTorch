import torch
import torch.nn as nn 
import pickle
from transformers import AutoModelForSequenceClassification
import config

class ClfModel(nn.Module):
    def __init__(self,):
        super().__init__()
         
        self.bert = AutoModelForSequenceClassification.from_pretrained(config.checkpoint,num_labels=1)

        self.fc = nn.Linear(in_features=512,out_features=1)

    def forward(self,input_id,mask):
        out = self.bert(input_ids= input_id, attention_mask=mask,return_dict = False)
        return torch.sigmoid(out[0])




