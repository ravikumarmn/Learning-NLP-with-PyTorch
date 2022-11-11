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

    def forward(self,x):
        out = self.bert(**x)

        return torch.sigmoid(out.logits)




