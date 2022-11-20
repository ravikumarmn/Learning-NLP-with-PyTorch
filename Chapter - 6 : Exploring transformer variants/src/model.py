from torch import nn
import torch
from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.linear1 = nn.Linear(in_features=768,out_features=6)

    def forward(self,input_ids,attention_mask,token_type_ids):
        output = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids  = token_type_ids
            )
        logits = self.linear1(output.pooler_output)
        return logits