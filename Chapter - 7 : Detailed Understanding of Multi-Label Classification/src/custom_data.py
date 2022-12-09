import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self,data,tokenizer,args):
        super().__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.targets = data[args['LABELS']].values
        self.text_pair = data['pairs']
        self.pos_by_neg_weight = self.pos_neg(self.targets.sum(axis=0),len(self))
        self.neg_by_pos_weight = self.neg_pos(self.targets.sum(axis=0),len(self))

    def __len__(self):
        return len(self.text_pair)

    def __getitem__(self,idx):
        pairs = eval(self.text_pair.iloc[idx])      
        inputs = self.tokenizer( 
            pairs,
            max_length = self.args['MAX_LEN'],
            padding=True,
            return_tensors = "pt"
            )
        ids = inputs.squeeze(0)
        target = self.targets[idx]

        return {
            "input_ids" : ids.long(),
            "target" : torch.tensor(target,dtype = torch.float)
        }


    def pos_neg(self,d,total_len):
        weights = d/(total_len- d)
        return weights

    def neg_pos(self,d,total_len):
        weights = (total_len- d)/d
        return weights