import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self,dataframe,tokenizer,args):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = dataframe
        self.args = args
        self.targets = dataframe[args['LABELS']].values
        self.text_pair = dataframe['pairs']

    def __len__(self):
        return len(self.text_pair)

    def __getitem__(self,idx):
        pairs = self.text_pair.iloc[idx]
        inputs = self.tokenizer.encode_plus(
            pairs,
            max_length = self.args['MAX_LEN'],
            padding='max_length',
            truncation = True,
            return_tensors = "pt"
        )
        ids = inputs['input_ids'].squeeze(0)
        mask = inputs['attention_mask'].squeeze(0)
        token_type_ids = inputs["token_type_ids"].squeeze(0)
        target = self.targets[idx]

        return {
            "input_ids" : ids.long(),
            "token_type_ids" : token_type_ids.long(),
            "attention_mask" : mask.long(),
            "target" : torch.tensor(target,dtype = torch.float)
            
        }


    