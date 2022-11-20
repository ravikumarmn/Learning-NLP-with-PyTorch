import os
import torch
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from custom_data import CustomDataset
import config
from torch.utils.data import Dataset
from model import BertClassifier


class CustomDataset(Dataset):
    def __init__(self,dataframe,tokenizer,args):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = dataframe
        self.args = args
        # self.targets = dataframe[args['LABELS']].values
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

        return {
            "input_ids" : ids.long(),
            "token_type_ids" : token_type_ids.long(),
            "attention_mask" : mask.long(),
        }

def CustumLoader(params):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    test_dataset = pd.read_csv(params["TEST_PRE_DIR"])
    testing_set = CustomDataset(
        test_dataset,
        tokenizer=tokenizer,
        args=params)

    test_dataloader = DataLoader(
        testing_set,
        batch_size = params["BATCH_SIZE"],
        drop_last = True)
        # (16tuple_in,1,450)
    return test_dataloader,test_dataset['ID']

class Inference:
    def __init__(self,params):
        self.params = params
        all_checkpoints = os.listdir(params['CHECKPOINT_DIR'])
        checkpoints = torch.load(params["CHECKPOINT_DIR"]+all_checkpoints[-1])
        self.model = BertClassifier()
        self.model.load_state_dict(checkpoints['model_state_dict'])
        self.model.eval()
        self.test_dataloader,self.id_column= CustumLoader(params)

    def predict(self):
        tqdm_obj_batch = tqdm(self.test_dataloader,total=len(self.test_dataloader),leave=None)
        ones_list = list()
        for idx,batch_data in enumerate(tqdm_obj_batch):
            input_ids = batch_data['input_ids']
            token_type_ids = batch_data['token_type_ids']
            attention_mask = batch_data['attention_mask']
            
            logits=  self.model(input_ids,attention_mask,token_type_ids)

            output = (torch.sigmoid(logits)>0.5).long()
            ones_list.append(output)
        all_preds = torch.cat(ones_list,dim = 0)
        df = pd.DataFrame(all_preds,columns=self.params['LABELS'])
        ids = self.id_column[:len(all_preds)]
        df.insert(0,"ID",ids)
        return df
    
    def save(self,df):
        df.to_csv(self.params["DATASET_DIR"]+"sample_submission.csv",index = False)
        return df
        
def main():
    params = {k:v for k,v in config.__dict__.items() if "__" not in k}
    print("#####INFERENCE####\n")
    inf = Inference(params=params)
    df = inf.predict()
    print("\n")
    print(f'Saving file to {params["DATASET_DIR"]+"sample_submission.csv"}')
    inf.save(df)
    return df


if __name__ == "__main__":
    
    df = main()