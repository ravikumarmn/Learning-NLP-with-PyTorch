import config
import torch
import wandb
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from helper import split_dataset
from custom_data import CustomDataset
from torch.utils.data import DataLoader
from model import BertClassifier
from torch.optim import Adam
from sklearn.metrics import f1_score,accuracy_score
from torch import nn
from tqdm.auto import tqdm

def train(model,train_loader,optimizer,params,loss):
    train_loss = list()
    all_pred = list()
    all_true = list()
    model.train()
    tqdm_obj_batch = tqdm(train_loader,total=len(train_loader),leave=None)
    # batch_tqdm_obj.set_description_str("")
    for batch_data in tqdm_obj_batch:
        
        optimizer.zero_grad()
        input_ids = batch_data['input_ids'].to(params['DEVICE'])
        token_type_ids = batch_data['token_type_ids'].to(params['DEVICE'])
        attention_mask = batch_data['attention_mask'].to(params['DEVICE'])
        target = batch_data['target'].to(params['DEVICE'])
        # model takes input of (batch,seq_len)

        logits = model(input_ids,attention_mask,token_type_ids) # (16,450) --> (16,768) --> (16,6)
        b_loss = loss(logits,target)
        b_loss.backward()
        optimizer.step()
        pred = nn.Sigmoid()(logits)
        train_loss.append(b_loss.item())
        all_pred.extend(torch.round(pred).cpu().detach().numpy())
        all_true.extend(target.cpu().detach().numpy())
    acc = accuracy_score(all_true, all_pred)

    f1score = f1_score(all_true,all_pred,average='micro')
    metrics = {"accuracy" : acc,"f1_score":f1score}
    # train_losses.extend(train_loss)
    return train_loss,all_pred,all_true,metrics

def evaluate(model,test_loader,optimizer,params,loss):
    model.eval()
    all_pred = list()
    all_true = list()
    test_loss = list()
    tqdm_obj_batch = tqdm(test_loader,total=len(test_loader),leave=None)

    for batch_data in tqdm_obj_batch:
        input_ids = batch_data['input_ids'].to(params['DEVICE'])
        token_type_ids = batch_data['token_type_ids'].to(params['DEVICE'])
        attention_mask = batch_data['attention_mask'].to(params['DEVICE'])
        target = batch_data['target'].to(params['DEVICE'])

        with torch.no_grad():
            logits = model(input_ids,attention_mask,token_type_ids) # (16,450) --> (16,768) --> (16,6)
            b_loss = loss(logits,target)
            test_loss.append(b_loss.item())
            pred = nn.Sigmoid()(logits)
            all_pred.extend(torch.round(pred).cpu().detach().numpy())
            all_true.extend(target.cpu().detach().numpy())
    acc = accuracy_score(all_true, all_pred)
    f1score = f1_score(all_true,all_pred,average='micro')
    metrics = {"accuracy" : acc,"f1_score":f1score}
    return test_loss,all_pred,all_true,metrics

def CustumLoader(params):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train_df = pd.read_csv(params["TRAIN_DIR"])
    train_dataset,valid_dataset = split_dataset(
        dataframe=train_df
    )
    training_set = CustomDataset(
        train_dataset,
        tokenizer=tokenizer,
        args=params
    )
    validation_set = CustomDataset(
        valid_dataset,
        tokenizer=tokenizer,
        args=params
    )

    train_dataloader = DataLoader(
        training_set,
        batch_size = params["BATCH_SIZE"],
        shuffle = True,
        drop_last = True
        ) # (16tuple_in,1,450)
    validation_dataloader = DataLoader(
        validation_set,
        batch_size = params["BATCH_SIZE"],
        drop_last = True) # # (16,1,450)
    return train_dataloader,validation_dataloader

def main():
    params = {k:v for k,v in config.__dict__.items() if "__" not in k}
    train_dataloader,validation_dataloader = CustumLoader(params)
    model = BertClassifier().to(params['DEVICE'])
    optimizer = Adam(model.parameters(), lr=params["LEARNING_RATE"])
    loss = nn.BCEWithLogitsLoss()
    tqdm_obj_epoch = tqdm(range(params['EPOCHS']),total = params['EPOCHS'],leave = False)
    tqdm_obj_epoch.set_description_str("Epoch")
    val_loss = np.inf
    for epoch in tqdm_obj_epoch:
        train_loss,train_all_pred,train_all_true,train_metrics = train(model,train_dataloader,optimizer,params,loss)
        test_loss,val_all_pred,val_all_true,val_metrics = evaluate(model,validation_dataloader,optimizer,params,loss)
        training_loss = sum(train_loss)/len(train_loss)
        training_accuracy = accuracy_score(train_all_true, train_all_pred)
        validation_loss = sum(test_loss)/len(test_loss)
        validation_accuracy = accuracy_score(val_all_true, val_all_pred)
        
        print(f'\nEpoch: {epoch+1}/{params["EPOCHS"]}')
        print(f"TRAIN : Loss :{training_loss:.4}\tACC :{training_accuracy:.4}\t f1_score :{train_metrics['f1_score']:.4}")
        print(f"VAL   : Loss :{validation_loss:.4},\tACC :{validation_accuracy:.4}\t f1_score :{val_metrics['f1_score']:.4}")
        
        if validation_loss < val_loss:
            val_loss = validation_loss

            early_stopping = 0  
            torch.save(
                    {  
                        "model_state_dict":model.state_dict(),
                        "params":params
                    },params["CHECKPOINT_DIR"])
        else:
            early_stopping += 1
        if early_stopping == params["PATIENCE"]:
            print("Early stopping")
            break


        wandb.log({
                "validation/loss" : validation_loss,
                "validation/error" : 1 - validation_accuracy,
                "validation/accuracy" : validation_accuracy,
                "validation/f1_score" : val_metrics['f1_score'],
                "training/loss" : training_loss,
                "training/error" : 1 - training_accuracy,
                "training/accuracy" : training_accuracy,
                "training/f1_score" : train_metrics['f1_score'],
                    })

if __name__ == "__main__":

    params =  {k:v for k,v in config.__dict__.items() if "__" not in k}
    print(f"Running on : {params['DEVICE']}")    
    print("Params :",params, sep="\n")
    wandb.init(
        project='Multi-label classification',
        entity="ravikumarmn",
        name = params["RUNTIME_NAME"],
        notes = "used base-uncased bert model and add final linear layer.",
        tags = ["bert-base-uncased-"],
        group = "multi-label",
        config=params,
        mode = 'online')
    
    main()
