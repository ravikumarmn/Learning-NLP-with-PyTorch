import config
import torch
import wandb
import pickle
import numpy as np
import pandas as pd
from helper import data_split_save,create_vocabulary,get_parameters
from custom_data import CustomDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.metrics import f1_score,accuracy_score
from torch import nn
from tqdm.auto import tqdm
from model import ClassifierModel
from tokenizer import Tokenizer
import os
import matplotlib.pyplot as plt

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
        target = batch_data['target'].to(params['DEVICE'])
        # model takes input of (batch,seq_len)

        logits = model(input_ids) # (16,450) --> (16,768) --> (16,6)
        b_loss = loss(logits,target)
        b_loss.backward()
        optimizer.step()
        pred = nn.Sigmoid()(logits)
        train_loss.append(b_loss.item())
        all_pred.extend(torch.round(pred).cpu().detach().tolist())
        all_true.extend(target.cpu().detach().tolist())
    acc = accuracy_score(all_true, all_pred)

    f1score = f1_score(all_true,all_pred,average='micro')
    metrics = {"accuracy" : acc,"f1_score":f1score}
    # train_losses.extend(train_loss)
    return train_loss,all_pred,all_true,metrics

def evaluate(model,test_loader,params,loss):
    model.eval()
    all_pred = list()
    all_true = list()
    test_loss = list()
    tqdm_obj_batch = tqdm(test_loader,total=len(test_loader),leave=None)

    for batch_data in tqdm_obj_batch:
        input_ids = batch_data['input_ids'].to(params['DEVICE'])
        target = batch_data['target'].to(params['DEVICE'])

        with torch.no_grad():
            logits = model(input_ids) # (16,450) --> (16,768) --> (16,6)
            b_loss = loss(logits,target)
            test_loss.append(b_loss.item())
            pred = torch.sigmoid(logits)
            all_pred.extend(torch.round(pred).tolist())
            all_true.extend(target.tolist())
    acc = accuracy_score(all_true, all_pred)
    f1score = f1_score(all_true,all_pred,average='micro')
    metrics = {"accuracy" : acc,"f1_score":f1score}
    return test_loss,all_pred,all_true,metrics

def CustumLoader(params):
    tokenizer_obj = Tokenizer("topic-modelling-research-articles")
    splitted_data = pickle.load(open(params["SPLIT_DATA_DIR"],"rb"))
    valid_dataset = splitted_data["valid_dataset"][params["LABELS"] +["pairs"]]
    train_dataset = splitted_data["train_dataset"][params["LABELS"] +["pairs"]]


    training_set = CustomDataset(
        train_dataset,
        tokenizer=tokenizer_obj,
        args=params,
    )
    validation_set = CustomDataset(
        valid_dataset,
        tokenizer=tokenizer_obj,
        args=params,
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

    params["N_WORDS"] = len(training_set.tokenizer.word_to_token)
    return train_dataloader,validation_dataloader

def main():
    params = {k:v for k,v in config.__dict__.items() if "__" not in k}

    train_dataloader,validation_dataloader = CustumLoader(params)

    model = ClassifierModel(params).to(params['DEVICE'])
    df,n_params = get_parameters(model)
    tbl = wandb.Table(data=df)
    wandb.log({"Parameters_size":tbl})
    print(n_params)
    
    optimizer = Adam(model.parameters(), lr=params["LEARNING_RATE"])
    pos_weights = torch.from_numpy(train_dataloader.dataset.pos_weight).to(params["DEVICE"])

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights,reduction="mean")
    tqdm_obj_epoch = tqdm(range(params['EPOCHS']),total = params['EPOCHS'],leave = False)
    tqdm_obj_epoch.set_description_str("Epoch")
    val_loss = np.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=params["STEP_SIZE"],gamma=params["GAMMA"])
    for epoch in tqdm_obj_epoch:
        train_loss,train_all_pred,train_all_true,train_metrics = train(model,train_dataloader,optimizer,params,loss_fn)
        scheduler.step()
        test_loss,val_all_pred,val_all_true,val_metrics = evaluate(model,validation_dataloader,params,loss_fn)
        training_loss = sum(train_loss)/len(train_loss)
        training_accuracy = accuracy_score(torch.tensor(train_all_true), torch.tensor(train_all_pred))
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
                    },params["CHECKPOINT_NAME"])
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
                "learning_rate" : scheduler.get_last_lr()[-1],
                    })

if __name__ == "__main__":

    params =  {k:v for k,v in config.__dict__.items() if "__" not in k}
    print(f"Running on : {params['DEVICE']}")    
    print("Params :",params, sep="\n")
    if not os.path.isfile(params["SPLIT_DATA_DIR"]):
        data_split_save(params)
    if not os.path.isfile(params["VOCAB_DIR"]):
        create_vocabulary(params)
    wandb.init(
        project='Multi-label classification',
        entity="ravikumarmn",
        name = params["RUNTIME_NAME"],
        notes = "used transformer",
        tags = ["transformer"],
        group = "multi-label",
        config=params,
        mode = 'online')
    
    main()
