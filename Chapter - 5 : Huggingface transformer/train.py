import json
import torch
import config
import torch.nn as nn
from tqdm.auto import tqdm
from torch.optim import Adam,SGD
import torch.nn.functional as F
from prepare_data import CustomDataset
from torch.utils.data import Dataset,DataLoader
from model import ClfModel
from helper import Metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import wandb

def train(model, device, train_loader, optimizer):
    model.train()
    all_pred = list()
    all_true = list()
    train_loss = list()
    all_probs = list()
    loss_fn = nn.BCELoss().to(config.device)

    train_obj_epoch = tqdm(train_loader,total = len(train_loader),leave=None)

    for data in train_obj_epoch:
        data, target = data['input'], data['label']
        data = {k:v.to(config.device) for k,v in data.items()}
        target = target.to(config.device)
        optimizer.zero_grad()

        probs = model(data['input_ids'],data['attention_mask']).squeeze()
        loss = loss_fn(probs,target)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.cpu().item())
        pred = (probs.cpu().data>=0.5).int()
        all_probs.extend(probs.data.tolist())

        all_pred.extend(pred.tolist())
        all_true.extend(target.tolist())

    metrics = Metrics(all_true,all_pred,all_probs)
    return train_loss,all_pred,all_true,metrics

def evaluate(model, device, test_loader):
    model.eval()
    all_pred = list()
    all_true = list()
    test_loss = list()
    all_probs = list()
    loss_fn = nn.BCELoss().to(config.device)
    with torch.no_grad():
        test_obj_epoch = tqdm(test_loader,total = len(test_loader),leave=None)
        for data in test_obj_epoch:
            data, target = data['input'], data['label']
            data = {k:v.to(config.device) for k,v in data.items()}
            target = target.to(config.device)
            probs = model(data['input_ids'] ,data['attention_mask']).squeeze()
            loss = loss_fn(probs, target)
            pred = (probs.data>=0.5).float()
            test_loss.append(loss.cpu().item())
            all_probs.extend(probs.cpu().data.tolist())
            all_pred.extend(pred.cpu().int().tolist())
            all_true.extend(target.cpu().int().tolist())
        metrics = Metrics(all_true,all_pred,all_probs)
        return test_loss,all_pred,all_true,metrics

def main():
    args = {k:v for k,v in config.__dict__.items() if "__" not in k}

    vocab = json.load(open(config.base_dir + config.vocab_file_name,'r'))
    word2index = vocab["word2index"]
    args['vocab_len'] = len(word2index)
    
    if config.debug_mode:
        trains = CustomDataset("debug")
        tests = CustomDataset("debug")
    else:
       trains = CustomDataset("train")
       tests = CustomDataset("test")       

    train_dataloader = DataLoader(trains,batch_size = config.BATCH_SIZE,shuffle = True,drop_last = True)
    test_dataloader = DataLoader(tests,batch_size = config.BATCH_SIZE,drop_last = True)
    
    num_dim = vocab['vocab_len']
    model = ClfModel().to(config.device)
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    tqdm_obj_epoch = tqdm(range(config.EPOCHS),total = config.EPOCHS)
    tqdm_obj_epoch.set_description_str("Epoch")
    val_loss = np.inf

    for epoch in tqdm_obj_epoch:
        train_loss,train_all_pred,train_all_true,train_metrics = train(model,config.device,train_dataloader,optimizer)
        training_loss = sum(train_loss)/len(train_loss)
        training_accuracy = train_metrics.compute_accuracy()

        test_loss,test_all_pred,test_all_true,test_metrics = evaluate(model,config.device,test_dataloader)
        validation_loss = sum(test_loss)/len(test_loss)
        validation_accuracy = test_metrics.compute_accuracy()

        confu_matrix = test_metrics.compute_confustion_matrix()
        
        x,y = confu_matrix.shape
        if validation_loss < val_loss:
            val_loss = validation_loss

            early_stopping = 0  
        else:
            early_stopping += 1
        if early_stopping == params["patience"]:
            print(f"Model checkpoints saved to {config.checkpoints_file}")
            print("Early stopping")
            break
        print(f'\nEpoch: {epoch+1}/{params["EPOCHS"]}')
        print(f"TRAIN : Loss :{training_loss:.4}\tACC :{training_accuracy:.4}\t AUC :{train_metrics.compute_auc():.4}")
        print(f"VAL   : Loss :{val_loss:.4},\tACC :{validation_accuracy:.4}\t AUC :{test_metrics.compute_auc():.4}")
        
        if not params['debug_mode']:
            wandb.log({
                "validation/loss" : validation_loss,
                "validation/error" : 1 - validation_accuracy,
                "validation/accuracy" : validation_accuracy,
                
                "validation/recall" : test_metrics.compute_recall(),
                "validation/precision" : test_metrics.compute_precision(),
                "validation/f1_score" : test_metrics.compute_f1_score(),
                "validation/auc_score" : test_metrics.compute_auc(),
                "validation/confu_matrix" : wandb.plot.confusion_matrix(
                    probs=None,
                    preds = test_all_pred,
                    y_true=test_all_true,
                    class_names= list(params['mapping'].keys())),
                "training/loss" : training_loss,
                "training/error" : 1 - training_accuracy,
                "training/accuracy" : training_accuracy,
                "training/recall" : train_metrics.compute_recall(),
                "training/precision" : train_metrics.compute_precision(),
                "training/f1_score" : train_metrics.compute_f1_score(),
                "training/auc_score" : train_metrics.compute_auc(),
                "training/confu_matrix" : wandb.plot.confusion_matrix(
                    probs=None,
                    preds = train_all_pred,
                    y_true=train_all_true,
                    class_names= list(params['mapping'].keys()))
                    })

if __name__ == '__main__':
    params =  {k:v for k,v in config.__dict__.items() if "__" not in k}
    
    print("Params :",params, sep="\n")
    if not params['debug_mode']:
        wandb.init(project='Binary-Classification',
                entity="ravikumarmn",
                name = params["runtime_name"] + f'hidden_{params["HIDDEN_SIZE"]}_embed_{params["EMBED_SIZE"]}',
                notes = "used base-uncased bert model and add final linear layer.",
                tags = ["bert"],
                group = "binary",
                config=params,
                mode = 'online')
    else:
        print(f"DEBUG MODE :{params['debug_mode']}")

    main()