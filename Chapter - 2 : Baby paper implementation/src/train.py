import json
import torch
import config
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam,SGD
import torch.nn.functional as F
from prepare_data import CustomDataset
from torch.utils.data import Dataset,DataLoader
from model import BCModel
from helper import Metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import wandb
import os

def train(model, device, train_loader, optimizer):
    model.train()
    all_pred = list()
    all_true = list()
    train_loss = list()
    all_probs = list()

    for batch_idx, data in enumerate(train_loader):
        data, target = data['seq_padded'].to(device), data['label'].to(device)
        optimizer.zero_grad()
        probs = model(data).squeeze()
        loss = nn.BCELoss()(probs, target)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        pred = (probs.data>=0.5).int()
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
    with torch.no_grad():
        for data in test_loader:
            data, target = data['seq_padded'].to(device), data['label'].to(device)
            probs = model(data).squeeze()
            
            loss = nn.BCELoss()(probs, target).item()  # sum up batch loss
            pred = (probs.data>=0.5).float()
    
            test_loss.append(loss)
            all_probs.extend(probs.data.tolist())
            all_pred.extend(pred.int().tolist())
            all_true.extend(target.int().tolist())
        metrics = Metrics(all_true,all_pred,all_probs)
        return test_loss,all_pred,all_true,metrics

def main():
    params =  {k:v for k,v in config.__dict__.items() if "__" not in k}
    current_dirs = os.getcwd()
    path = os.path.dirname(current_dirs)+"/Chapter - 1 : Coding a basic project" #this is base for dataset
    params["base_data_dir"] = path
    print("Params :",params, sep="\n")
    if not params['debug_mode']:
        wandb.init(project='Binary-Classification',
                entity="ravikumarmn",
                name = params["runtime_name"] + f'hidden_{params["HIDDEN_SIZE"]}_embed_{params["EMBED_SIZE"]}',
                notes = "",
                tags = ["CNN"],
                group = "binary",
                config=params,
                mode = 'disabled')
    else:
        print(f"DEBUG MODE :{params['debug_mode']}")

    vocab = json.load(open(params["base_data_dir"] + params["vocab_file_name"],'r'))
    word2index = vocab["word2index"]
    params['vocab_len'] = len(word2index)
    
    if params["debug_mode"]:
        trains = CustomDataset("debug")
        tests = CustomDataset("debug")
    else:
        trains = CustomDataset("train")
        tests = CustomDataset("test")       

    train_dataloader = DataLoader(trains,batch_size = params["BATCH_SIZE"],shuffle = True,drop_last = True)
    test_dataloader = DataLoader(tests,batch_size = params["BATCH_SIZE"],drop_last = True)
    
    num_dim = vocab['vocab_len']
    model = BCModel(params,num_dim).to(params["device"])
    optimizer = Adam(model.parameters(), lr=params["LEARNING_RATE"])
    
    tqdm_obj_epoch = tqdm(range(params["EPOCHS"]),total = params["EPOCHS"],leave = False)
    tqdm_obj_epoch.set_description_str("Epoch")
    val_loss = np.inf

    for epoch in tqdm_obj_epoch:
        # train_loss,all_pred,all_true,metrics = train(model, params["device"], train_dataloader, optimizer, epoch)
        # test(model, params["device"], test_dataloader)
        train_loss,train_all_pred,train_all_true,train_metrics = train(model,params["device"],train_dataloader,optimizer)
        training_loss = sum(train_loss)/len(train_loss)
        training_accuracy = train_metrics.compute_accuracy()

        test_loss,test_all_pred,test_all_true,test_metrics = evaluate(model,params["device"],test_dataloader)
        validation_loss = sum(test_loss)/len(test_loss)
        validation_accuracy = test_metrics.compute_accuracy()

        confu_matrix = test_metrics.compute_confustion_matrix()
        
        x,y = confu_matrix.shape
        if validation_loss < val_loss:
            val_loss = validation_loss

            early_stopping = 0  
            if not params["debug_mode"]:
                torch.save(
                    {  
                        "model_state_dict":model.state_dict(),
                        "params":params
                    },params["checkpoints_file"])
        else:
            early_stopping += 1
        if early_stopping == params["patience"]:
            #Learning-NLP-with-PyTorch/Chapter - 1 : Coding a basic project/checkpoints
            print(f'Model checkpoints saved to {params["checkpoints_file"]}')
            df_cm = pd.DataFrame(confu_matrix, range(x), range(y))
            # df_norm_col=(df_cm-df_cm.mean())/df_cm.std()
            ax = plt.axes()
            sn.set(font_scale=1.4) # for label size
            
            sn.heatmap(df_cm, annot=True, annot_kws={"size": 16},) # font size
            ax.set_title('Confusion matrix of the classifier')
            # plt.savefig('Chapter - 1 _ Coding a basic project/confusion_matrix1.jpg')
            print("Early stopping")
            break
            

        # print(f'Epoch: {epoch+1}/{params["EPOCHS"]}\
        #     Train loss: {training_loss}\
        #         Train acc: {training_accuracy}\
        #             Val loss:{validation_loss}\
        #                 Val acc:{validation_accuracy}\
        #                     Recall: {test_metrics.compute_recall()}\
        #                         Precision : {test_metrics.compute_precision()}\
        #                             f1_score : {test_metrics.compute_f1_score()}\
        #                                 auc_score : {test_metrics.compute_auc()}')
        # training_accuracy = accuracy_score(train_all_true,train_all_pred)
        # validation_accuracy = accuracy_score(test_all_true,test_all_pred)
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


    main()