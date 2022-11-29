import config
import pandas as pd
import pickle
import json 

from sklearn.model_selection import train_test_split


def split_dataset_save(dataframe):
    train_dataset,valid_dataset= train_test_split(
        dataframe,
        test_size=0.2,
        shuffle=True
    )
    return train_dataset,valid_dataset

def data_split_save(params):
    data = pd.read_csv(params["TRAIN_DIR"])
    train_dataset,valid_dataset= train_test_split(
        data,
        test_size=0.2,
        shuffle=True
    )

    splitted_data = {
        "train_dataset" : train_dataset,
        "valid_dataset" : valid_dataset
    }
    pickle.dump(splitted_data,open(params["SPLIT_DATA_DIR"],"wb"))
    print("Split data is saved.")


def create_vocabulary(params):
    all_text = list()
    vocabs = dict()
    pairs = pd.read_csv(params["TRAIN_DIR"])['pairs']
    all_text = list()
    for pair in pairs:
        all_text.append(" ".join(eval(pair)))
    all_words = list(set(" ".join(all_text).split()))
    for idx,word in enumerate(all_words):
        vocabs[word] = idx

    json.dump(
        {
            "vocabs" : vocabs,
            "num_words" : len(vocabs)
        },
            open(params["VOCAB_DIR"],"w"))
    print("vocab created.")

def get_parameters(model):
    total_weights = 0
    total_bias =0
    bias_weight = dict()

    for idx,(name,p) in enumerate(model.named_children()):
        count_weight = 0
        count_bias = 0
        for k,v in p.state_dict().items():
            if "weight" in k.lower():
                count_weight += v.numel()
            elif "bias" in k.lower():
                count_bias += v.numel()
        total_weights += count_weight
        total_bias += count_bias
        bias_weight.update({name : {"Weights":f"{count_weight:,}",
                    "Bias":f"{count_bias:,}"}})
    df = pd.DataFrame(bias_weight).T.reset_index(names="Layers")

    return df,{"total_weights":f"{total_weights:,}","total_bias":f"{total_bias:,}"}