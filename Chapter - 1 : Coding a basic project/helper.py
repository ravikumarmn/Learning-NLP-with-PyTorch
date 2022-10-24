import re
import gensim
import config
import pickle
import numpy as np
import pandas as pd
from config import MIN_COUNT,TRIM_SIZE
from gensim.models import KeyedVectors
from nltk.stem import 	WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from sklearn.metrics import auc,roc_curve
import json 

def read_data_file(base_dir,file_name):
    dataset = pd.read_csv(base_dir+file_name)
    return dataset

def clean_text(text):
    # Remove puncuation,stopwords,only words,lowercase,lematization
    text = re.sub(r'(\#\w+)'," ",text)
    text = re.sub(r"br","",text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub('[^a-zA-Z]+',' ',text)
    text = " ".join([wordnet_lemmatizer.lemmatize(x) for x in text.split() ])
    text = text.lower()
    return text.strip()

def trim_sentence(sentences):
    trim = list()
    for sentence in sentences:
        sentence_len = len(sentence.split())
        if sentence_len >= MIN_COUNT:
            trim.append(" ".join(sentence.split()[:TRIM_SIZE]))
        else:
            trim.append(sentence)
    return trim

class Metrics:
    epsilon = 1e-7
    def __init__(self,y_true:list,y_pred:list,y_probs:list = None):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_probs = y_probs
        assert len(y_true)==len(y_pred), f"Length target {len(y_true)} did \
                                            not match predictions {len(y_pred)}"

    def compute_tp_tn_fp_fn(self) -> float:
        """
        True positive  - actual = 1, predicted = 1
        False positive - actual = 1, predicted = 0
        False negative - actual = 0, predicted = 1
        True negative  - actual = 0, predicted = 0
        """
        tp = sum((self.y_true == 1) & (self.y_pred == 1))
        tn = sum((self.y_true == 0) & (self.y_pred == 0))
        fp = sum((self.y_true == 1) & (self.y_pred == 0))
        fn = sum((self.y_true == 0) & (self.y_pred == 1))

        return (tp,tn,fp,fn)

    def compute_accuracy(self) -> float:
        """
        Accuracy  = TP + TN / FP + FN + TP + TN

        """
        accuracy_score = (self.y_true==self.y_pred).mean()
        return float(accuracy_score)

    def compute_precision(self) -> float:
        """
        Precision = TP / TP + FP
        
        """
        tp,tn,fp,_ = self.compute_tp_tn_fp_fn()
        precision_score = tp/(tp + fp + self.epsilon)

        return float(precision_score)

    def compute_recall(self) -> float:
        """
        Recall = TP / TP + FN

        """
        tp,tn,_,fn = self.compute_tp_tn_fp_fn()
        recall_score = tp /(tp + fn + self.epsilon)

        return float(recall_score)

    def compute_f1_score(self) -> float:
        """
        F1-Score = (2*precision * recall)/(precision + recall)

        """
        precision = self.compute_precision()
        recall = self.compute_recall()
        f1_score = (2 * precision * recall)/(precision + recall + self.epsilon)
        
        return f1_score

    def compute_auc(self) -> float:
        if self.y_probs != None:
            fpr, tpr, thresholds = roc_curve(self.y_true, self.y_probs)
            auc_score = auc(fpr, tpr)
            return auc_score
        else:
            
            print("y_probs are not defined")
            return 0


    def compute_confustion_matrix(self) -> list:
        tp,tn,fp,fn = self.compute_tp_tn_fp_fn()
        confusion_mat = [[tp,fp],[fn,tn]]
        return np.array(confusion_mat)

    def metrics_report(self) -> dict:
        """
        Gives report, containing all the metrics outputs.
        """
        results = {}
        function_name = [x for x in dir(self) if "__" not in x and (x!= 'metrics_report')]
        for x in function_name:
            if callable(getattr(self,x)):
              results.update({x:getattr(self,x)()})
        return results


def build_word2vec(all_sentence,embedding_size):
    w2v_model = gensim.models.Word2Vec(sentences=all_sentence,min_count=1,vector_size= embedding_size)
    w2v_model.build_vocab(all_sentence)
    print("Length of samples : ",w2v_model.corpus_count)
    print("Length of vocab   : ",len(w2v_model.wv.key_to_index))
    print("Training and saving model...")
    w2v_model.train(all_sentence,total_examples=w2v_model.corpus_count,epochs=w2v_model.epochs)

    w2v_model.save(f'dataset/prep_word2vectors_{config.EMBED_SIZE}.embedding')
    wordvecs = gensim.models.Word2Vec.load(f'dataset/prep_word2vectors_{config.EMBED_SIZE}.embedding')
    local_vocab = json.load(open(config.vocab_file_name,"r"))
    word2index = local_vocab['word2index']

    matrix_vec = np.zeros((len(word2index),config.EMBED_SIZE))
    for word,idx in word2index.items():
        try:
            vector_x = wordvecs.wv[word]
            matrix_vec[idx,:] = vector_x 
        except KeyError:
            pass


    pickle_data = {"embedding_vector" : matrix_vec,
                    'vocab_len' : len(word2index)
                    }
    pickle.dump(pickle_data,open(config.emb_vec_file,'wb'))
    print("Done")
    print(f'Model saved to {config.base_dir+config.emb_vec_file}')

def words_sentence(dataframe):
    all_words = set()
    all_sentence = list()
    for sentence in dataframe["trimmed_review"]:
        all_sentence.append(sentence.lower().split())
        for word in sentence.lower().split():
            all_words.add(word)
    len_all_words = len(all_words)
    len_all_sentence = len(all_sentence)
    print(f"Total number of words : {len_all_words}")
    print(f"Total number of sentence : {len_all_sentence}")
    return all_words,all_sentence