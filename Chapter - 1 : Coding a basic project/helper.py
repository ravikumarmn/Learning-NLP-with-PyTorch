import numpy as np
import re
import pandas as pd
from sklearn.metrics import auc,roc_curve
from nltk.stem import 	WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from config import MIN_COUNT,TRIM_SIZE

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