from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from scipy import sparse

import pickle
import pandas as pd
import numpy as np

def train_classifier(X_train, y_train, penalty='l1', C=1):
    """
      X_train, y_train — training data
      
      return: trained classifier
    """
    # Create and fit LogisticRegression wraped into OneVsRestClassifier.
    
    clf = LogisticRegression(penalty=penalty, C=C, dual=False, solver='liblinear')
    clf = OneVsRestClassifier(clf)
    clf.fit(X_train, y_train)
    
    return clf  

def start_training():
    # execute only if run as the entry point into the program
    X_train_tfidf = sparse.load_npz('./data/processed/X_train.npz')
    y_train = np.array([])
    with open("./data/processed/y_train.npy", "rb") as f:
        y_train = np.load(f)

    clf = train_classifier(X_train_tfidf, y_train, penalty='l2', C=10)
    with open("./models/tfidf_model.pkl", "wb") as f:
        pickle.dump(clf, f)

if __name__ == "__main__":
    start_training()
