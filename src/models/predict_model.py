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

def print_evaluation_scores(y_val, predicted):
    print('Accuracy score: ', accuracy_score(y_val, predicted))
    print('F1 score: ', f1_score(y_val, predicted, average='weighted'))
    print('Average precision score: ', average_precision_score(y_val, predicted, average='macro'))

if __name__ == "__main__":
    # execute only if run as the entry point into the program
    clf = pickle.load(open("./models/tfidf_model.pkl", "rb"))
    X_val_tfidf = sparse.load_npz('./data/processed/X_val.npz')
    y_val = np.array([])
    with open("./data/processed/y_val.npy", "rb") as f:
        y_val = np.load(f)

    y_val_predicted_labels_tfidf = clf.predict(X_val_tfidf)
    y_val_predicted_scores_tfidf = clf.decision_function(X_val_tfidf)
    print_evaluation_scores(y_val, y_val_predicted_labels_tfidf)

    