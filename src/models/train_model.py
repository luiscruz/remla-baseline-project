from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score

import pickle
import pandas as pd

from src.features.messy_code import X_train_tfidf

def print_evaluation_scores(y_val, predicted):
    print('Accuracy score: ', accuracy_score(y_val, predicted))
    print('F1 score: ', f1_score(y_val, predicted, average='weighted'))
    print('Average precision score: ', average_precision_score(y_val, predicted, average='macro'))

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

if __name__ == "__main__":
    # execute only if run as the entry point into the program
    X_train_tfidf = pd.read_csv('../../data/processed/X_train.csv')
    y_train = pd.read_csv('../../data/processed/y_train.csv')

    clf = train_classifier(X_train_tfidf, y_train, penalty='l2', C=10)
    with open("../../models/tfidf_model.pkl", "wb") as f:
        pickle.dump(clf, f)
