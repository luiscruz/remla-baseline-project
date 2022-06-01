"""
    ML Testing the StackOverflow label predictor for the ML infrastructure. Making use of the [todo library name] library.
"""
from math import isclose

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from libtest.ml_infrastructure import test_reproducibility_training
from model_training import train_classifier


def test_reproducibility_training_specific():
    print("Reproducibility Training example")
    X_train_mybag, X_train_tfidf, X_val_mybag, X_val_tfidf = joblib.load("../output/vectorized_x.joblib")
    y_train, y_val = joblib.load("../output/y_preprocessed.joblib")
    tags_counts = joblib.load("../output/tags_counts.joblib")

    mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
    y_train = mlb.fit_transform(y_train)
    y_val = mlb.fit_transform(y_val)

    clf = LogisticRegression(penalty='l1', C=1, dual=False, solver='liblinear')
    clf = OneVsRestClassifier(clf)

    test_reproducibility_training(clf, X_train_mybag, y_train, X_val_mybag, y_val)

# todo add more
