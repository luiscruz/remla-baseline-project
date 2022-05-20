"""Module for calculating model evaluation scores."""
import pickle
import sys

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score
)

from src.config.definitions import ROOT_DIR


def print_evaluation_scores(y_val_, predicted):
    print('Accuracy score: ', accuracy_score(y_val_, predicted))
    print('F1 score: ', f1_score(y_val_, predicted, average='weighted'))
    print('Average precision score: ', average_precision_score(y_val_, predicted, average='macro'))
    print('ROC AUC score: ', roc_auc_score(y_val_, predicted, multi_class='ovo'))


if __name__ == '__main__':

    # load model
    if len(sys.argv) == 2:
        MODEL_NAME = sys.argv[1]
    else:
        MODEL_NAME = 'tfidf.pkl' # default
    with open(ROOT_DIR / 'models' / MODEL_NAME, 'rb') as f:
        classifier = pickle.load(f)

    # load data
    with open(ROOT_DIR / 'data/processed/validation.pkl', 'rb') as f:
        X_val_tfidf, y_val = pickle.load(f)

    pred = classifier.predict(X_val_tfidf)

    with open(ROOT_DIR / 'models/mlb.pkl', 'rb') as f:
        mlb = pickle.load(f)
    # (seq of seqs -> binary array)
    y_val = mlb.fit_transform(y_val)

    print_evaluation_scores(y_val, pred)
