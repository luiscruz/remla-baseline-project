"""Module used to train the model in the ML pipeline"""
import pickle

import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

with open("params.yaml", encoding="utf-8") as f:
    params = yaml.safe_load(f)
featurize_params = params["featurize"]
train_params = params["train"]

INPUT_TRAIN_PATH = featurize_params["output_train"]
OUT_PATH_MODEL = train_params["model_out"]

"""
MultiLabel classifier
In this task each example can have multiple tags therefore transform labels in a binary form
and the prediction will be a mask of 0s and 1s.
For this purpose it is convenient to use MultiLabelBinarizer from sklearn.
"""


def train_classifier(X_train, y_train, penalty="l1", C=1):
    """
    X_train, y_train — training data

    return: trained classifier
    """

    # Create and fit LogisticRegression wrapped into OneVsRestClassifier.
    clf = LogisticRegression(penalty=penalty, C=C, dual=False, solver="liblinear")
    clf = OneVsRestClassifier(clf, verbose=1)
    clf.fit(X_train, y_train)

    return clf


def train_classifier_for_transformations(X_train_tfidf, y_train):
    """
    :param X_train_tfidf:
    :param y_train:
    :return: Trained classifier for tf-idf
    """
    classifier_tfidf = train_classifier(X_train_tfidf, y_train)
    return classifier_tfidf


def pickle_model(clf):
    """Save model to disk using pickle"""
    with open(OUT_PATH_MODEL, "wb") as fd:
        pickle.dump(clf, fd, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickled_train_data():
    """Load train data from disk using pickle"""
    with open(INPUT_TRAIN_PATH, "rb") as fd:
        X_train, y_train = pickle.load(fd)
    return X_train, y_train


def main():
    """Run train model steps"""
    X_train, y_train = load_pickled_train_data()
    clf = train_classifier_for_transformations(X_train, y_train)

    pickle_model(clf)


if __name__ == "__main__":
    main()
