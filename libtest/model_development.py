"""
    ML Test library functions for model development.
    Based on section 3 of the paper referenced below.

    Eric Breck, Shanqing Cai, Eric Nielsen, Michael Salib, D. Sculley (2016). What’s your ML test score? A rubric for ML production systems. Reliable Machine Learning in the Wild - NIPS 2016 Workshop (2016).
    Available: https://storage.googleapis.com/pub-tools-public-publication-data/pdf/45742.pdf
"""
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

"""Test that compares the baseline scores to the model scores
    params: 
        scores: dictionary of scores 
            {"Acc": , "APC": , "F1": , "ROC_AUC": } 
            (Accuracy, Average Precision Score, F1-score, ROC-AUC score)
        X_train: list of features for training data
        X_test: list of features for testing data
        Y_train: list of outputs for training data
        Y_test: list of outputs for testing data
        model: type of baseline model
            "logistic", default: "linear" 
            
"""
def compare_against_baseline(scores, X_train, X_test, Y_train, Y_test, model="linear"):
    options = ["linear", "logistic"]
    if model not in options:
        model = "linear"

    # TRAINS Classifier
    if model == "linear":
        classifier = LinearRegression().fit(X_train, Y_train)
    if model == "logistic":
        classifier = LogisticRegression().fit(X_train, Y_train)

    # TESTS Classifier & RETURNS score difference
    y_pred = classifier.predict(X_test)
    to_return = {}
    if "Acc" in scores.key():
        accuracy = accuracy_score(Y_test, y_pred)
        to_return["Acc"] = scores["Acc"] - accuracy
    if "APC" in scores.key():
        aps = average_precision_score(Y_test, y_pred)
        to_return["APC"] = scores["APC"] - aps
    if "F1" in scores.key():
        f1_score = f1_score(Y_test, y_pred)
        to_return["F1"] = scores["F1"] - f1_score
    if "ROC_AUC" in scores.key():
        roc_auc = roc_auc_score(Y_test, y_pred)
        to_return["ROC_AUC"] = scores["ROC_AUC"] - roc_auc
    return to_return
