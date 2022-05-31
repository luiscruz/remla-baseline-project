"""
    ML Test library functions for model development.
    Based on section 3 of the paper referenced below.

    Eric Breck, Shanqing Cai, Eric Nielsen, Michael Salib, D. Sculley (2016). What’s your ML test score? A rubric for ML production systems. Reliable Machine Learning in the Wild - NIPS 2016 Workshop (2016).
    Available: https://storage.googleapis.com/pub-tools-public-publication-data/pdf/45742.pdf
"""
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


def compare_against_baseline(own_score, X_train, X_val, Y_train, Y_val, model="linear"):
    options = ["linear", "logistic"]
    if model not in options:
        model = "linear"

    # TRAINS Classifier
    if model == "linear":
        classifier = LinearRegression().fit(X_train, Y_train)
    if model == "logistic":
        classifier = LogisticRegression().fit(X_train, Y_train)

    # TESTS Classifier & RETURNS score difference
    baseline_score = classifier.score(X_val, Y_val)
    return own_score - baseline_score
