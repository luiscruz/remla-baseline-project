"""
    ML Testing the StackOverflow label predictor for model development. Making use of the [todo library name] library.
"""
import unittest
import joblib
import numpy as np

import libtest.model_development as lib

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression

output_directory = "../output"


def test_against_baseline():
    # Run own model and get score
    X_train, X_val, _ = joblib.load(output_directory + "/X_preprocessed.joblib")
    Y_train, Y_val = joblib.load(output_directory + "/y_preprocessed.joblib")

    (accuracy, f1, avg_precision) = joblib.load(output_directory + "/TFIDF_scores.joblib")
    scores = {"ACC": accuracy, "F1": f1, "AP": avg_precision}

    score_differences = lib.compare_against_baseline(scores, X_train, X_val, Y_train, Y_val, model="linear")

    # Assert every score differs at least 10 percent from the baseline
    for score, diff in score_differences:
        assert (diff > 0.1)


def test_tunable_hyperparameters():
    X_train, X_val, _ = joblib.load(output_directory + "/X_preprocessed.joblib")
    Y_train, Y_val = joblib.load(output_directory + "/y_preprocessed.joblib")
    curr_params = joblib.load(output_directory + "/logistic_regression_params.joblib")
    classifier = joblib.load(output_directory + "/logistic_regression.joblib")
    # classifier_mybag, classifier_tfidf = joblib.load(output_directory + "/classifiers.joblib")

    mlb = MultiLabelBinarizer()
    X_train = mlb.fit_transform(X_train)
    Y_train = mlb.fit_transform(Y_train)

    print(np.shape(X_train))
    print(np.shape(Y_train))

    tunable_parameters = {
        "penalty": ['l1', 'l2', 'elasticnet', 'none'],
        "C": [0.0001, 0.001, 0.1, 1.0, 10.0, 100.0, 1000.0],
    }

    # percentage_mybag, optimal_parameters_mybag = lib.tunable_hyperparameters(LogisticRegression(), tunable_parameters,
    #                                                                          curr_params, X_train, Y_train)
    # print("dissimilar percentage_mybag: " + percentage_mybag + ", current: " + curr_params + ", optimal: "
    #       + optimal_parameters_mybag)

    # percentage_tfidf, optimal_parameters_tfidf = lib.tunable_hyperparameters(classifier, tunable_parameters,
    #                                                                          curr_params, X_train, Y_train)
    # print("dissimilar percentage_tfidf: " + percentage_tfidf + ", current: " + curr_params + ", optimal: "
    #       + optimal_parameters_tfidf)


if __name__ == '__main__':
    # unittest.main()
    test_tunable_hyperparameters()
