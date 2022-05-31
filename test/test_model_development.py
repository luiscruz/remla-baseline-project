"""
    ML Testing the StackOverflow label predictor for model development. Making use of the [todo library name] library.
"""
import joblib

import libtest.model_development as lib

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
