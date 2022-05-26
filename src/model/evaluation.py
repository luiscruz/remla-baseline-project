from typing import Dict

import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score

from sklearn.base import BaseEstimator


def print_evaluation_scores(y_val: np.ndarray, predicted: np.ndarray):
    print("Accuracy score: ", accuracy_score(y_val, predicted))
    print("F1 score: ", f1_score(y_val, predicted, average="weighted"))
    print(
        "Average precision score: ",
        average_precision_score(y_val, predicted, average="macro"),
    )


def print_words_for_tag(
    classifier: BaseEstimator,
    tag: str,
    tags_classes: list,
    index_to_words: Dict[int, str],
):
    """
    Print top 5 positive and top 5 negative words for current tag

    Parameters
    ---------
    classifier
            trained classifier
    tag
            a particular tag
    tags_classes
            list of classes names from MultiLabelBinarizer
    index_to_words
            index_to_words transformation
    all_words
            all words in the dictionary
    """

    print("fTag:\t{tag}")

    # Extract an estimator from the classifier for the given tag.
    # Extract feature coefficients from the estimator.

    model = classifier.estimators_[tags_classes.index(tag)]
    top_positive_words = [
        index_to_words[x] for x in model.coef_.argsort().tolist()[0][-5:]
    ]
    top_negative_words = [
        index_to_words[x] for x in model.coef_.argsort().tolist()[0][:5]
    ]

    print(f"Top positive words:\t{', '.join(top_positive_words)}")
    print(f"Top negative words:\t{', '.join(top_negative_words)}\n")
