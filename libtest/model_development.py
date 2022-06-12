"""
    ML Test library functions for model development.
    Based on section 3 of the paper referenced below.

    Eric Breck, Shanqing Cai, Eric Nielsen, Michael Salib, D. Sculley (2016). What’s your ML test score? A rubric for ML production systems. Reliable Machine Learning in the Wild - NIPS 2016 Workshop (2016).
    Available: https://storage.googleapis.com/pub-tools-public-publication-data/pdf/45742.pdf
"""
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier


def compare_against_classification_baseline(scores, X_train, X_test, Y_train, Y_test, model="linear"):
    """
     Compares the performance of a classifier against a baseline classifier model.
    :param scores: dictionary of scores
            {"ACC": , "AP": , "F1": , "ROC_AUC": }
            (Accuracy, Average Precision score, F1-score, ROC-AUC score)
    :param X_train: list of features for training data
    :param X_test: list of features for testing data
    :param Y_train: list of outputs for training data
    :param Y_test: list of outputs for testing data
    :param model: type of classifier model, "logistic", default: "linear"
    :return: score differences between model and baseline model
    """
    # TRAIN Classifier
    if model == "logistic":
        classifier = LogisticRegression().fit(X_train, Y_train)
    elif model == "linear":  # Default linear model
        classifier = LinearRegression()
    else:
        classifier = DummyClassifier()
    classifier = OneVsRestClassifier(classifier)
    classifier.fit(X_train, Y_train)

    # TEST Classifier
    y_pred = classifier.predict(X_test)

    # Calculate and compare scores
    baseline_scores = {}
    score_differences = {}
    covered_scores = scores.keys()
    if "ACC" in covered_scores:
        acc = accuracy_score(Y_test, y_pred)
        baseline_scores["ACC"] = acc
        score_differences["ACC"] = scores["ACC"] - acc
    if "AP" in covered_scores:
        ap = average_precision_score(Y_test, y_pred)
        baseline_scores["AP"] = ap
        score_differences["AP"] = scores["AP"] - ap
    if "F1" in covered_scores:
        f1 = f1_score(Y_test, y_pred, average='weighted')  # average needs to be set
        baseline_scores["F1"] = f1
        score_differences["F1"] = scores["F1"] - f1

    # RETURNS score difference
    return baseline_scores, score_differences


def tunable_hyperparameters(model, tunable_parameters, curr_parameters, X_train, Y_train):
    """
       Uses grid search to find the optimal (hyper)parameters.
    :param model: the classification model
    :param tunable_parameters: parameters to be tuned
    :param curr_parameters: current parameters used
    :param X_train: list of features for training data
    :param Y_train: list of outputs for training data
    :return: Returns percentage of non optimal (hyper)parameters and the list of optimal (hyper)parameters.
    """
    grid = GridSearchCV(estimator=model, param_grid=tunable_parameters)
    print(grid)
    grid.fit(X_train, Y_train)

    dissimilar = [i for i, j in zip(grid.best_params_, curr_parameters) if i != j]

    return len(dissimilar) / len(curr_parameters), grid.best_params_


def data_slices(model, slices, X_val, Y_val):
    """
        Runs the given model the data slices and compares the difference in score in all slices.
    :param model: the classification model
    :param slices: dictionary with slices
        key: category of slicing (eg. length size)
        value: list of lists [[x_slice], y_slice]]
    :param Y_train_slices: array with slices of Y_train data
    :param X_val: X validation data
    :param Y_val: Y validation data
    :return: the difference between min and max score over all slices
    """
    min = 100
    max = 0

    for key in slices.keys():
        x_slice = slices[key][0]
        y_slice = slices[key][1]
        model.fit(x_slice, y_slice)
        score = model.score(X_val, Y_val)
        if score < min:
            min = score
        if score > max:
            max = score

    assert max - min < 0.15


def model_staleness(new_model_metrics, old_model_metrics):
    """
        Compares the metrics of the old model to the metrics of a new model
    :param new_model_metrics: dictionary of scores for the new model (Accuracy, Average Precision score, F1-score, ROC-AUC score)
        {"ACC": , "AP": , "F1": , "ROC_AUC": }
    :param old_model_metrics: dictionary of scores for the old model (Accuracy, Average Precision score, F1-score, ROC-AUC score)
        {"ACC": , "AP": , "F1": , "ROC_AUC": }
    :return dictionary of differences in the scores of the old model and new model
    """

    score_differences = {}
    old_metrics = old_model_metrics.keys()
    new_metrics = new_model_metrics.keys()
    if "ACC" in old_metrics & "ACC" in new_metrics:
        # acc = accuracy_score(y_test, y_pred)
        score_differences["ACC"] = old_model_metrics["ACC"] - new_model_metrics["ACC"]
    if "AP" in old_metrics & "AP" in new_metrics:
        # aps = average_precision_score(y_test, y_pred)
        score_differences["AP"] = old_model_metrics["AP"] - new_model_metrics["AP"]
    if "F1" in old_metrics & "F1" in new_metrics:
        # f1 = f1_score(y_test, y_pred)
        score_differences["F1"] = old_model_metrics["F1"] - new_model_metrics["F1"]
    if "ROC_AUC" in old_metrics & "ROC_AUC" in new_metrics:
        # auc_roc = roc_auc_score(y_test, y_pred)
        score_differences["ROC_AUC"] = old_model_metrics["ROC_AUC"] - new_model_metrics["ROC_AUC"]

    # return score_differences
    for metric, score in score_differences.items():
        assert score < 0.1, f"difference less than 0.1 expected, got: {score} " f"for {metric}"
