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
from sklearn.model_selection import GridSearchCV

"""Test that compares the baseline scores to the model scores
    params: 
        scores: dictionary of scores 
            {"ACC": , "AP": , "F1": , "ROC_AUC": } 
            (Accuracy, Average Precision score, F1-score, ROC-AUC score)
        X_train: list of features for training data
        X_test: list of features for testing data
        Y_train: list of outputs for training data
        Y_test: list of outputs for testing data
        model: type of baseline model
            "logistic", default: "linear" 
            
"""


def compare_against_baseline(scores, X_train, X_test, Y_train, Y_test, model="linear"):
    # TRAINS Classifier
    if model == "logistic":
        classifier = LogisticRegression().fit(X_train, Y_train)
    else:  # Default linear model
        classifier = LinearRegression().fit(X_train, Y_train)

    # TESTS Classifier
    y_pred = classifier.predict(X_test)
    score_differences = {}
    covered_scores = scores.keys()
    if "ACC" in covered_scores:
        acc = accuracy_score(Y_test, y_pred)
        score_differences["ACC"] = scores["ACC"] - acc
    if "AP" in covered_scores:
        ap = average_precision_score(Y_test, y_pred)
        score_differences["AP"] = scores["AP"] - ap
    if "F1" in covered_scores:
        f1 = f1_score(Y_test, y_pred)
        score_differences["F1"] = scores["F1"] - f1
    if "ROC_AUC" in covered_scores:
        roc_auc = roc_auc_score(Y_test, y_pred)
        score_differences["ROC_AUC"] = scores["ROC_AUC"] - roc_auc

    # RETURNS score difference
    return score_differences


def tunable_hyperparameters(model, tunable_parameters, curr_parameters, train_X, train_Y):
    """
       Uses grid search to find the optimal (hyper)parameters.
       Takes as input the model, parameters to be tuned, current parameters, training data.
       Returns percentage of non optimal (hyper)parameters and the list of optimal (hyper)parameters.
    """
    grid = GridSearchCV(estimator=model, param_grid=tunable_parameters, n_jobs=-1)

    grid.fit(train_X, train_Y)

    dissimilar = [i for i, j in zip(grid.best_params_, curr_parameters) if i != j]

    return len(dissimilar) / len(curr_parameters), grid.best_params_
