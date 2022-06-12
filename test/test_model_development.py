"""
    ML Testing the StackOverflow label predictor for model development. Making use of the mltest library.
"""
import math
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

import libtest.model_development as lib

output_directory = "../output"


def test_tfidf_against_baseline():
    # Run own model and get score
    _, X_train_tfidf, _, X_val_tfidf = joblib.load(output_directory + "/vectorized_x.joblib")
    Y_train, Y_val = joblib.load(output_directory + "/fitted_y.joblib")

    (accuracy, f1, avg_precision) = joblib.load(output_directory + "/TFIDF_scores.joblib")
    scores = {"ACC": accuracy, "F1": f1, "AP": avg_precision}

    baseline_scores, score_differences = lib.compare_against_classification_baseline(scores, X_train_tfidf, X_val_tfidf,
                                                                                     Y_train, Y_val, model="linear")

    # Assert every score differs at least 10 percent from the baseline
    for score, diff in score_differences.items():
        # print(score, " score difference: ", diff)
        assert (diff > 0.1)


def test_tunable_hyperparameters():
    X_train, X_val, _ = joblib.load(output_directory + "/X_preprocessed.joblib")
    Y_train, Y_val = joblib.load(output_directory + "/y_preprocessed.joblib")
    curr_params = joblib.load(output_directory + "/logistic_regression_params.joblib")
    classifier = joblib.load(output_directory + "/logistic_regression.joblib")
    classifier_mybag, classifier_tfidf = joblib.load(output_directory + "/classifiers.joblib")

    mlb = MultiLabelBinarizer()
    X_train = mlb.fit_transform(X_train)
    Y_train = mlb.fit_transform(Y_train)

    tunable_parameters = {
        "estimator__penalty": ['l1', 'l2'],
        "estimator__C": [0.1, 1.0],
    }

    percentage_mybag, optimal_parameters_mybag = lib.tunable_hyperparameters(classifier_mybag, tunable_parameters,
                                                                             curr_params, X_train, Y_train)
    print("dissimilar percentage_mybag: " + percentage_mybag + ", current: " + curr_params + ", optimal: "
          + optimal_parameters_mybag)

    percentage_tfidf, optimal_parameters_tfidf = lib.tunable_hyperparameters(classifier_tfidf, tunable_parameters,
                                                                             curr_params, X_train, Y_train)
    print("dissimilar percentage_tfidf: " + percentage_tfidf + ", current: " + curr_params + ", optimal: "
          + optimal_parameters_tfidf)


def test_data_slicing():
    X_train, _, _ = joblib.load(output_directory + "/X_preprocessed.joblib")
    Y_train, Y_val = joblib.load(output_directory + "/y_preprocessed.joblib")


    X_train_mybag, _, X_val_mybag, _ = joblib.load(output_directory + "/vectorized_x.joblib")
    length = (len(x.split()) for x in X_train)
    tuples = list((x, y, z) for x, y, z in zip(X_train_mybag, Y_train, length))
    tuples.sort(key=lambda y: y[2])



    slices = {}
    for t in tuples:
        slice = t[2]%5
        if slice not in slices.keys():
            slices[slice] = []
        slices[slice].append((t[0], t[1]))

    model = OneVsRestClassifier(LogisticRegression(penalty='l1', C=1, dual=False, solver='liblinear'))
    tags_counts = joblib.load(output_directory + "/tags_counts.joblib")
    mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
    Y_val = mlb.fit_transform(Y_val)


    for key in slices.keys():
        x_slice = []
        for x in slices[key]:
            x_slice.append(x[0].toarray())
        x_slice = np.stack(x_slice, axis=0)

        y_slice = []
        for y in slices[key]:
            y_slice.append(y[1])
        y_slice = mlb.fit_transform(y_slice)

        nsamples, nx, ny = x_slice.shape
        x_slice = x_slice.reshape((nsamples, nx * ny))

        slices[key] = [x_slice, y_slice]

    lib.data_slices(model, slices, X_val_mybag, Y_val)



def test_model_staleness():
    # X_train, X_val, _ = joblib.load(output_directory + "/X_preprocessed.joblib")
    X_train_mybag, _, X_val_mybag, _ = joblib.load(output_directory + "/vectorized_x.joblib")
    Y_train, Y_val = joblib.load(output_directory + "/y_preprocessed.joblib")

    tags_counts = joblib.load(output_directory + "/tags_counts.joblib")
    mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
    Y_train = mlb.fit_transform(Y_train)
    Y_val = mlb.fit_transform(Y_val)

    three_fourth_x_train = math.floor(X_train_mybag.shape[0] * 0.75)
    three_fourth_y_train = math.floor(len(Y_train) * 0.75)
    old_train_x = X_train_mybag[:three_fourth_x_train]
    # new_train_x = X_train_mybag[three_fourth_x_train:]
    old_train_y = Y_train[:three_fourth_y_train]
    # new_train_y = Y_train[three_fourth_y_train:]

    three_fourth_x_test = math.floor(X_val_mybag.shape[0] * 0.75)
    three_fourth_y_test = math.floor(len(Y_val) * 0.75)
    old_test_x = X_val_mybag[:three_fourth_x_test]
    # new_test_x = X_val_mybag[three_fourth_x_test:]
    old_test_y = Y_val[:three_fourth_y_test]
    # new_test_y = Y_val[three_fourth_y_test:]

    #     Train model for old set
    model = OneVsRestClassifier(LogisticRegression())
    tunable_parameters = {
        "estimator__penalty": [None, 'l2'],
        "estimator__C": [0.01, 0.1, 1.0],
    }
    grid_old = GridSearchCV(estimator=model, param_grid=tunable_parameters)
    grid_old.fit(old_train_x, old_train_y)
    y_pred_old = grid_old.predict(old_test_x)

    old_model_metrics = {}
    f1_old = f1_score(old_test_y, y_pred_old, average='samples')
    old_model_metrics["F1"] = f1_old
    acc = accuracy_score(old_test_y, y_pred_old)
    old_model_metrics["ACC"] = acc
    roc_auc = roc_auc_score(old_test_y, y_pred_old)
    old_model_metrics["ROC_AUC"] = roc_auc
    aps = average_precision_score(old_test_y, y_pred_old)
    old_model_metrics["AP"] = aps

    # Train model for new set
    grid_new = GridSearchCV(estimator=model, param_grid=tunable_parameters)
    grid_new.fit(X_train_mybag, Y_train)
    y_pred_new = grid_new.predict(X_val_mybag)

    new_model_metrics = {}
    f1_new = f1_score(Y_val, y_pred_new, average='samples')
    new_model_metrics["F1"] = f1_new
    acc_new = accuracy_score(Y_val, y_pred_new)
    new_model_metrics["ACC"] = acc_new
    roc_auc_new = roc_auc_score(Y_val, y_pred_new)
    new_model_metrics["ROC_AUC"] = roc_auc_new
    aps_new = average_precision_score(Y_val, y_pred_new)
    new_model_metrics["AP"] = aps_new

    # get metrics for both sets


    lib.model_staleness(new_model_metrics, old_model_metrics)


if __name__ == '__main__':
    # test_tunable_hyperparameters()
    test_data_slicing()
