"""
    ML Test library functions for the ML infrastructure.
    Based on section 4 of the paper referenced below.

    Eric Breck, Shanqing Cai, Eric Nielsen, Michael Salib, D. Sculley (2016). What’s your ML test score? A rubric for
    ML production systems. Reliable Machine Learning in the Wild - NIPS 2016 Workshop (2016). Available:
    https://storage.googleapis.com/pub-tools-public-publication-data/pdf/45742.pdf """
from math import isclose

from sklearn.metrics import f1_score, average_precision_score, accuracy_score


def reproducibility_training(model, x_train, y_train, x_test, y_test, eval_metrics=None, precision=1e-3):
    """
    Train two models on the same data, and observe any differences in aggregate metrics, sliced metrics,
    or example-by-example predictions. Large differences due to non-determinism can exacerbate debugging and
    troubleshooting.
    :param model: dictionary with models to be trained
    :param x_train: list of features for training data
    :param y_train: list of outputs for training data
    :param x_test: list of features for testing data
    :param y_test: list of outputs for testing data
    :param eval_metrics:
    :param precision: precision value, default is 1e-3
    :return: boolean array where two arrays are element-wise equal within given precision value.
    """
    models = {}
    for i in range(2):
        trained_model = model.fit(x_train, y_train)

        models["model_" + str(i)] = get_accuracy(trained_model, x_test, y_test)

    lst = all(isclose(x, y, rel_tol=precision) for x, y in zip(models["model_0"], models["model_1"]))
    print(lst, models)
    assert lst


def improved_model_quality(model_1, model_2, x_test, y_test):
    """
    Useful tests include testing against data with known correct outputs and validating the aggregate quality,
    as well as comparing predictions to a previous version of the model.
    :param model_1: first version of the model
    :param model_2: second version of the model
    :param x_test: list of features for testing data
    :param y_test: list of outputs for testing data
    :return:
    """
    res1 = get_accuracy(model_1, x_test, y_test)
    res2 = get_accuracy(model_2, x_test, y_test)
    lst = all(y >= x for x, y in zip(res1, res2))
    assert lst


def get_accuracy(trained_model, x_test, y_test):
    """
    Helper method that returns the accuracy of a model.
    :param trained_model: trained classification model
    :param x_test: list of features for testing data
    :param y_test: list of outputs for testing data
    :return: accuracy of model
    """
    y_labels = trained_model.predict(x_test)
    return [f1_score(y_test, y_labels, average='weighted'),
            average_precision_score(y_test, y_labels, average='macro'),
            accuracy_score(y_test, y_labels)]

# todo add more
