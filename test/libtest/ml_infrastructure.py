"""
    ML Test library functions for the ML infrastructure.
    Based on section 4 of the paper referenced below.

    Eric Breck, Shanqing Cai, Eric Nielsen, Michael Salib, D. Sculley (2016). What’s your ML test score? A rubric for ML production systems. Reliable Machine Learning in the Wild - NIPS 2016 Workshop (2016).
    Available: https://storage.googleapis.com/pub-tools-public-publication-data/pdf/45742.pdf
"""
from math import isclose

from sklearn.metrics import f1_score, average_precision_score, accuracy_score, roc_auc_score


def test_reproducibility_training(model, x_train, y_train, x_test, y_test, eval_metrics=None, precision=1e-3):
    models = {}
    for i in range(2):
        trained_model = model.fit(x_train, y_train)
        y_labels = trained_model.predict(x_test)
        y_score = trained_model.decision_function(x_test)
        models["model_" + str(i)] = [f1_score(y_test, y_labels, average='weighted'),
                                     average_precision_score(y_test, y_labels, average='macro'),
                                     accuracy_score(y_test, y_labels),
                                     roc_auc_score(y_test, y_score, multi_class='ovo')]

    lst = all(isclose(x, y, rel_tol=precision) for x, y in zip(models["model_0"], models["model_1"]))
    print(lst, models)
    assert lst

# todo add more
