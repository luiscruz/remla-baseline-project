from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


def get_diff_stats(X_labels_og, X_labels_new, y_val):

    results = {}

    results["accuracy"] = abs(accuracy_score(y_val, X_labels_og)-accuracy_score(y_val, X_labels_new))
    results["f1"] = abs(f1_score(y_val, X_labels_og, average='weighted') -
                        f1_score(y_val, X_labels_new, average='weighted'))
    results["precision"] = abs(average_precision_score(y_val, X_labels_og, average='macro') -
                               average_precision_score(y_val, X_labels_new, average='macro'))
    results["roc"] = abs(roc_auc_score(y_val, X_labels_og, multi_class="ovo") -
                         roc_auc_score(y_val, X_labels_new, multi_class="ovo"))

    return results


def check_diff(values, limit=0.1):
    for key, difference in values.items():
        assert difference < limit, f"The value of the {key} stat, differs in {difference}"
