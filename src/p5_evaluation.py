from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from joblib import load
import os
import json


def dump_eval_results(y_val, prediction_results, type_pred):

    RESULTS_DIRECTION = "results/metrics-$type.json"

    labels = prediction_results[type_pred]["labels"]
    scores = prediction_results[type_pred]["scores"]

    results = {}
    results["accuracy"] = accuracy_score(y_val, labels)
    results["f1"] = f1_score(y_val, labels, average='weighted')
    results["precision"] = average_precision_score(y_val, labels, average='macro')
    results["roc"] = roc_auc_score(y_val, scores, multi_class="ovo")

    RESULTS_DIRECTION = RESULTS_DIRECTION.replace("$type", type_pred)

    if os.path.exists(RESULTS_DIRECTION):
        creation_time = int(os.path.getctime(RESULTS_DIRECTION)*10e5)
        os.rename(RESULTS_DIRECTION, RESULTS_DIRECTION.replace(".json", f"-{creation_time}.json"))

    with open(f"{RESULTS_DIRECTION}", "w") as file_write:
        file_write.write(json.dumps(results, indent=4))


# Change variable names


def main():
    prediction_results = load('output/predictions.joblib')
    y_val = load('output/val_data.joblib')

    dump_eval_results(y_val, prediction_results, "bag")
    dump_eval_results(y_val, prediction_results, "tfidf")


if __name__ == "__main__":
    main()
