from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from joblib import dump, load
import os
import json
RESULTS_DIRECTION = "results/metrics.json"


def dump_eval_results(results):

    if os.path.exists(RESULTS_DIRECTION):
        creation_time = int(os.path.getctime(RESULTS_DIRECTION)*10e5)
        os.rename(RESULTS_DIRECTION, RESULTS_DIRECTION.replace("metrics.", f"metrics-{creation_time}.json"))

    with open(f"{RESULTS_DIRECTION}", "w") as file_write:
        file_write.write(json.dumps(results, indent=4))


def print_evaluation_scores(y_val, predicted, prediction_results):

    results = {}
    results["accuracy"] = accuracy_score(y_val, predicted)
    results["f1"] = f1_score(y_val, predicted, average='weighted')
    results["precision"] = average_precision_score(y_val, predicted, average='macro')
    results["roc"] = roc_auc_score(y_val, prediction_results["scores"], multi_class="ovo")

    dump_eval_results(results)

# Change variable names


def main():
    prediction_results = load('output/prediction_results.joblib')

    scores = prediction_results["scores"]
    labels = prediction_results["labels"]
    y_val = load('output/val_data.joblib')

    print_evaluation_scores(y_val, labels, prediction_results)


if __name__ == "__main__":
    main()
