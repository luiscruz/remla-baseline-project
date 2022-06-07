"""Module used to evaluate the trained model in the ML pipeline"""
import json
import pickle

import yaml
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from sklearn.metrics import roc_auc_score as roc_auc

# Fetch params from yaml params file
with open("params.yaml", encoding="utf-8") as f:
    params = yaml.safe_load(f)
train_params = params["train"]
featurize_params = params["featurize"]
evaulate_params = params["evaluate"]

MODEL_PATH = train_params["model_out"]
VAL_DATA_PATH = featurize_params["output_val"]
# TEST_DATA_SET = featurize_params['output_test']

# PRC_IMG_PATH = evaulate_params['prc_img']
# ROC_IMG_PATH = evaulate_params['roc_img']
PRC_JSON_PATH = evaulate_params["prc_json"]
ROC_JSON_PATH = evaulate_params["roc_json"]
SCORES_JSON_PATH = evaulate_params["scores_path"]


def create_all_plots_and_scores(classifier_tfidf, X_val_tfidf, y_val_tdidf):
    """Driver to create the plots/metrics"""
    y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
    y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)

    create_evaluation_scores_json(y_val_tdidf, y_val_predicted_labels_tfidf, y_val_predicted_scores_tfidf)

    # create_prc_curve_json(y_val_tdidf, y_val_predicted_scores_tfidf)
    # create_roc_curve_json(y_val_tdidf, predicted_probas_tfidf)


def create_evaluation_scores_json(y_val, predicted_labels, probas):
    """
    Evaluate the model and save metrics to json file
    :param y_val: true y values
    :param predicted_labels: predicted labels
    :param probas: predicted probabilities
    """
    with open(SCORES_JSON_PATH, "w", encoding="utf-8") as fd:
        json.dump(
            {
                "accuracy_score": accuracy_score(y_val, predicted_labels),
                "f1_score": f1_score(y_val, predicted_labels, average="weighted"),
                "avg_precision_score": average_precision_score(y_val, predicted_labels, average="macro"),
                "roc_auc_score": roc_auc(y_val, probas, multi_class="ovo"),
            },
            fd,
            indent=4,
        )


# def create_prc_curve_json(y_val, probas_pred):
#     """
#     Create json file of prc points
#     """
#     precision, recall, prc_thresholds = precision_recall_curve(y_val, probas_pred)
#     nth_point = math.ceil(len(prc_thresholds) / 1000)
#     prc_points = list(zip(precision, recall, prc_thresholds))[::nth_point]
#     with open(PRC_JSON_PATH, "w") as fd:
#         json.dump(
#             {
#                 "prc": [
#                     {"precision": p, "recall": r, "threshold": t}
#                     for p, r, t in prc_points
#                 ]
#             },
#             fd,
#             indent=4,
#         )

# def create_roc_curve_json(y_val, probas_pred):
#     """
#     Create json file of roc points
#     """
#     fpr, tpr, roc_thresholds = roc_curve(y_val, probas_pred)
#     roc_points = list(zip(fpr, tpr, roc_thresholds))
#     with open(ROC_JSON_PATH, "w") as fd:
#         json.dump(
#             {
#                 "roc": [
#                     {"fpr": fpr, "tpr": tpr, "threshold": t}
#                     for fpr, tpr, t in roc_points
#                 ]
#             },
#             fd,
#             indent=4,
#         )


def _load_model():
    with open(MODEL_PATH, "rb") as fd:
        return pickle.load(fd)


def _load_val_data():
    with open(VAL_DATA_PATH, "rb") as fd:
        X_val, y_val = pickle.load(fd)
    return X_val, y_val


def main():
    """Load model and evaluate"""
    clf = _load_model()
    X_val, y_val = _load_val_data()
    create_all_plots_and_scores(clf, X_val, y_val)


if __name__ == "__main__":
    main()
