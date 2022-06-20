import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    recall_score,
)
from sklearn.metrics import roc_auc_score as roc_auc

from src.project_types import ModelName

source_file = Path(__file__)
project_dir = source_file.parent.parent.parent


def validation_file(model: ModelName):
    input_filepath = project_dir / "data/processed/"
    return pickle.load(input_filepath.joinpath(f"{model}_val.pickle").open("rb"))


def model_file(model: ModelName):
    model_filepath = project_dir / "models/"
    return pickle.load(model_filepath.joinpath(f"{model}_model.pickle").open("rb"))


def main():
    logger = logging.getLogger(__name__)
    logger.info("Starting the program")

    logger.info("Load data")

    evaluator = Evaluator()

    logger.info("*** Bag-of-words scores ***")
    print_evaluation_scores(logger, evaluator.evaluate(ModelName.bow))
    logger.info("*** Tfidf scores ***")
    print_evaluation_scores(logger, evaluator.evaluate(ModelName.tfidf))


class Evaluator:

    evaluation_metrics = [
        "accuracy",
        "recall",
        "f1-score",
        "average-precision-score",
        "roc-score",
    ]

    def __init__(self):
        self.mlb_y_val = validation_file(ModelName.mlb)

    def evaluate(self, model: ModelName) -> Dict[str, float]:
        model_val = validation_file(model)
        classifier_model = model_file(model)

        y_val_predicted_labels = classifier_model.predict(model_val)
        y_val_predicted_scores = classifier_model.decision_function(model_val)

        return {
            "accuracy": accuracy_score(self.mlb_y_val, y_val_predicted_labels),
            "recall": recall_score(
                self.mlb_y_val, y_val_predicted_labels, average="macro"
            ),
            "f1-score": f1_score(
                self.mlb_y_val, y_val_predicted_labels, average="weighted"
            ),
            "average-precision-score": average_precision_score(
                self.mlb_y_val, y_val_predicted_labels, average="macro"
            ),
            "roc-score": roc_auc(
                self.mlb_y_val, y_val_predicted_scores, multi_class="ovo"
            ),
        }


default_format = "{:.2f}"


def print_evaluation_scores(
    logger: logging.Logger, scores: Dict[str, float], format: str = default_format
):
    text = evaluation_scores_to_text(scores, format)
    for line in text:
        logger.info(line)


def evaluation_scores_to_text(
    scores: Dict[str, float], format: str = default_format
) -> List[str]:
    return [
        "Accuracy score: " + format.format(scores["accuracy"]),
        "Recall score: " + format.format(scores["recall"]),
        "F1 score: " + format.format(scores["f1-score"]),
        "Average precision score: " + format.format(scores["average-precision-score"]),
        "ROC score: " + format.format(scores["roc-score"]),
        "-------- \n",
    ]


if __name__ == "__main__":
    print("name", __name__)
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
