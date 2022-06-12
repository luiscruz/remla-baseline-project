import logging
import pickle

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    recall_score,
)
from sklearn.metrics import roc_auc_score as roc_auc


def print_evaluation_scores(logger, y_val, predicted_labels, predicted_scores):
    logger.info(
        "Accuracy score: " + "{:.2f}".format(accuracy_score(y_val, predicted_labels))
    )
    logger.info(
        "Recall score: "
        + "{:.2f}".format(recall_score(y_val, predicted_labels, average="macro"))
    )
    logger.info(
        "F1 score: "
        + "{:.2f}".format(f1_score(y_val, predicted_labels, average="weighted"))
    )
    logger.info(
        "Average precision score: "
        + "{:.2f}".format(
            average_precision_score(y_val, predicted_labels, average="macro")
        )
    )
    logger.info(
        "ROC score: "
        + "{:.2f}".format(roc_auc(y_val, predicted_scores, multi_class="ovo"))
    )
    logger.info("-------- \n")


def main():
    logger = logging.getLogger(__name__)
    logger.info("Starting the program")

    logger.info("Load data")
    input_filepath = "../../data/processed/"

    bow_val = pickle.load(open(input_filepath + "bow_val.pickle", "rb"))
    tfidf_val = pickle.load(open(input_filepath + "tfidf_val.pickle", "rb"))
    mlb_y_val = pickle.load(open(input_filepath + "mlb_val.pickle", "rb"))

    model_filepath = "../../models/"
    classifier_bow = pickle.load(open(model_filepath + "bow_model.pickle", "rb"))
    classifier_tfidf = pickle.load(open(model_filepath + "tfidf_model.pickle", "rb"))

    y_val_predicted_labels_bow = classifier_bow.predict(bow_val)
    y_val_predicted_scores_bow = classifier_bow.decision_function(bow_val)

    y_val_predicted_labels_tfidf = classifier_tfidf.predict(tfidf_val)
    y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(tfidf_val)

    logger.info("*** Bag-of-words scores ***")
    print_evaluation_scores(
        logger, mlb_y_val, y_val_predicted_labels_bow, y_val_predicted_scores_bow
    )
    logger.info("*** Tfidf scores ***")
    print_evaluation_scores(
        logger, mlb_y_val, y_val_predicted_labels_tfidf, y_val_predicted_scores_tfidf
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
