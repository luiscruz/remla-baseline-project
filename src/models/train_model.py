import logging
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


def train_classifier(X_train, y_train, penalty="l1", C=1):
    """
    X_train, y_train — training data

    return: trained classifier
    """

    # Create and fit LogisticRegression wraped into OneVsRestClassifier.

    clf = LogisticRegression(penalty=penalty, C=C, dual=False, solver="liblinear")
    clf = OneVsRestClassifier(clf)
    clf.fit(X_train, y_train)

    return clf


def main():
    logger = logging.getLogger(__name__)
    logger.info("Starting the program")

    logger.info("Load data")
    input_filepath = "data/processed/"
    bow_train = pickle.load(open(input_filepath + "bow_train.pickle", "rb"))
    tfidf_train = pickle.load(open(input_filepath + "tfidf_train.pickle", "rb"))
    mlb_y_train = pickle.load(open(input_filepath + "mlb_train.pickle", "rb"))

    logger.info("Train BOW classifier")
    classifier_mybag = train_classifier(bow_train, mlb_y_train)
    logger.info("Train TFIDF classifier")
    classifier_tfidf = train_classifier(tfidf_train, mlb_y_train)

    logger.info("Store model")
    output_filepath = "models/"
    pickle.dump(classifier_mybag, open(output_filepath + "bow_model.pickle", "wb"))
    pickle.dump(classifier_tfidf, open(output_filepath + "tfidf_model.pickle", "wb"))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
