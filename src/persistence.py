import os
from joblib import dump, load

from src.model import get_classifiers

CLF_MYBAG_FILE = 'clf_mybag.joblib'
CLF_TFIDF_FILE = 'clf_tfidf.joblib'
MLB_FILE = 'mlb.joblib'
TFID_VOCAB_FILE = 'tfid-vocab.joblib'
BAG_WORDS_VARS_FILE = 'bag-words-vars.joblilb'

DEFAULT_EXPORTS_PATH = '../exports'


def train_and_save_models(path=DEFAULT_EXPORTS_PATH):
    classifier_mybag, classifier_tfidf, y_train, y_val, mlb, tfidf_vectorizer, words_to_index, dict_size = get_classifiers()

    dump(classifier_mybag, os.path.join(path, CLF_MYBAG_FILE))
    dump(classifier_tfidf, os.path.join(path, CLF_TFIDF_FILE))
    dump(mlb, os.path.join(path, MLB_FILE))
    dump(tfidf_vectorizer, os.path.join(path, TFID_VOCAB_FILE))
    dump((words_to_index, dict_size), os.path.join(path, BAG_WORDS_VARS_FILE))


def load_models(path=DEFAULT_EXPORTS_PATH):
    classifier_mybag = load(os.path.join(path, CLF_MYBAG_FILE))
    classifier_tfidf = load(os.path.join(path, CLF_TFIDF_FILE))
    mlb = load(os.path.join(path, MLB_FILE))
    tfidf_vectorizer = load(os.path.join(path, TFID_VOCAB_FILE))
    words_to_index, dict_size = load(os.path.join(path, BAG_WORDS_VARS_FILE))

    return classifier_mybag, classifier_tfidf, mlb, tfidf_vectorizer, words_to_index, dict_size
