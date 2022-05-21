from joblib import dump, load

from model import get_classifiers


def train_and_save_models(data_dir, model_path):
    classifier_mybag, classifier_tfidf, y_train, y_val, mlb, tfidf_vectorizer, words_to_index, dict_size = get_classifiers(data_dir)
    dump((classifier_mybag, classifier_tfidf, mlb, tfidf_vectorizer, words_to_index, dict_size), model_path)


def load_models(model_path):
    return load(model_path)
