"""Module used in building features in the ML pipeline."""
import pickle

import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

# Fetch params from yaml params file
with open("params.yaml", encoding="utf-8") as f:
    params = yaml.safe_load(f)
preprocess_params = params["preprocess"]
featurize_params = params["featurize"]

INPUT_TRAIN_PATH = preprocess_params["output_train"]
INPUT_VAL_PATH = preprocess_params["output_val"]
INPUT_TEST_PATH = preprocess_params["output_test"]

OUT_PATH_TRAIN = featurize_params["output_train"]
OUT_PATH_VAL = featurize_params["output_val"]
OUT_PATH_TEST = featurize_params["output_test"]
OUT_MLB_PICKLE = featurize_params["mlb_out"]


def word_tags_count(X_train, y_train):
    """
    :param: X_train
    :param: y_train
    :return: tags_counts, words_counts
    """
    # Dictionary of all tags from train corpus with their counts.
    tags_counts = {}
    # Dictionary of all words from train corpus with their counts.
    words_counts = {}

    for sentence in X_train:
        for word in sentence.split():
            if word in words_counts:
                words_counts[word] += 1
            else:
                words_counts[word] = 1

    for tags in y_train:
        for tag in tags:
            if tag in tags_counts:
                tags_counts[tag] += 1
            else:
                tags_counts[tag] = 1

    # We are assuming that tags_counts and words_counts are dictionaries like {'some_word_or_tag': frequency}. After
    # applying the sorting procedure, results will be look like this: [('most_popular_word_or_tag', frequency),
    # ('less_popular_word_or_tag', frequency), ...]

    return tags_counts, words_counts


def tfidf_features(X_train, X_val, X_test):
    """
    X_train, X_val, X_test — samples
    :return: TF-IDF vectorized representation of each sample and vocabulary
    """
    # Create TF-IDF vectorizer with a proper parameters choice
    # Fit the vectorizer on the train set
    # Transform the train, test, and val sets and return the result
    tfidf_vectorizer = TfidfVectorizer(  # nosec
        # pylint: disable = anomalous - backslash - in -string
        min_df=5,
        max_df=0.9,
        ngram_range=(1, 2),
        token_pattern="(\S+)",  # noqa: W60
    )

    X_train = tfidf_vectorizer.fit_transform(X_train)
    X_val = tfidf_vectorizer.transform(X_val)
    X_test = tfidf_vectorizer.transform(X_test)

    return X_train, X_val, X_test, tfidf_vectorizer.vocabulary_


def mlb_y_data(y_train, y_val, tags_counts):
    """Perform MultiLabelBinarization on the counts of all tags
    :param y_train: training tags
    :param y_val: validation tags
    :param Dict[str->int] tags_counts:  where keys are tags and values are how often they occur in the trainset
    """
    mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
    y_train = mlb.fit_transform(y_train)
    y_val = mlb.fit_transform(y_val)
    return y_train, y_val, mlb


def _pickle_mlb(mlb_obj):
    with open(OUT_MLB_PICKLE, "wb") as fd:
        pickle.dump(mlb_obj, fd, protocol=pickle.HIGHEST_PROTOCOL)


def _pickle_sparse_matrix(csr_matrix, label_csr, output_path):
    with open(output_path, "wb") as fd:
        pickle.dump((csr_matrix, label_csr), fd)


def _pickle_sparse_test_matrix(csr_matrix, output_path):
    with open(output_path, "wb") as fd:
        pickle.dump(csr_matrix, fd)


def _load_pickled_data(input_path):
    with open(input_path, "rb") as fd:
        return pickle.load(fd)


def main():
    """
    Controller for building the features.
    First loads all input data, performs Multi Label Binarization.
    Creates the features and pickles these.
    """
    X_train, y_train = _load_pickled_data(INPUT_TRAIN_PATH)
    X_val, y_val = _load_pickled_data(INPUT_VAL_PATH)
    X_test = _load_pickled_data(INPUT_TEST_PATH)

    tags_counts, _ = word_tags_count(X_train=X_train, y_train=y_train)
    y_train, y_val, mlb = mlb_y_data(y_train, y_val, tags_counts)

    X_train_csr, X_val_csr, X_test_csr, _ = tfidf_features(X_train, X_val, X_test)

    _pickle_sparse_matrix(X_train_csr, y_train, OUT_PATH_TRAIN)
    _pickle_sparse_matrix(X_val_csr, y_val, OUT_PATH_VAL)
    _pickle_sparse_test_matrix(X_test_csr, OUT_PATH_TEST)

    _pickle_mlb(mlb)


if __name__ == "__main__":
    main()
