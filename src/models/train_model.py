from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from src.data.make_data import init_data
from src.features.build_features import *


def train_mybag(X_train, X_val, X_test, y_train):
    """
    The data is transformed to sparse representation (this might take up to a minute), to store the useful information efficiently.
    There are many [types](https://docs.scipy.org/doc/scipy/reference/sparse.html) of such representations, however sklearn algorithms can work only with [csr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix) matrix.
    :return: All samples with my_bag_of_words applied to it.
    """
    words_counts, _ = word_tags_count(X_train, y_train)
    INDEX_TO_WORDS = sorted(words_counts, key=words_counts.get, reverse=True)[:DICT_SIZE]
    WORDS_TO_INDEX = {word: i for i, word in enumerate(INDEX_TO_WORDS)}
    ALL_WORDS = WORDS_TO_INDEX.keys()


    X_train_mybag = sp_sparse.vstack(
        [sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_train])
    X_val_mybag = sp_sparse.vstack(
        [sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_val])
    X_test_mybag = sp_sparse.vstack(
        [sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_test])
    print('X_train shape ', X_train_mybag.shape)
    print('X_val shape ', X_val_mybag.shape)
    print('X_test shape ', X_test_mybag.shape)
    return X_train_mybag, X_val_mybag, X_test_mybag


"""
MultiLabel classifier
In this task each example can have multiple tags therefore transform labels in a binary form and the prediction will be a mask of 0s and 1s. 
For this purpose it is convenient to use MultiLabelBinarizer from sklearn.
"""


def train_classifier(X_train, y_train, y_val, penalty='l1', C=1):
    """
      X_train, y_train — training data
      
      return: trained classifier
    """

    tags_counts, words_counts, most_common_tags, most_common_words = word_tags_count(X_train, y_train)

    mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
    y_train = mlb.fit_transform(y_train)
    y_val = mlb.fit_transform(y_val)

    # Create and fit LogisticRegression wraped into OneVsRestClassifier.

    clf = LogisticRegression(penalty=penalty, C=C, dual=False, solver='liblinear')
    clf = OneVsRestClassifier(clf)
    clf.fit(X_train, y_train)

    return y_train, y_val, clf


def train_classifier_for_transformations(X_train, X_val, X_test, y_train):
    X_train_mybag, X_val_mybag, X_test_mybag = train_mybag(X_train, X_val, X_test, y_train)
    X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vocab = tfidf_features(X_train, X_val, X_test)
    classifier_mybag = train_classifier(X_train_mybag, y_train)
    classifier_tfidf = train_classifier(X_train_tfidf, y_train)
    return classifier_mybag, classifier_tfidf

