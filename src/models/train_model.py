from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
import pickle
import sys
import pandas as pd

OUT_PATH_MODEL = 'models/model.pkl'

# def train_mybag(X_train, X_val, X_test, y_train):
#     """
#     The data is transformed to sparse representation (this might take up to a minute), to store the useful
#     information efficiently. There are many [types](https://docs.scipy.org/doc/scipy/reference/sparse.html) of such
#     representations, however sklearn algorithms can work only with [csr](
#     https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix)
#     matrix.

#     :return: All samples with my_bag_of_words applied to it.
#     """
#     words_counts, _ = word_tags_count(X_train, y_train)
#     INDEX_TO_WORDS = sorted(words_counts, key=words_counts.get, reverse=True)[:DICT_SIZE]
#     WORDS_TO_INDEX = {word: i for i, word in enumerate(INDEX_TO_WORDS)}
#     ALL_WORDS = WORDS_TO_INDEX.keys()

#     X_train_mybag = sp_sparse.vstack(
#         [sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_train])
#     X_val_mybag = sp_sparse.vstack(
#         [sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_val])
#     X_test_mybag = sp_sparse.vstack(
#         [sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_test])
#     print('X_train shape ', X_train_mybag.shape)
#     print('X_val shape ', X_val_mybag.shape)
#     print('X_test shape ', X_test_mybag.shape)
#     return X_train_mybag, X_val_mybag, X_test_mybag


"""
MultiLabel classifier
In this task each example can have multiple tags therefore transform labels in a binary form and the prediction will be a mask of 0s and 1s. 
For this purpose it is convenient to use MultiLabelBinarizer from sklearn.
"""


def train_classifier(X_train, y_train, penalty='l1', C=1):
    """
      X_train, y_train — training data
      
      return: trained classifier
    """

    # Create and fit LogisticRegression wrapped into OneVsRestClassifier.
    clf = LogisticRegression(penalty=penalty, C=C, dual=False, solver='liblinear')
    clf = OneVsRestClassifier(clf)
    clf.fit(X_train, y_train)

    return clf


def train_classifier_for_transformations(
    X_train_tfidf,
    y_train
):
    """
    :param X_train_tfidf:
    :param y_train:
    :return: Trained classifier for tf-idf
    """
    classifier_tfidf = train_classifier(X_train_tfidf, y_train)
    return classifier_tfidf

def pickle_model(clf):
    with open(OUT_PATH_MODEL, 'wb') as fd:
        pickle.dump(clf, fd, protocol=pickle.HIGHEST_PROTOCOL)

def main():
    if len(sys.argv) != 2:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython src/models/train_model.py train-file-path\n")
        sys.exit(1)

    train = pd.read_csv(sys.argv[1])

    X_train, y_train = train[['X_train']], train[['y_train']]
    clf = train_classifier_for_transformations(X_train, y_train)

    pickle_model(clf)

if __name__ == '__main__':
    main()