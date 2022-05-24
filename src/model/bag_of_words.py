import numpy as np
from scipy import sparse as sp_sparse


ALL_WORDS = []

def initialize(words_counts, X_train, X_val, X_test):
    DICT_SIZE = 5000
    INDEX_TO_WORDS = sorted(words_counts, key=words_counts.get, reverse=True)[
        :DICT_SIZE]  # YOUR CODE HERE #######
    WORDS_TO_INDEX = {word: i for i, word in enumerate(INDEX_TO_WORDS)}
    ALL_WORDS = WORDS_TO_INDEX.keys()

    X_train_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(
        my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_train])
    X_val_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(
        my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_val])
    X_test_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(
        my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_test])
    print('X_train shape ', X_train_mybag.shape)
    print('X_val shape ', X_val_mybag.shape)
    print('X_test shape ', X_test_mybag.shape)

    return X_train_mybag, X_val_mybag


def my_bag_of_words(text, words_to_index, dict_size):
    """
            text: a string
            dict_size: size of the dictionary

            return a vector which is a bag-of-words representation of 'text'
    """
    result_vector = np.zeros(dict_size)

    for word in text.split():
        if word in words_to_index:
            result_vector[words_to_index[word]] += 1
    return result_vector
