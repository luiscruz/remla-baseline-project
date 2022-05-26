from typing import Dict

import numpy as np
from scipy import sparse as sp_sparse


DICT_SIZE = 5000


def bag_of_words(text: str, words_to_index: Dict[str, int], dict_size: int):
    result_vector = np.zeros(dict_size)

    for word in text.split():
        if word in words_to_index:
            result_vector[words_to_index[word]] += 1

    return result_vector


def sparse_bag_of_words(X: list[str], words_to_index: Dict[str, int]) -> sp_sparse.bmat:
    return sp_sparse.vstack(
        [
            sp_sparse.csr_matrix(bag_of_words(text, words_to_index, DICT_SIZE))
            for text in X
        ]
    )


def initialize(
    words_counts: Dict[str, int],
    X_train: list[str],
    X_val: list[str],
    X_test: list[str],
) -> sp_sparse.bmat:
    index_to_words: list[str] = sorted(
        words_counts, key=words_counts.get, reverse=True
    )[:DICT_SIZE]

    words_to_index = {word: i for i, word in enumerate(index_to_words)}

    X_train_bag = sparse_bag_of_words(X_train, words_to_index)
    X_val_bag = sparse_bag_of_words(X_val, words_to_index)
    X_test_bag = sparse_bag_of_words(X_test, words_to_index)

    print("X_train shape ", X_train_bag.shape)
    print("X_val shape ", X_val_bag.shape)
    print("X_test shape ", X_test_bag.shape)

    return X_train_bag, X_val_bag
