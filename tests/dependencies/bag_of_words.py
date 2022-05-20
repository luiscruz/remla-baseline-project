from joblib import dump, load
import numpy as np
from scipy import sparse as sp_sparse


def bag_of_words(text, words_to_index, dict_size):
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


def main():
    preprocessed_data = load('output/preprocessed_data.joblib')

    X_train = preprocessed_data["X_train"]
    X_val = preprocessed_data["X_val"]
    X_test = preprocessed_data["X_test"]
    y_train = preprocessed_data["y_train"]
    y_val = preprocessed_data["y_val"]

    # Dictionary of all tags from train corpus with their counts.
    tags_counts = {}
    # Dictionary of all words from train corpus with their counts.
    words_counts = {}

    # Get the counts of every word
    for sentence in X_train:
        for word in sentence.split():
            if word in words_counts:
                words_counts[word] += 1
            else:
                words_counts[word] = 1

    # Get the counts of every tag that we have
    for tags in y_train:
        for tag in tags:
            if tag in tags_counts:
                tags_counts[tag] += 1
            else:
                tags_counts[tag] = 1

    DICT_SIZE = 5000
    INDEX_TO_WORDS = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:DICT_SIZE]
    WORDS_TO_INDEX = {word: i for i, word in enumerate(INDEX_TO_WORDS)}
    ALL_WORDS = WORDS_TO_INDEX.keys()

    X_train = sp_sparse.vstack([sp_sparse.csr_matrix(
        bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_train])
    X_val = sp_sparse.vstack([sp_sparse.csr_matrix(
        bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_val])
    X_test = sp_sparse.vstack([sp_sparse.csr_matrix(
        bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_test])

    output_data = {"X_train": X_train, "X_val": X_val,
                   "X_test": X_test, "y_train": y_train, "y_val": y_val, "tags_counts": tags_counts}

    dump(output_data, 'output/bag_of_words.joblib')

    dump({"all_words": list(ALL_WORDS), "vocabulary": INDEX_TO_WORDS}, "output/words_dictionaries.joblib")


if __name__ == "__main__":
    main()
