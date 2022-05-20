"""
    Main logic of this stage:
    * loads the loblibs
    * vectorizes (embeds) the data
    * dumps the vectors in a joblib
"""
import joblib
import numpy as np
from scipy import sparse as sp_sparse
from sklearn.feature_extraction.text import TfidfVectorizer

output_directory = "output"
np.random.seed(seed=0)


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


def tfidf_features(X_train, X_val, X_test):
    """
        X_train, X_val, X_test — samples
        return TF-IDF vectorized representation of each sample and vocabulary
    """
    # Create TF-IDF vectorizer with a proper parameters choice
    # Fit the vectorizer on the train set
    # Transform the train, test, and val sets and return the result

    tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2),
                                       token_pattern='(\S+)')  # pylint: disable=anomalous-backslash-in-string

    X_train = tfidf_vectorizer.fit_transform(X_train)
    X_val = tfidf_vectorizer.transform(X_val)
    X_test = tfidf_vectorizer.transform(X_test)

    return X_train, X_val, X_test, tfidf_vectorizer.vocabulary_


# CODE TO TEST RUN THE METHODS
def main():
    """
        Main logic of this stage:
        * loads the loblibs
        * vectorizes (embeds) the data
        * dumps the vectors in a joblib
    """
    X_train, X_val, X_test = joblib.load(output_directory + "/X_preprocessed.joblib")
    words_counts = joblib.load(output_directory + "/words_counts.joblib")

    DICT_SIZE = 5000
    INDEX_TO_WORDS = sorted(words_counts, key=words_counts.get, reverse=True)[
                     :DICT_SIZE]  # YOUR CODE HERE #######
    WORDS_TO_INDEX = {word: i for i, word in enumerate(INDEX_TO_WORDS)}
    # ALL_WORDS = WORDS_TO_INDEX.keys()

    X_train_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(
        my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_train])
    X_val_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(
        my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_val])
    X_test_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(
        my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_test])

    print('X_train shape ', X_train_mybag.shape)
    print('X_val shape ', X_val_mybag.shape)
    print('X_test shape ', X_test_mybag.shape)

    # TF-IDF
    X_train_tfidf, X_val_tfidf, _, _ = tfidf_features(X_train, X_val, X_test)
    # tfidf_reversed_vocab = {i: word for word, i in tfidf_vocab.items()}

    # tfidf_vocab["c#"]
    #
    # tfidf_reversed_vocab[1879]

    joblib.dump((X_train_mybag, X_train_tfidf, X_val_mybag, X_val_tfidf), output_directory + "/vectorized_x.joblib")


if __name__ == "__main__":
    main()
