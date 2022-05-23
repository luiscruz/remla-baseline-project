import logging

import numpy as np
import pandas as pd
from scipy import sparse as sp_sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.util.util import read_data, write_data


def tfidf_features(X_train, X_val, X_test):
    """
        X_train, X_val, X_test — samples
        return TF-IDF vectorized representation of each sample and vocabulary
    """
    # Create TF-IDF vectorizer with a proper parameters choice
    # Fit the vectorizer on the train set
    # Transform the train, test, and val sets and return the result

    tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2),
                                       token_pattern='(\S+)')  ####### YOUR CODE HERE #######

    X_train = tfidf_vectorizer.fit_transform(X_train)
    X_val = tfidf_vectorizer.transform(X_val)
    X_test = tfidf_vectorizer.transform(X_test)

    return X_train, X_val, X_test, tfidf_vectorizer.vocabulary_


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


def count_words_strings(list_of_words):
    count_dict = {}
    for sentence in list_of_words:
        for word in sentence.split():
            if word in count_dict:
                count_dict[word] += 1
            else:
                count_dict[word] = 1

    return count_dict


def count_words_lists(list_of_lists):
    count_dict = {}
    for tags in list_of_lists:
        for tag in tags:
            if tag in count_dict:
                count_dict[tag] += 1
            else:
                count_dict[tag] = 1

    return count_dict


def bow_features(X_train, y_train, X_val, X_test):
    words_counts = count_words_strings(X_train)
    tags_counts = count_words_lists(y_train)

    DICT_SIZE = 5000
    INDEX_TO_WORDS = sorted(words_counts, key=words_counts.get, reverse=True)[:DICT_SIZE]
    WORDS_TO_INDEX = {word: i for i, word in enumerate(INDEX_TO_WORDS)}
    ALL_WORDS = WORDS_TO_INDEX.keys()

    X_train_mybag = sp_sparse.vstack(
        [sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_train])
    X_val_mybag = sp_sparse.vstack(
        [sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_val])
    X_test_mybag = sp_sparse.vstack(
        [sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_test])

    return X_train_mybag, X_val_mybag, X_test_mybag


def main(input_filepath='../../data/interim/', output_filepath='../../data/processed/'):
    """ Runs data processing scripts to turn pre-processed data from (../interim) into
        feature data ready to be trained/tested (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from interim data')

    train_file_name_in = input_filepath + 'train.tsv'
    val_file_name_in = input_filepath + 'validation.tsv'
    test_file_name_in = input_filepath + 'test.tsv'

    train_file_name_out = output_filepath + 'train.tsv'
    val_file_name_out = output_filepath + 'validation.tsv'
    test_file_name_out = output_filepath + 'test.tsv'

    # Load data from tsv files in directory
    train = read_data(train_file_name_in)
    validation = read_data(val_file_name_in)
    test = read_data(test_file_name_in)

    logger.info(
        'Finished reading data from: \n\t' + train_file_name_in + '\n\t' + val_file_name_in + '\n\t' + test_file_name_in)

    # Select columns to use
    X_train, y_train = train['title'].values, train['tags'].values
    X_val, y_val = validation['title'].values, validation['tags'].values
    X_test = test['title'].values

    bow_train, bow_val, bow_test = bow_features(X_train, y_train, X_val, X_test)
    logger.info('Finished generating the bag of words matrices')

    tfidf_train, tfidf_val, tfidf_test, tfidf_vocab = tfidf_features(X_train, X_val, X_test)
    logger.info('Finished generating the tfidf')

    #  Lists to pd for easy writing
    train_out = pd.DataFrame(list(zip(X_train, y_train, bow_train, tfidf_train)),
                             columns=['title', 'tags', 'bow', 'tfidf'])
    val_out = pd.DataFrame(list(zip(X_val, y_val, bow_val, tfidf_val)),
                           columns=['title', 'tags', 'bow', 'tfidf'])
    test_out = pd.DataFrame(list(zip(X_test, bow_test, tfidf_test)), columns=['title', 'bow', 'tfidf'])

    # Write data to files
    write_data(train_out, train_file_name_out, ';')
    write_data(val_out, val_file_name_out, ';')
    write_data(test_out, test_file_name_out, ';')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
