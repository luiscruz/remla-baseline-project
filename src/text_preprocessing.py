import nltk
from nltk.corpus import stopwords
from ast import literal_eval
import pandas as pd
import numpy as np
import re
from scipy import sparse as sp_sparse
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def read_data(filename):
    data = pd.read_csv(filename, sep='\t')
    data['tags'] = data['tags'].apply(literal_eval)
    return data

def text_prepare(text):
    """
        text: a string

        return: modified initial string
    """
    text = text.lower()  # lowercase text
    text = re.sub(REPLACE_BY_SPACE_RE, " ", text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = re.sub(BAD_SYMBOLS_RE, "", text)  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = " ".join([word for word in text.split() if not word in STOPWORDS])  # delete stopwords from text
    return text

def get_word_tags_counts(X_train, y_train):
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

    return words_counts, tags_counts


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
                                       token_pattern='(\S+)')  ####### YOUR CODE HERE #######

    X_train = tfidf_vectorizer.fit_transform(X_train)
    X_val = tfidf_vectorizer.transform(X_val)
    X_test = tfidf_vectorizer.transform(X_test)

    return X_train, X_val, X_test, tfidf_vectorizer


def get_train_test_data(data):
    '''
    :param data: 1 for just the data, 2 for including the bag-of-words representation, 3 for including the tf-idf representation
    :return: partial or all data
    '''
    train = read_data('../data/raw/train/train.tsv')
    validation = read_data('../data/raw/eval/validation.tsv')
    test = pd.read_csv('../data/raw/eval/test.tsv', sep='\t')
    X_train, y_train = train['title'].values, train['tags'].values
    X_val, y_val = validation['title'].values, validation['tags'].values
    X_test = test['title'].values
    X_train = [text_prepare(x) for x in X_train]
    X_val = [text_prepare(x) for x in X_val]
    X_test = [text_prepare(x) for x in X_test]
    if data == 1:
        return X_train, y_train, X_val, y_val, X_test

    DICT_SIZE = 5000
    words_counts, tags_counts = get_word_tags_counts(X_train, y_train)
    INDEX_TO_WORDS = sorted(words_counts, key=words_counts.get, reverse=True)[:DICT_SIZE]
    WORDS_TO_INDEX = {word: i for i, word in enumerate(INDEX_TO_WORDS)}
    ALL_WORDS = WORDS_TO_INDEX.keys()
    X_train_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_train])
    X_val_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_val])
    X_test_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_test])

    if data == 2:
        return X_train, y_train, X_val, y_val, X_test, X_train_mybag, X_val_mybag, X_test_mybag

    X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vectorizer = tfidf_features(X_train, X_val, X_test)
    return X_train, y_train, X_val, y_val, X_test, X_train_mybag, X_val_mybag, X_test_mybag, X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vectorizer, tags_counts, WORDS_TO_INDEX, DICT_SIZE


def process_for_inference(data, words_to_index, dict_size, tfidf_vectorizer):
    data = [text_prepare(x) for x in data]
    data_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, words_to_index, dict_size)) for text in data])
    data_tfidf = tfidf_vectorizer.transform(data)
    return data_mybag, data_tfidf
