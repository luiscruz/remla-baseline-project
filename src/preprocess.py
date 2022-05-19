"""Preprocess script used for reading, preparing and transforming the train, test and validation data before dumping it to the output folder."""

import re
from ast import literal_eval
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from joblib import dump


nltk.download('stopwords')


def read_data(filename):
    """
    Read, store and return the data from the given filename.

    :param filename: filename of where the data is saved.
    :return: the data in the form of a dataframe.
    """
    data = pd.read_csv(filename, sep='\t')
    data['tags'] = data['tags'].apply(literal_eval)
    return data


def text_prepare(text):
    """
    Take the given text as input data, turn it into lowercase letters, replace certain symbols by space, remove bad symbols and stopwords, and return the final text as result.

    :param text: A single record from the input data.
    :return: prepared version of the text.
    """
    REPLACE_BY_SPACE_RE = re.compile(r'[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    STOPWORDS = set(stopwords.words('english'))

    # lowercase text
    text = text.lower()
    # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = re.sub(REPLACE_BY_SPACE_RE, " ", text)
    # delete symbols which are in BAD_SYMBOLS_RE from text
    text = re.sub(BAD_SYMBOLS_RE, "", text)
    # delete stopwords from text
    text = " ".join([word for word in text.split() if word not in STOPWORDS])
    return text


def tfidf_features(X_train, X_val, X_test):
    """
    TF-IDF vectorizer with fixed choices of parameters is used to fit the training data, and to transform the training, validation and test data.

    :param X_train: training set.
    :param X_val: validation set.
    :param X_test: test set.
    :return: transformed train, test and validation set.
    """
    # Create TF-IDF vectorizer with a proper parameters choice
    # Fit the vectorizer on the train set
    # Transform the train, test, and val sets and return the result

    tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2), token_pattern=r'(\S+)')  # nosec B106

    X_train = tfidf_vectorizer.fit_transform(X_train)
    X_val = tfidf_vectorizer.transform(X_val)
    X_test = tfidf_vectorizer.transform(X_test)

    return X_train, X_val, X_test


def main():
    """Is the main function."""
    train = read_data('data/train.tsv')
    validation = read_data('data/validation.tsv')
    test = pd.read_csv('data/test.tsv', sep='\t')

    X_train, y_train = train['title'].values, train['tags'].values
    X_val, y_val = validation['title'].values, validation['tags'].values
    X_test = test['title'].values

    X_train = [text_prepare(x) for x in X_train]
    X_val = [text_prepare(x) for x in X_val]
    X_test = [text_prepare(x) for x in X_test]

    X_train_tfidf, X_val_tfidf, X_test_tfidf = tfidf_features(X_train, X_val, X_test)

    dump((X_train_tfidf, y_train), 'output/train_tfidf.joblib')
    dump((X_val_tfidf, y_val), 'output/validation_tfidf.joblib')
    dump(X_test_tfidf, 'output/test_tfidf.joblib')


if __name__ == "__main__":
    main()
