"""
Module for handling feature building.
"""
import pickle
import re

import pandas as pd
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

from src.config.definitions import ROOT_DIR
from src.common.data import read_data

nltk.data.path.append((ROOT_DIR / 'data/external').as_posix()) # specify path to nltk data


def text_prepare(text):
    """
        text: a string
        return: modified initial string
    """
    text = text.lower() # lowercase text
    text = re.sub(REPLACE_BY_SPACE_RE, " ", text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = re.sub(BAD_SYMBOLS_RE, "", text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = " ".join([word for word in text.split() if not word in STOPWORDS]) # delete stopwords from text
    return text


def tfidf_features(X_train, X_val, X_test):
    """
        X_train, X_val, X_test — samples
        return TF-IDF vectorized representation of each sample and vocabulary
    """
    # Create TF-IDF vectorizer with a proper parameters choice
    # Fit the vectorizer on the train set
    # Transform the train, test, and val sets and return the result
    tfidf_vectorizer = TfidfVectorizer(
        min_df=5, max_df=0.9, ngram_range=(1,2), token_pattern=r'(\S+)') ####### YOUR CODE HERE #######
    X_train = tfidf_vectorizer.fit_transform(X_train)
    X_val = tfidf_vectorizer.transform(X_val)
    X_test = tfidf_vectorizer.transform(X_test)

    return X_train, X_val, X_test, tfidf_vectorizer.vocabulary_


if __name__ == '__main__':
    train = read_data(ROOT_DIR / 'data/raw/train.tsv')
    validation = read_data(ROOT_DIR / 'data/raw/validation.tsv')
    test = pd.read_csv(ROOT_DIR / 'data/raw/test.tsv', sep='\t')

    X_train, y_train = train['title'].values, train['tags'].values
    X_val, y_val = validation['title'].values, validation['tags'].values
    X_test = test['title'].values

    REPLACE_BY_SPACE_RE = re.compile(r'[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    STOPWORDS = set(stopwords.words('english'))

    # remove bad symbols
    X_train = [text_prepare(x) for x in X_train]
    X_val = [text_prepare(x) for x in X_val]
    X_test = [text_prepare(x) for x in X_test]

    X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vocab = tfidf_features(X_train, X_val, X_test)

    with open(ROOT_DIR / 'data/processed/train.pkl', 'wb') as f:
        pickle.dump((X_train_tfidf, y_train), f)
    with open(ROOT_DIR / 'data/processed/validation.pkl', 'wb') as f:
        pickle.dump((X_val_tfidf, y_val), f)
    with open(ROOT_DIR / 'data/processed/test.pkl', 'wb') as f:
        pickle.dump(X_test_tfidf, f)
