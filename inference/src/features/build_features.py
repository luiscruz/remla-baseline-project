from ast import literal_eval
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import re 
from scipy import sparse
from data.preprocess import train_tfidf_vectorizer, preprocess_sentences

import pickle

def read_data(filename):
    data = pd.read_csv(filename, sep='\t')
    data['tags'] = data['tags'].apply(literal_eval)
    return data

def count_tags_and_words(X_train, y_train):
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
    return (tags_counts, words_counts)

def preprocess_data(input_dir, output_dir):
    train = read_data(input_dir + '/train.tsv')
    validation = read_data(input_dir + '/validation.tsv')
    test = pd.read_csv(input_dir + '/test.tsv', sep='\t')

    X_train, y_train = train['title'].values, train['tags'].values
    X_val, y_val = validation['title'].values, validation['tags'].values
    X_test = test['title'].values

    (tags_counts, word_counts) = count_tags_and_words(X_train, y_train)

    vectorizer = train_tfidf_vectorizer(X_train)
    vocab = vectorizer.vocabulary_

    X_train = preprocess_sentences(X_train, vectorizer)
    X_val = preprocess_sentences(X_val, vectorizer)
    X_test = preprocess_sentences(X_test, vectorizer)

    mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
    y_train = mlb.fit_transform(y_train)
    y_val = mlb.fit_transform(y_val)

    np.savetxt(output_dir + '/tags.txt', sorted(tags_counts.keys()), fmt='%s')
    sparse.save_npz(output_dir + '/X_train.npz', X_train)
    sparse.save_npz(output_dir + '/X_val.npz', X_val)
    sparse.save_npz(output_dir + '/X_test.npz', X_test)

    with open("./models/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    
    with open(f'{output_dir}/y_train.npy', 'wb') as f:
        np.save(f, y_train)
    
    with open(f'{output_dir}/y_val.npy', 'wb') as f:
        np.save(f, y_val)

if __name__ == "__main__":
    # execute only if run as the entry point into the program
    preprocess_data(input_dir="./data/raw", output_dir="./data/processed")