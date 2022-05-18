from ast import literal_eval
import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

def read_data(filename):
    data = pd.read_csv(filename, sep='\t')
    data['tags'] = data['tags'].apply(literal_eval)
    return data

def split_data():
    train = read_data('../../data/train.tsv')
    validation = read_data('../../data/validation.tsv')
    test = pd.read_csv('../../data/test.tsv', sep='\t')
    return train, validation, test

def init_data():
    train, validation, test = split_data()
    X_train, y_train = train['title'].values, train['tags'].values
    X_val, y_val = validation['title'].values, validation['tags'].values
    X_test = test['title'].values
    return X_train, X_val, X_test, y_train, y_val
