# -*- coding: utf-8 -*-
import os
from ast import literal_eval
import re

import click
import logging
from pathlib import Path
from src.util.util import read_data, write_data

import pandas as pd
from dotenv import find_dotenv, load_dotenv

import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords


def filterDuplicates(X_y_train, X_y_val, X_test):
    sample_set = set()
    text_set = set()

    X_y_train_res = []
    X_y_val_res = []
    X_test_res = []

    for r in X_y_train:
        key = r[0] + str(r[1])
        if key not in sample_set:
            sample_set.add(key)
            text_set.add(r[0])
            X_y_train_res.append(r)

    for r in X_y_val:
        key = r[0] + str(r[1])
        if key not in sample_set:
            sample_set.add(key)
            text_set.add(r[0])
            X_y_val_res.append(r)

    for r in X_test:
        if r not in text_set:
            text_set.add(r)
            X_test_res.append(r)

    return X_y_train_res, X_y_val_res, X_test_res


def main(input_filepath='data/raw/', output_filepath='data/interim/'):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    path = os.getcwd()
    logger.info('working dir: ' + path)

    train_file_name = 'train.tsv'
    validation_file_name = 'validation.tsv'
    test_file_name = 'test.tsv'

    # Load data from tsv files in directory
    train = read_data(input_filepath + train_file_name)
    validation = read_data(input_filepath + validation_file_name)
    test = pd.read_csv(input_filepath + test_file_name, sep='\t')

    # Select columns to use
    X_train, y_train = train['title'].values, train['tags'].values
    X_val, y_val = validation['title'].values, validation['tags'].values
    X_test = test['title'].values

    # Preprocess data
    X_train = [text_prepare(x) for x in X_train]
    X_val = [text_prepare(x) for x in X_val]
    X_test = [text_prepare(x) for x in X_test]

    X_y_train = list(zip(X_train, y_train))
    X_y_val = list(zip(X_val, y_val))
    X_y_train, X_y_val, X_test = filterDuplicates(X_y_train, X_y_val, X_test)

    #  Lists to pd for easy writing
    train_out = pd.DataFrame(X_y_train,
                             columns=['title', 'tags'])
    val_out = pd.DataFrame(X_y_val,
                           columns=['title', 'tags'])
    test_out = pd.DataFrame(X_test, columns=['title'])

    # Write data to files
    write_data(train_out, output_filepath + train_file_name)
    write_data(val_out, output_filepath + validation_file_name)
    write_data(test_out, output_filepath + test_file_name)


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))


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


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
