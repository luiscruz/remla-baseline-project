# -*- coding: utf-8 -*-
import os
from ast import literal_eval
import re

import click
import logging
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.util.util import read_data, write_data
from os import listdir
from os.path import isfile, join

import pandas as pd
from dotenv import find_dotenv, load_dotenv

import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

DATA_WINDOW_SIZE = 3


def main(input_filepath='data/raw/', output_filepath='data/interim/'):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making interim data set from raw data')

    train_file_name = 'train.tsv'
    validation_file_name = 'validation.tsv'
    test_file_name = 'test.tsv'

    # # Load data from tsv files in directory
    # train = read_data(input_filepath + train_file_name)
    # validation = read_data(input_filepath + validation_file_name)
    # test = pd.read_csv(input_filepath + test_file_name, sep='\t')

    onlyfiles = [join(input_filepath, f) for f in listdir(input_filepath) if isfile(join(input_filepath, f))]
    all_data = pd.DataFrame(columns=['title','tags'])
    file_count = 0
    for f in onlyfiles:
        if file_count >= DATA_WINDOW_SIZE:
            break
        if ".tsv" not in f:
            continue
        logger.info('Reading from file: ' + f)
        file_data = read_data(f)
        all_data = pd.concat([all_data, file_data])
        file_count = file_count + 1

    X_train, X_test, y_train, y_test = train_test_split(all_data['title'], all_data['tags'], test_size=0.1,
                                                      random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25,
                                                      random_state=1)

    # Select columns to use
    # X_train, y_train = train['title'].values, train['tags'].values
    # X_val, y_val = validation['title'].values, validation['tags'].values
    # X_test = test['title'].values

    # Preprocess data
    X_train = [text_prepare(x) for x in X_train.values]
    X_val = [text_prepare(x) for x in X_val.values]
    X_test = [text_prepare(x) for x in X_test.values]

    #  Lists to pd for easy writing
    train_out = pd.DataFrame(list(zip(X_train, y_train)),
                             columns=['title', 'tags'])
    val_out = pd.DataFrame(list(zip(X_val, y_val)),
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
