from joblib import dump, load
import numpy as np
import pandas as pd
from ast import literal_eval
from nltk.corpus import stopwords
import sys
import nltk
import re
nltk.download('stopwords')

# Do we need the validation really?

# RegEx expressions to clean the data
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


def main():
    # Read the data to be used in the project
    train = read_data('data/train.tsv')
    validation = read_data('data/validation.tsv')
    test = pd.read_csv('data/test.tsv', sep='\t')

    # Separate trainning and validation
    X_train, y_train = train['title'].values, train['tags'].values
    X_val, y_val = validation['title'].values, validation['tags'].values
    X_test = test['title'].values

    # Clean all the data
    prepared_questions = []
    cont = 0
    test_texts = []
    test_outs = []
    for line in open('data/text_prepare_tests.tsv', encoding='utf-8'):
        test_texts.append(line)
        line = text_prepare(line.strip())
        prepared_questions.append(line)

        test_outs.append(line)
        cont += 1

    text_prepare_results = '\n'.join(prepared_questions)

    # Retrieve preprocesed data
    X_train = [text_prepare(x) for x in X_train]
    X_val = [text_prepare(x) for x in X_val]
    X_test = [text_prepare(x) for x in X_test]

    preprocessed_data = {"X_train": X_train, "X_val": X_val, "X_test": X_test, "y_train": y_train, "y_val": y_val}

    dump(preprocessed_data, 'output/preprocessed_data.joblib')
    return preprocessed_data


if __name__ == "__main__":
    main()
