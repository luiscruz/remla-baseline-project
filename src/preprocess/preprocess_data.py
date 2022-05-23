from ast import literal_eval
import re
import yaml
import pickle
import pandas as pd

# For this project we will need to use a list of stop words. It can be downloaded from nltk:
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

# One of the most known difficulties when working with natural data is that it's unstructured.
# For example, if you use it "as is" and extract tokens just by splitting the titles by whitespaces,
# you will see that there are many "weird" tokens like *3.5?*, *"Flip*, etc.
# To prevent the problems, it's usually useful to prepare the data somehow.
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

# Fetch params from yaml params file
params = yaml.safe_load(open("params.yaml"))['preprocess']

INPUT_TRAIN_PATH =  params['input_train']
INPUT_VAL_PATH = params['input_val']
INPUT_TEST_PATH = params['input_test']

OUT_PATH_TRAIN = params['output_train']
OUT_PATH_VAL = params['output_val']
OUT_PATH_TEST = params['output_test']

"""
In this task you will deal with a dataset of post titles from StackOverflow. 
You are provided a split to 3 sets: train, validation and test. All corpora (except for test) contain titles of the posts and 
corresponding tags (100 tags are available). The test set doesn’t contain answers.
"""


def _read_data(filename):
    """
    filename: Input filename

    :return: pandas dataframe of the file
    """
    data = pd.read_csv(filename, sep='\t')
    data['tags'] = data['tags'].apply(literal_eval)
    return data


def _split_data(path_to_train, path_to_val, path_to_test):
    """
    :return: Dataframe split into train, validation and test set.
    """
    train = _read_data(path_to_train)
    validation = _read_data(path_to_val)
    test = pd.read_csv(path_to_test, sep='\t')
    return train, validation, test


def init_data(path_to_train, path_to_val, path_to_test):
    """
    :return: For a more comfortable usage, returns an initialized X_train, X_val, X_test, y_train, y_val.
    """
    train, validation, test = _split_data(path_to_train, path_to_val, path_to_test)
    X_train, y_train = train['title'].values, train['tags'].values
    X_val, y_val = validation['title'].values, validation['tags'].values
    X_test = test['title'].values
    return X_train, X_val, X_test, y_train, y_val


def text_prepare(text):
    """
    text: a string

    :return: modified initial string
    """
    text = text.lower()  # lowercase text
    text = re.sub(REPLACE_BY_SPACE_RE, " ", text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = re.sub(BAD_SYMBOLS_RE, "", text)  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = " ".join([word for word in text.split() if not word in STOPWORDS])  # delete stopwords from text
    return text


def preprocess_text_prepare(X_train, X_val, X_test):
    """

    :param X_train:
    :param X_val:
    :param X_test:
    :return:
    """

    X_train = [text_prepare(x) for x in X_train]
    X_val = [text_prepare(x) for x in X_val]
    X_test = [text_prepare(x) for x in X_test]
    return X_train, X_val, X_test

def pickle_train_val_data(X_data, y_data, out_path):
    with open(out_path,'wb') as fd:
        pickle.dump((X_data, y_data), fd,protocol=pickle.HIGHEST_PROTOCOL)

def pickle_test_data(X_data, out_path):
    with open(out_path,'wb') as fd:
        pickle.dump(X_data, fd, protocol=pickle.HIGHEST_PROTOCOL)

def main():
    X_train, X_val, X_test, y_train, y_val = init_data(
        INPUT_TRAIN_PATH,
        INPUT_VAL_PATH,
        INPUT_TEST_PATH
    )

    X_train, X_val, X_test = preprocess_text_prepare(X_train, X_val, X_test)
    
    pickle_train_val_data(X_train, y_train, OUT_PATH_TRAIN)
    pickle_train_val_data(X_val, y_val, OUT_PATH_VAL)
    pickle_test_data(X_test, OUT_PATH_TEST)

if __name__ == '__main__':
    main()
    