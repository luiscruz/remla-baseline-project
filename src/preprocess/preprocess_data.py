from ast import literal_eval
import pandas as pd
import re
import sys

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

OUT_PATH_TRAIN = 'data/processed/train_preprocessed.tsv'
OUT_PATH_VAL = 'data/processed/validation_preprocessed.tsv'
OUT_PATH_TEST = 'data/processed/test_preprocessed.tsv'

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


def main():
    if len(sys.argv) != 4:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython src/preprocess/preprocess_data.py train-file-path validation-file-path test-file-path\n")
        sys.exit(1)

    X_train, X_val, X_test, y_train, y_val = init_data(sys.argv[1], sys.argv[2], sys.argv[3])
    X_train, X_val, X_test = preprocess_text_prepare(X_train, X_val, X_test)
    
    train_data_out = pd.DataFrame.from_dict({'X_train': X_train, 'y_train': y_train})
    train_data_out.to_csv(OUT_PATH_TRAIN, sep='\t', index=False)
        
    val_data_out = pd.DataFrame.from_dict({'X_val': X_val, 'y_val': y_val})
    val_data_out.to_csv(OUT_PATH_VAL, sep='\t', index=False)

    test_data_out = pd.DataFrame.from_dict({'X_test': X_test})
    test_data_out.to_csv(OUT_PATH_TEST, sep='\t', index=False)

if __name__ == '__main__':
    main()
    