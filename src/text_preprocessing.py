"""
    Main logic of this stage:
    * reads the data
    * splits the data into features and labels
    * preprocesses the data by:
        * ensuring all lowercase letters
        * replacing some symbols by spaces
        * deleting some symbols
        * deleting stopwords
    * calculating word and tag counts
    * dumps the preprocessed data in a joblib
"""
import re
from ast import literal_eval

import joblib
import pandas as pd
from nltk.corpus import stopwords

data_directory = "data"
output_directory = "output"
feature_list = ['title']
label_list = ['tags']


def read_data(filename):
    """
        filename: string representation of path + file

        reads the data in the file and returns it as a dataframe
    """
    data = pd.read_csv(filename, sep='\t', dtype={'title': 'str', 'tags': 'str'}) # pylint: disable=column-selection-pandas
    # Pylint doesn't recognize that this column selection is valid.
    data = data[feature_list + label_list]
    data['tags'] = data['tags'].apply(literal_eval)
    return data


def split_data(train, validation, test):
    """
        train: training data (dataframe)
        validation: validation data (dataframe)
        test: test data (dataframe)

        Splits the data into features and labels
    """
    X_train, y_train = train['title'].values, train['tags'].values
    X_val, y_val = validation['title'].values, validation['tags'].values
    X_test = test['title'].values
    return X_train, y_train, X_val, y_val, X_test


def text_prepare(text):
    """
        text: a string

        return: modified initial string
    """
    REPLACE_BY_SPACE_RE = re.compile(r'[/(){}[]|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    STOPWORDS = set(stopwords.words('english'))
    text = text.lower()  # lowercase text
    text = re.sub(REPLACE_BY_SPACE_RE, " ", text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = re.sub(BAD_SYMBOLS_RE, "", text)  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = " ".join([word for word in text.split() if not word in STOPWORDS])  # delete stopwords from text
    return text


def main():
    # Read data
    train = read_data(data_directory + '/train.tsv')
    validation = read_data(data_directory + '/validation.tsv')
    test = pd.read_csv(data_directory + '/test.tsv', sep='\t', dtype={'title': 'str'}) # pylint: disable=column-selection-pandas
    # Pylint doesn't recognize that this column selection is valid.
    test = test[feature_list]

    # Split data
    X_train, y_train, X_val, y_val, X_test = split_data(train, validation, test)

    # Prepare tests
    prepared_questions = []
    with open(data_directory + '/text_prepare_tests.tsv', encoding='utf-8') as file:
        for line in file:
            line = text_prepare(line.strip())
            prepared_questions.append(line)

    # text_prepare_results = '\n'.join(prepared_questions)
    X_train = [text_prepare(x) for x in X_train]
    X_val = [text_prepare(x) for x in X_val]
    X_test = [text_prepare(x) for x in X_test]

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

    # print(tags_counts)
    # print(words_counts)
    #
    # print(sorted(words_counts, key=words_counts.get, reverse=True)[:3])
    # most_common_tags = sorted(tags_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    # most_common_words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:3]

    joblib.dump((X_train, X_val, X_test), output_directory + "/X_preprocessed.joblib")
    joblib.dump((y_train, y_val), output_directory + "/y_preprocessed.joblib")
    joblib.dump(words_counts, output_directory + "/words_counts.joblib")
    joblib.dump(tags_counts, output_directory + "/tags_counts.joblib")


if __name__ == "__main__":
    main()
