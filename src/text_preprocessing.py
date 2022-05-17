import re
from ast import literal_eval

import joblib
import pandas as pd
from nltk.corpus import stopwords


def read_data(filename):
    data = pd.read_csv(filename, sep='\t')
    data['tags'] = data['tags'].apply(literal_eval)
    return data


def split_data(train, validation, test):
    X_train, y_train = train['title'].values, train['tags'].values
    X_val, y_val = validation['title'].values, validation['tags'].values
    X_test = test['title'].values
    return X_train, y_train, X_val, y_val, X_test


def text_prepare(text):
    """
        text: a string
        
        return: modified initial string
    """
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    STOPWORDS = set(stopwords.words('english'))
    text = text.lower()  # lowercase text
    text = re.sub(REPLACE_BY_SPACE_RE, " ", text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = re.sub(BAD_SYMBOLS_RE, "", text)  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = " ".join([word for word in text.split() if not word in STOPWORDS])  # delete stopwords from text
    return text


def main():
    data_directory = "../data"

    # Read data
    train = read_data(data_directory + '/train.tsv')
    validation = read_data(data_directory + '/validation.tsv')
    test = pd.read_csv(data_directory + '/test.tsv', sep='\t')

    # Split data
    X_train, y_train, X_val, y_val, X_test = split_data(train, validation, test)

    # Prepare tests
    prepared_questions = []
    for line in open(data_directory + '/text_prepare_tests.tsv', encoding='utf-8'):
        line = text_prepare(line.strip())
        prepared_questions.append(line)

    text_prepare_results = '\n'.join(prepared_questions)
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

    joblib.dump((X_train, X_val, X_test), "../output/X_preprocessed.joblib")
    joblib.dump((y_train, y_val), "../output/y_preprocessed.joblib")
    joblib.dump(words_counts, "../output/words_counts.joblib")
    joblib.dump(tags_counts, "../output/tags_counts.joblib")


if __name__ == "__main__":
    main()
