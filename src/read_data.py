from ast import literal_eval

import pandas as pd


def read_data(filename):
    data = pd.read_csv(filename, sep='\t')
    data['tags'] = data['tags'].apply(literal_eval)
    return data


if __name__ == '__main__':
    train = read_data('../data/train.tsv')
    validation = read_data('../data/validation.tsv')
    test = pd.read_csv('../data/test.tsv', sep='\t')

    X_train, y_train = train['title'].values, train['tags'].values
    X_val, y_val = validation['title'].values, validation['tags'].values
    # CODE SMELL DON"T USE VALUES
    # TODO:CHANGE LATER
    X_test = test['title'].values

    print(train.head())
