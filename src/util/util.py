from ast import literal_eval

import pandas as pd


def read_data(filename):
    data = pd.read_csv(filename, sep='\t')
    data['tags'] = data['tags'].apply(literal_eval)
    return data

def write_data(data, filename):
    data.to_csv(filename)
    return True
