from ast import literal_eval

import pandas as pd


def read_data(filename, delim='\t'):
    data = pd.read_csv(filename, sep=delim)
    if 'tags' in data.columns:
        data['tags'] = data['tags'].apply(literal_eval)
    return data


def write_data(data, filename, delim='\t'):
    data.to_csv(filename, sep=delim)
    return True
