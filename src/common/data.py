"""Common functions handling data."""
from ast import literal_eval

import pandas as pd


def read_data(infile):
    data = pd.read_csv(infile, sep='\t').astype("string")
    data['tags'] = data['tags'].apply(literal_eval)
    return data
