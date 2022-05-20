"""Common functions handling data."""
from ast import literal_eval

import pandas as pd


def read_data(infile, dtype):
    """Create dataframe from csv file."""
    data = pd.read_csv(infile, sep='\t', dtype=dtype)
    data['tags'] = data['tags'].apply(literal_eval)
    return data
