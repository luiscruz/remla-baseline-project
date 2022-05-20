"""Common functions handling data."""
from ast import literal_eval

import pandas as pd


def read_data(infile):
    """Create dataframe from csv file."""
    data = pd.read_csv(infile, sep='\t', dtype="string")
    data['tags'] = data['tags'].apply(literal_eval)
    return data
