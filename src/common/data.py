"""Common functions handling data."""
from ast import literal_eval

import pandas as pd


def read_data(infile, *args, **kwargs):
    """Create dataframe from csv file."""
    # Disable pylint false positive
    # pylint: disable=column-selection-pandas,datatype-pandas
    data = pd.read_csv(infile, *args, sep='\t', **kwargs)
    data['tags'] = data['tags'].apply(literal_eval)
    return data
