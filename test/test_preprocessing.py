import pandas as pd
from src.model import preprocessing

def test_read_file():
    train = preprocessing.read_data('./data/train.tsv')
    validation = preprocessing.read_data('data/validation.tsv')
    test = pd.read_csv('data/test.tsv', sep='\t')

    assert train.empty is False
    assert validation.empty is False
    assert test.empty is False


