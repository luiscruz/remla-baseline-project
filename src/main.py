import pandas as pd

from src.read_data import read_data

if __name__ == '__main__':
    train = read_data('data/train.tsv')
    validation = read_data('data/validation.tsv')
    test = pd.read_csv('data/test.tsv', sep='\t')