import os

import pandas
import pandas as pd
import pytest
from src.util import util

from os import listdir
from os.path import isfile, join

from util.util import read_data


@pytest.fixture()
def all_data():
    all_data = []
    input_filepath = '../data/interim/'
    directory = os.getcwd()

    print('########', directory)
    onlyfiles = [join(input_filepath, f) for f in listdir(input_filepath) if isfile(join(input_filepath, f))]

    for f in onlyfiles:
        if ".tsv" not in f:
            continue
        file_data = read_data(f)
        all_data.append(file_data)

    yield all_data


def test_no_duplicates(all_data):
    for df in all_data:
        if "tags" not in df.columns:
            df["tags"] = " "
        df['key'] = df['title'] + df['tags'].apply(lambda r: str(r))
        print('ROW: ', df.groupby(['key']).size().sort_values())
        assert df.groupby(['key']).size().max() == 1




def test_no_duplicates_cross_dataset(all_data):
    df_cross = pd.DataFrame(columns=['title', 'tags'])
    for df in all_data:
        if "tags" not in df.columns:
            df["tags"] = " "
        df_cross = pd.concat([df_cross, df])

    df_cross['key'] = df_cross['title'] + df_cross['tags'].apply(lambda r: str(r))
    print('ROW: ', df_cross.groupby(['key']).size().sort_values())
    assert df_cross.groupby(['key']).size().max() == 1


def test_no_empty_tag(all_data):
    for df in all_data:
        if 'tags' not in df.columns:
            continue
        df['len'] = df['tags'].map(len)
        assert df['len'].min() >= 1


def test_no_empty_text(all_data):
    for df in all_data:
        df['len'] = df['title'].map(len)
        assert df['len'].min() >= 1
