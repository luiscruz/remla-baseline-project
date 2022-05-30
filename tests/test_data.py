import pandas
import pytest
from util import util


@pytest.fixture()
def df1():
    df1 = util.read_data('../data/interim/test.tsv')
    yield df1


@pytest.fixture()
def df2():
    df2 = util.read_data('../data/interim/train.tsv')
    yield df2


@pytest.fixture()
def df3():
    df3 = util.read_data('../data/interim/validation.tsv')
    yield df3


@pytest.fixture()
def df_cross(df1, df2, df3):
    df1['tags'] = df1['title'].apply(lambda _ : [])
    df_cross = df1.append(df2).append(df3)
    yield df_cross


def test_no_duplicates1(df1):
    assert df1.groupby(['title']).size().max() == 1


def test_no_duplicates2(df2):
    df2['key'] = df2['title'] + df2['tags'].apply(lambda r: str(r))

    assert df2.groupby(['key']).size().max() == 1


def test_no_duplicates3(df3):
    df3['key'] = df3['title'] + df3['tags'].apply(lambda r: str(r))
    assert df3.groupby(['key']).size().max() == 1


def test_no_duplicates_cross_dataset(df_cross):
    df_cross['key'] = df_cross['title'] + df_cross['tags'].apply(lambda r: str(r))
    print('ROW: ', df_cross.groupby(['key']).size().sort_values())
    assert df_cross.groupby(['key']).size().max() == 1


def test_no_empty_tag1(df2):
    df2['len'] = df2['tags'].map(len)
    assert df2['len'].min() >= 1


def test_no_empty_tag2(df3):
    df3['len'] = df3['tags'].map(len)
    assert df3['len'].min() >= 1


def test_no_empty_text1(df1):
    df1['len'] = df1['title'].map(len)
    assert df1['len'].min() >= 1


def test_no_empty_text2(df2):
    df2['len'] = df2['title'].map(len)
    assert df2['len'].min() >= 1


def test_no_empty_text3(df3):
    df3['len'] = df3['title'].map(len)
    assert df3['len'].min() >= 1
