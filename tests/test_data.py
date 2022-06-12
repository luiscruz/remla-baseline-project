import os
from os import listdir
from os.path import isfile, join

import pandas
import pandas as pd
import pytest

from src.util.util import read_data


@pytest.fixture()
def all_data(data_folder):
    all_data = []
    directory = os.getcwd()
    folder = data_folder / "interim"

    print("########", directory)
    onlyfiles = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]

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
        df["key"] = df["title"] + df["tags"].apply(lambda r: str(r))
        print("ROW: ", df.groupby(["key"]).size().sort_values())
        assert df.groupby(["key"]).size().max() == 1


def test_no_duplicates_cross_dataset(all_data):
    df_cross = pd.DataFrame(columns=["title", "tags"])
    for df in all_data:
        if "tags" not in df.columns:
            df["tags"] = " "
        df_cross = pd.concat([df_cross, df])

    df_cross["key"] = df_cross["title"] + df_cross["tags"].apply(lambda r: str(r))
    print("ROW: ", df_cross.groupby(["key"]).size().sort_values())
    assert df_cross.groupby(["key"]).size().max() == 1


def test_no_empty_tag(all_data):
    for df in all_data:
        if "tags" not in df.columns:
            continue
        df["len"] = df["tags"].map(len)
        assert df["len"].min() >= 1


def test_no_empty_text(all_data):
    for df in all_data:
        df["len"] = df["title"].map(len)
        assert df["len"].min() >= 1
