from ast import literal_eval
from pathlib import Path
from typing import Union

import pandas as pd


def read_data(filename: Union[str, Path], delim: str = "\t"):
    data = pd.read_csv(str(filename), sep=delim)
    if "tags" in data.columns:
        data["tags"] = data["tags"].apply(literal_eval)
    return data


def write_data(data, filename: Union[str, Path], delim="\t"):
    data.to_csv(str(filename), sep=delim)
    return True
