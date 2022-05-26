from ast import literal_eval
import re

import pandas as pd

import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

REPLACE_BY_SPACE_RE = re.compile("[/(){}\[\]\|@,;]")
BAD_SYMBOLS_RE = re.compile("[^0-9a-z #+_]")
STOP_WORDS = set(stopwords.words("english"))


def read_data(filename: str):
    data = pd.read_csv(filename, sep="\t")
    data["tags"] = data["tags"].apply(literal_eval)

    return data


def text_prepare(text: str):
    text = text.lower()
    text = re.sub(REPLACE_BY_SPACE_RE, " ", text)
    text = re.sub(BAD_SYMBOLS_RE, "", text)
    text = " ".join([word for word in text.split() if not word in STOP_WORDS])

    return text
