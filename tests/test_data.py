import pandas as pd
import tensorflow_data_validation as tfdv

from src.preprocess import read_data


def test_tfdv_schema():
    schema_labled = tfdv.load_schema_text("data/tfdv_schema_labled")
    schema_unlabled = tfdv.load_schema_text("data/tfdv_schema_unlabled")

    train = read_data("data/train.tsv")
    train["title"] = train["title"].astype(str)
    train["tags"] = train["tags"].astype(str)

    validation = read_data("data/validation.tsv")
    validation["title"] = validation["title"].astype(str)
    validation["tags"] = validation["tags"].astype(str)

    test = pd.read_csv("data/test.tsv", sep="\t", dtype={"title": str})
    test = test[["title"]]
    test["title"] = test["title"].astype(str)

    train_stats = tfdv.generate_statistics_from_dataframe(train)
    validation_stats = tfdv.generate_statistics_from_dataframe(validation)
    test_stats = tfdv.generate_statistics_from_dataframe(test)

    train_schema = tfdv.infer_schema(train_stats)
    validation_schema = tfdv.infer_schema(validation_stats)
    test_schema = tfdv.infer_schema(test_stats)

    assert train_schema == schema_labled
    assert validation_schema == schema_labled
    assert test_schema == schema_unlabled
