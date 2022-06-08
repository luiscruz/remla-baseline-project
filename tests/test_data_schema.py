import pickle
import pytest
import pandas as pd
import tensorflow_data_validation as tfdv


REFRESH_SCHEMAS = True


@pytest.mark.parametrize(
    "data_step, data_set",
    [
        # ('raw', 'test'),       ('raw', 'train'),       ('raw', 'validation'),
        ("interim", "test"),
        ("interim", "train"),
        ("interim", "validation"),
    ],
)
def test_schema(data_folder, test_folder, data_step, data_set):
    """test schema"""
    data_path = data_folder / "{}/{}.tsv".format(data_step, data_set)
    schema_path = test_folder / "schemas/{}/{}_schema.pbtxt".format(data_step, data_set)

    # infer schema from data
    stats = tfdv.generate_statistics_from_csv(str(data_path), delimiter="\t")
    inferred_schema = tfdv.infer_schema(stats)

    # load schema from file
    if REFRESH_SCHEMAS:
        schema_path.parent.mkdir(parents=True, exist_ok=True)
        schema_path.touch()
        tfdv.write_schema_text(inferred_schema, str(schema_path))
    expected_schema = tfdv.load_schema_text(str(schema_path))

    # ensure there are no anomalies
    anomalies = tfdv.validate_statistics(stats, expected_schema)
    assert 0 == len(anomalies.anomaly_info)
