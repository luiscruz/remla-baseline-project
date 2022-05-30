import pickle
import pytest
import pandas as pd
import tensorflow_data_validation as tfdv


REFRESH_SCHEMAS = False


@pytest.mark.parametrize(
    'data_step, data_set',
    [('raw', 'test'),       ('raw', 'train'),       ('raw', 'validation'),
     ('interim', 'test'),   ('interim', 'train'),   ('interim', 'validation')]
)
def test_schema(data_step, data_set):

    data_path = '../data/{}/{}.tsv'.format(data_step, data_set)
    schema_path = 'schemas/{}/{}_schema.pbtxt'.format(data_step, data_set)

    # infer schema from data
    stats = tfdv.generate_statistics_from_csv(data_path, delimiter='\t')
    inferred_schema = tfdv.infer_schema(stats)

    # load schema from file
    if REFRESH_SCHEMAS:
        tfdv.write_schema_text(inferred_schema, schema_path)
    expected_schema = tfdv.load_schema_text(schema_path)

    # ensure there are no anomalies
    anomalies = tfdv.validate_statistics(stats, expected_schema)
    assert 0 == len(anomalies.anomaly_info)



