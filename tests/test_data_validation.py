import tensorflow_data_validation as tfdv
import pandas as pd
from tensorflow_data_validation.utils.schema_util import write_schema_text, load_schema_text

def test_data_validation():
    train = pd.read_csv('data/raw/train.tsv', sep="\t")
    test = pd.read_csv('data/raw/test.tsv', sep="\t")
    validation = pd.read_csv('data/raw/validation.tsv', sep="\t")

    train_stats = tfdv.generate_statistics_from_dataframe(train)
    test_stats = tfdv.generate_statistics_from_dataframe(test)
    validation_stats = tfdv.generate_statistics_from_dataframe(validation)

    #schema = tfdv.infer_schema(train_stats)
    #schema.default_environment[:] = ['TRAINING', 'TESTING', 'VALIDATION']
    ## Specify that 'tags' feature is not in TESTING environment.
    #tfdv.get_feature(schema, 'tags').not_in_environment[:] = ['TESTING']
    #write_schema_text(schema, "data/raw/schema")
    schema = load_schema_text("data/raw/schema")

    test_anomalies = tfdv.validate_statistics(
        test_stats, schema, environment='TESTING')
    validation_anomalies = tfdv.validate_statistics(validation_stats, schema=schema)

    failing_test_anomalies = tfdv.validate_statistics(test_stats, schema=schema)

    # anomaly_info is empty/falsy if there were no anomalies detected
    assert(not test_anomalies.anomaly_info)
    assert(not validation_anomalies.anomaly_info)
