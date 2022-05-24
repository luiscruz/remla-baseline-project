import tensorflow_data_validation as tfdv
import pandas as pd
from IPython.display import display, HTML


validation = pd.read_csv('../data/validation.tsv', sep='\t')
validation_stats = tfdv.generate_statistics_from_dataframe(validation)
#tfdv.visualize_statistics(validation_stats)

schema = tfdv.infer_schema(validation_stats)
#tfdv.display_schema(schema)

train = pd.read_csv('../data/train.tsv', sep='\t')
train_stats = tfdv.generate_statistics_from_dataframe(train)
anomalies = tfdv.validate_statistics(train_stats, schema=schema)
tfdv.display_anomalies(anomalies)

test = pd.read_csv('../data/test.tsv', sep='\t')
schema_for_test = tfdv.infer_schema(tfdv.generate_statistics_from_dataframe(validation.drop(columns=['tags'])))
test_stats = tfdv.generate_statistics_from_dataframe(test)
test_anomalies = tfdv.validate_statistics(test_stats, schema=schema_for_test)
tfdv.display_anomalies(test_anomalies)

