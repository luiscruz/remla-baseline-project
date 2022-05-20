import unittest
import os.path as path
import tensorflow_data_validation as tfdv


REFRESH_SCHEMAS = False
DATA_LOCATION = '../data'
SCHEMA_LOCATION = 'schemas'

class TestSchemas(unittest.TestCase):
    def setUp(self) -> None:
        self.datasets = [
            'train',
            'test',
            'validation',
            'text_prepare_tests',
        ]

    def test_anomalies(self):
        for dataset in self.datasets:
            with self.subTest(dataset=dataset):
                # get paths to resources
                data_path = path.join(DATA_LOCATION, '{}.tsv'.format(dataset))
                schema_path = path.join(SCHEMA_LOCATION, '{}_schema.pbtxt'.format(dataset))

                # infer schema from data
                stats = tfdv.generate_statistics_from_csv(data_path, delimiter='\t')
                inferred_schema = tfdv.infer_schema(stats)

                # load schema from file
                if REFRESH_SCHEMAS:
                    tfdv.write_schema_text(inferred_schema, schema_path)
                expected_schema = tfdv.load_schema_text(schema_path)

                # ensure there are no anomalies
                anomalies = tfdv.validate_statistics(stats, expected_schema)
                self.assertEqual(0, len(anomalies.anomaly_info))


if __name__ == '__main__':
    unittest.main()
