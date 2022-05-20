import unittest
import os.path as path
import tensorflow_data_validation as tfdv


REFRESH_SCHEMAS = False


class TestSchemas(unittest.TestCase):

    def test_anomalies(self):
        datasets = [
            'train',
            'test',
            'validation',
            'text_prepare_tests',
        ]
        for dataset in datasets:
            with self.subTest(dataset=dataset):
                # get paths to resources
                data_path = '../data/{}.tsv'.format(dataset)
                schema_path = 'schemas/{}_schema.pbtxt'.format(dataset)

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
