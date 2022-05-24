from collections import defaultdict
from requests import get
import tensorflow_data_validation as tfdv
import pandas as pd
import json


def read_info_anomaly(anomaly):
    affected_area = anomaly[0]

    reasons = anomaly[1].reason

    problems = []
    for reason in reasons:
        short_desc = reason.short_description
        long_desc = reason.description

        problems.append({"ShortDescription": short_desc, "LongDescription": long_desc})

    return affected_area, problems


def get_anomalies_detected(anomalies):
    anomalies_detected = defaultdict(list)

    for info in anomalies.anomaly_info.items():
        affected_area, problems = read_info_anomaly(info)
        anomalies_detected[affected_area].append(problems)

    return anomalies_detected


def dump_anomalies(anomalies_detected, option):
    file_anomalies = f'../validation/{option}.json'
    with open(file_anomalies, "w") as file_write:
        file_write.write(json.dumps(anomalies_detected, indent=4))


def detect_anomalies():
    train_df = pd.read_csv(f'data/train.tsv', sep="\t")
    val_df = pd.read_csv(f'data/validation.tsv', sep="\t")
    test_df = pd.read_csv(f'data/validation.tsv', sep="\t")

    valid_data = True

    # For the validation set

    train_stats = tfdv.generate_statistics_from_dataframe(train_df)
    val_stats = tfdv.generate_statistics_from_dataframe(val_df)

    schema = tfdv.infer_schema(train_stats)

    anomalies = tfdv.validate_statistics(val_stats, schema=schema)

    val_anomalies_detected = get_anomalies_detected(anomalies)

    if len(val_anomalies_detected.keys()) > 0:
        valid_data = False
        dump_anomalies(val_anomalies_detected)

    # For the test set

    train_df = train_df.drop(columns=['tags'])

    train_stats = tfdv.generate_statistics_from_dataframe(train_df)
    test_stats = tfdv.generate_statistics_from_dataframe(test_df)
    anomalies = tfdv.validate_statistics(test_stats, schema=schema)

    schema = tfdv.infer_schema(train_stats)

    test_anomalies_detected = get_anomalies_detected(anomalies)

    if len(test_anomalies_detected.keys()) > 0:
        valid_data = False
        dump_anomalies(test_anomalies_detected)

    return valid_data
