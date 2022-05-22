from collections import defaultdict
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


def detect_anomalies():
    train_df = pd.read_csv(f'../data/train.tsv', sep="\t")
    val_df = pd.read_csv(f'../data/validation.tsv', sep="\t")

    train_stats = tfdv.generate_statistics_from_dataframe(train_df)
    val_stats = tfdv.generate_statistics_from_dataframe(val_df)

    schema = tfdv.infer_schema(train_stats)

    anomalies = tfdv.validate_statistics(val_stats, schema=schema)
    anomalies_detected = defaultdict(list)

    for info in anomalies.anomaly_info.items():
        affected_area, problems = read_info_anomaly(info)
        anomalies_detected[affected_area].append(problems)

    valid_data = len(anomalies_detected.keys()) == 0

    if not valid_data:
        # What do we do with this?
        data_anomalies = json.dumps(anomalies_detected, indent=4)

    return valid_data
