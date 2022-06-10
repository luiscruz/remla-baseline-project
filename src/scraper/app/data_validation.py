import os
from typing import Tuple

import pandas as pd
import tensorflow_data_validation as tfdv
from tensorflow_data_validation.utils.display_util import get_anomalies_dataframe

train_stats = tfdv.generate_statistics_from_dataframe(pd.read_csv(f"{os.environ['DATA_DIR']}/train.tsv", sep="\t"))
schema = tfdv.infer_schema(train_stats)


def remove_anomalies(df_scraped: pd.DataFrame) -> Tuple[int, pd.DataFrame]:
    print(f"df_scraped type {type(df_scraped)}")
    print(f"df_scraped shape {df_scraped.shape}")
    df_temp = df_scraped.copy()
    df_temp["tags"] = df_temp.tags.astype(str)
    test_stats = tfdv.generate_statistics_from_dataframe(df_temp)
    anomalies = tfdv.validate_statistics(statistics=test_stats, schema=schema)
    df_anomalies = get_anomalies_dataframe(anomalies)

    # In case no anomalies have been found.
    if df_anomalies.empty:
        return 0, df_scraped
    else:
        unresolved_anomalies = 0
        for index, row in df_anomalies.iterrows():
            column_name = index.__str__().replace("'", "")
            error_message = row[1]
            print(f"Feature: {column_name} has anomaly: '{error_message}'")

            # Check for the case of additional columns besides question and tags being stored.
            if "New column" in error_message:
                df_scraped = df_scraped.drop([column_name], axis=1)
            else:
                # For returning anomalies that could not be resolved like questions or tags missing.
                unresolved_anomalies += 1

        return unresolved_anomalies, df_scraped


if __name__ == "__main__":
    df = pd.DataFrame({"title": ["title 1", "title2"], "tags": [[1, 2, 3], [12, 23, 12]]})
    print(df)
    remove_anomalies(df)
