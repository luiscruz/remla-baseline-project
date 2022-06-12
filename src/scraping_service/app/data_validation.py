from typing import Tuple

import pandas as pd


def remove_anomalies(df_scraped: pd.DataFrame) -> Tuple[int, pd.DataFrame]:
    valid_rows = df_scraped["tags"].map(lambda t: len(t)) > 0
    num_invalid_rows = len(df_scraped) - len(valid_rows)
    df_scraped = df_scraped[valid_rows]
    return num_invalid_rows, df_scraped


if __name__ == "__main__":
    df = pd.DataFrame({"title": ["title 1", "title2"], "tags": [[1, 2, 3], [12, 23, 12]]})
    print(df)
    remove_anomalies(df)
