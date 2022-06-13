from typing import Set, Tuple

import pandas as pd


def remove_anomalies(df_scraped: pd.DataFrame, valid_tags: Set[str]) -> Tuple[int, pd.DataFrame]:
    # Only keep rows with tags from the predefined tag list
    # Only keep tags from the predefined tag list
    num_rows_in = len(df_scraped)
    df_scraped.tags = df_scraped.tags.apply(lambda tag_list: [tag for tag in tag_list if tag in valid_tags])
    valid_rows = df_scraped.tags.apply(lambda t: len(t)) > 0
    df_scraped = df_scraped[valid_rows]
    num_invalid_rows = num_rows_in - len(df_scraped)

    return num_invalid_rows, df_scraped
