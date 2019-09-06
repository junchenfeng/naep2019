import os

import pandas as pd

from constant import (
    INDEX_VAR,
    EVENT_TIME,
    DURATION,
    DATA_PATH,
    DATA_A_TRAIN_10_FILE_NAME,
    DATA_A_TRAIN_20_FILE_NAME,
    DATA_A_TRAIN_30_FILE_NAME,
)


DATA_A_TRAIN_FILE_NAME = "data_a_train.csv"


def main():
    """
    把原始数据也切割为10，20，30
    """
    df = pd.read_csv(f"{DATA_PATH}/raw/{DATA_A_TRAIN_FILE_NAME}")

    df[EVENT_TIME] = pd.to_datetime(df.EventTime)
    last_entry_df = (
        df.groupby([INDEX_VAR])
        .apply(lambda x: min(x[EVENT_TIME]))
        .reset_index()
        .rename(columns={0: "FirstEntryEventTime"})
    )

    all_df = pd.merge(df, last_entry_df)
    all_df = all_df.assign(Duration=all_df.EventTime - all_df.FirstEntryEventTime)

    all_df[DURATION] = all_df[DURATION].dt.total_seconds()

    all_df[all_df[DURATION] <= 600].to_csv(f"{DATA_PATH}/raw/{DATA_A_TRAIN_10_FILE_NAME}")
    all_df[all_df[DURATION] <= 1200].to_csv(
        f"{DATA_PATH}/raw/{DATA_A_TRAIN_20_FILE_NAME}"
    )
    # 4493 entries exceeding the time limit. Probably retain the integrity of an item.
    all_df.to_csv(f"{DATA_PATH}/raw/{DATA_A_TRAIN_30_FILE_NAME}")


if __name__ == "__main__":
    main()
