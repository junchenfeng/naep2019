import os

import pandas as pd


DATA_PATH = os.path.abspath("data/raw")

DATA_A_TRAIN_FILE_NAME = "data_a_train.csv"
DATA_A_TRAIN_10_FILE_NAME = "data_a_train_10.csv"
DATA_A_TRAIN_20_FILE_NAME = "data_a_train_20.csv"
DATA_A_TRAIN_30_FILE_NAME = "data_a_train_30.csv"
DATA_TRAIN_LABEL_FILE_NAME = "data_train_label.csv"

DATA_A_HIDDEN_10_FILE_NAME = "data_a_hidden_10.csv"
DATA_A_HIDDEN_20_FILE_NAME = "data_a_hidden_20.csv"
DATA_A_HIDDEN_30_FILE_NAME = "data_a_hidden_30.csv"
HIDDEN_LABEL_FILE_NAME = "hidden_label.csv"


def main():
    """
    把原始数据也切割为10，20，30
    """
    df = pd.read_csv(f"{DATA_PATH}/{DATA_A_TRAIN_FILE_NAME}")

    df["EventTime"] = pd.to_datetime(df.EventTime)
    last_entry_df = (
        df.groupby(["STUDENTID"])
        .apply(lambda x: min(x["EventTime"]))
        .reset_index()
        .rename(columns={0: "FirstEntryEventTime"})
    )

    all_df = pd.merge(df, last_entry_df)
    all_df = all_df.assign(Duration=all_df.EventTime - all_df.FirstEntryEventTime)

    all_df["Duration"] = all_df["Duration"].dt.total_seconds()

    all_df[all_df["Duration"] <= 600].to_csv(f"{DATA_PATH}/{DATA_A_TRAIN_10_FILE_NAME}")
    all_df[all_df["Duration"] <= 1200].to_csv(
        f"{DATA_PATH}/{DATA_A_TRAIN_20_FILE_NAME}"
    )
    # 4493 entries exceeding the time limit. Probably retain the intergrity of an item.
    all_df.to_csv(f"{DATA_PATH}/{DATA_A_TRAIN_30_FILE_NAME}")


if __name__ == "__main__":
    main()
