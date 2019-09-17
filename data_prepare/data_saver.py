import os
import pandas as pd
from data_prepare.data_importer import DataImporter
from data_prepare.data_encoder import (
    DataOriginalLabelEncoder,
    DataObservationWeightEncoder,
    FEATURE_LIST,
    WEIGHT_LIST,
    DURATION_LIST,
)

SAVE_DATA_PATH = os.path.abspath("data/mid/")

TRAIN = "train"
HIDDEN = "hidden"

MINUTE_10 = "10"
MINUTE_20 = "20"
MINUTE_30 = "30"


class DataSaver(object):
    @classmethod
    def save_train_and_hidden(cls, train, hidden, time_limit_prefix):
        train_list, hidden_list, dim_list = DataOriginalLabelEncoder.transform_train_and_hidden(
            train, hidden
        )
        cls.save_dim_list(dim_list + [6], time_limit_prefix)
        cls.save_df_list(train_list, TRAIN, time_limit_prefix, FEATURE_LIST)
        cls.save_df_list(hidden_list, HIDDEN, time_limit_prefix, FEATURE_LIST)

    @classmethod
    def save_train_and_hidden_weight(cls, train, hidden, time_limit_prefix):
        train_list, hidden_list = DataObservationWeightEncoder.transform_train_and_hidden(
            train, hidden
        )
        cls.save_df_list(
            train_list, f"{TRAIN}", time_limit_prefix, WEIGHT_LIST + DURATION_LIST
        )
        cls.save_df_list(
            hidden_list, f"{HIDDEN}", time_limit_prefix, WEIGHT_LIST + DURATION_LIST
        )

    @classmethod
    def save_df_list(cls, df_list, train_hidden_prefix, time_limit_prefix, name_list):
        for df_index, df_value in enumerate(df_list):
            df_value.to_csv(
                os.path.join(
                    SAVE_DATA_PATH,
                    f"{train_hidden_prefix}_{time_limit_prefix}_{name_list[df_index]}.csv",
                )
            )

    @classmethod
    def save_dim_list(cls, dim_list, time_limit_prefix):
        dim_df = pd.DataFrame(
            {"feature_type": FEATURE_LIST + WEIGHT_LIST, "dim": dim_list}
        )
        dim_df.to_csv(os.path.join(SAVE_DATA_PATH, f"{time_limit_prefix}_dim.csv"))


if __name__ == "__main__":
    test_train = DataImporter().data_a_train_10
    test_hidden = DataImporter().data_a_hidden_10
    DataSaver.save_train_and_hidden_weight(test_train, test_hidden, MINUTE_10)
