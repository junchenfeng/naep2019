import os
from data_prepare.data_importer import DataImporter
from data_prepare.data_encoder import DataEncoder, FEATURE_LIST

SAVE_DATA_PATH = os.path.abspath("data/mid/")

TRAIN = "train"
HIDDEN = "hidden"

MINUTE_10 = "10"
MINUTE_20 = "20"
MINUTE_30 = "30"


class DataSaver(object):
    @classmethod
    def save_train_and_hidden(
        cls, train, hidden, time_limit_prefix
    ):
        train_list, hidden_list = DataEncoder.transform_train_and_hidden(train, hidden)
        cls.save_df_list(train_list, TRAIN, time_limit_prefix)
        cls.save_df_list(hidden_list, HIDDEN, time_limit_prefix)

    @classmethod
    def save_df_list(cls, df_list, train_hidden_prefix, time_limit_prefix):
        for df_index, df_value in enumerate(df_list):
            df_value.to_csv(
                os.path.join(
                    SAVE_DATA_PATH,
                    f"{train_hidden_prefix}_{time_limit_prefix}_{FEATURE_LIST[df_index]}.csv",
                )
            )


if __name__ == "__main__":
    test_train = DataImporter().data_a_train_10
    test_hidden = DataImporter().data_a_hidden_10
    DataSaver.save_train_and_hidden(test_train, test_hidden, MINUTE_10)
