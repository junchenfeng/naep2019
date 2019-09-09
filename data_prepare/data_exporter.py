import os
import pandas as pd
from data_prepare.data_importer import DataImporter
from data_prepare.data_cleaner import STUDENTID
from data_prepare.data_saver import (
    DataSaver,
    MINUTE_10,
    MINUTE_20,
    MINUTE_30,
    TRAIN,
    HIDDEN,
    FEATURE_LIST,
    WEIGHT_LIST,
    SAVE_DATA_PATH,
)

MINUTE_10_DATA = (
    DataImporter().data_a_train_10,
    DataImporter().data_a_hidden_10,
    MINUTE_10,
)
MINUTE_20_DATA = (
    DataImporter().data_a_train_20,
    DataImporter().data_a_hidden_20,
    MINUTE_20,
)
MINUTE_30_DATA = (
    DataImporter().data_a_train_30,
    DataImporter().data_a_hidden_30,
    MINUTE_30,
)


class DataExporter(object):
    @classmethod
    def save_all(cls):
        """
        线运行这里缓存所有数据
        """
        DataSaver.save_train_and_hidden(*MINUTE_10_DATA)
        DataSaver.save_train_and_hidden(*MINUTE_20_DATA)
        DataSaver.save_train_and_hidden(*MINUTE_30_DATA)
        DataSaver.save_train_and_hidden_weight(*MINUTE_10_DATA)
        DataSaver.save_train_and_hidden_weight(*MINUTE_20_DATA)
        DataSaver.save_train_and_hidden_weight(*MINUTE_30_DATA)

    @property
    def minute_10_tuple(self):
        return self.get_minute_list_tuple(MINUTE_10)

    @property
    def minute_20_tuple(self):
        return self.get_minute_list_tuple(MINUTE_20)

    @property
    def minute_30_tuple(self):
        return self.get_minute_list_tuple(MINUTE_30)

    @property
    def minute_10_weight_tuple(self):
        return self.get_minute_list_tuple(MINUTE_10, WEIGHT_LIST)

    @property
    def minute_20_weight_tuple(self):
        return self.get_minute_list_tuple(MINUTE_20, WEIGHT_LIST)

    @property
    def minute_30_weight_tuple(self):
        return self.get_minute_list_tuple(MINUTE_30, WEIGHT_LIST)

    @property
    def minute_10_dim(self):
        return self.get_minute_list_dim(MINUTE_10)

    @property
    def minute_20_dim(self):
        return self.get_minute_list_dim(MINUTE_20)

    @property
    def minute_30_dim(self):
        return self.get_minute_list_dim(MINUTE_30)

    @classmethod
    def get_minute_list_tuple(cls, minute_limit, name_list=FEATURE_LIST+WEIGHT_LIST):
        try:
            train_file = [
                pd.read_csv(
                    os.path.join(
                        SAVE_DATA_PATH, f"{TRAIN}_{minute_limit}_{feature}.csv"
                    )
                ).set_index(STUDENTID)
                for feature in name_list
            ]
            hidden_file = [
                pd.read_csv(
                    os.path.join(
                        SAVE_DATA_PATH, f"{HIDDEN}_{minute_limit}_{feature}.csv"
                    )
                ).set_index(STUDENTID)
                for feature in name_list
            ]
        except ValueError:
            raise ValueError("请先运行save_all一次进行数据持久化再调用。")
        return [train_file, hidden_file]

    @classmethod
    def get_minute_list_dim(cls, minute_limit):
        try:
            dim_df = pd.read_csv(
                os.path.join(SAVE_DATA_PATH, f"{minute_limit}_dim.csv")
            )
        except ValueError:
            raise ValueError("请先运行save_all一次进行数据持久化再调用。")
        return dim_df["dim"].to_list()


if __name__ == "__main__":
    data_exporter = DataExporter()
    data_exporter.save_all()
    test_df = data_exporter.minute_10_weight_tuple
    test_dm = data_exporter.minute_10_dim
