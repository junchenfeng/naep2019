import os

import pandas as pd


DATA_PATH = os.path.abspath("data")
DATA_A_HIDDEN_10_FILE_NAME = "data_a_hidden_10.csv"
DATA_A_HIDDEN_20_FILE_NAME = "data_a_hidden_20.csv"
DATA_A_HIDDEN_30_FILE_NAME = "data_a_hidden_30.csv"
DATA_A_TRAIN_FILE_NAME = "data_a_train.csv"
DATA_TRAIN_LABEL_FILE_NAME = "data_train_label.csv"
HIDDEN_LABEL_FILE_NAME = "hidden_label.csv"


class RawData(object):
    @property
    def data_a_hidden_10(self):
        return pd.read_csv(os.path.join(DATA_PATH, DATA_A_HIDDEN_10_FILE_NAME))

    @property
    def data_a_hidden_20(self):
        return pd.read_csv(os.path.join(DATA_PATH, DATA_A_HIDDEN_20_FILE_NAME))

    @property
    def data_a_hidden_30(self):
        return pd.read_csv(os.path.join(DATA_PATH, DATA_A_HIDDEN_30_FILE_NAME))

    @property
    def data_a_train(self):
        return pd.read_csv(os.path.join(DATA_PATH, DATA_A_TRAIN_FILE_NAME))

    @property
    def data_train_label(self):
        return pd.read_csv(os.path.join(DATA_PATH, DATA_TRAIN_LABEL_FILE_NAME))

    @property
    def hidden_label(self):
        return pd.read_csv(os.path.join(DATA_PATH, HIDDEN_LABEL_FILE_NAME))
