import os
import pandas as pd

DATA_PATH = os.path.abspath("data/raw")


class DataImporter(object):
    @property
    def data_a_hidden_10(self):
        return pd.read_csv(os.path.join(DATA_PATH, "data_a_hidden_10.csv"))

    @property
    def data_a_hidden_20(self):
        return pd.read_csv(os.path.join(DATA_PATH, "data_a_hidden_20.csv"))

    @property
    def data_a_hidden_30(self):
        return pd.read_csv(os.path.join(DATA_PATH, "data_a_hidden_30.csv"))

    @property
    def data_a_train(self):
        return pd.read_csv(os.path.join(DATA_PATH, "data_a_train.csv"))

    @property
    def data_a_train_10(self):
        return pd.read_csv(os.path.join(DATA_PATH, "data_a_train_10.csv"))

    @property
    def data_a_train_20(self):
        return pd.read_csv(os.path.join(DATA_PATH, "data_a_train_20.csv"))

    @property
    def data_a_train_30(self):
        return pd.read_csv(os.path.join(DATA_PATH, "data_a_train_30.csv"))

    @property
    def data_train_label(self):
        return pd.read_csv(os.path.join(DATA_PATH, "data_train_label.csv"))

    @property
    def hidden_label(self):
        return pd.read_csv(os.path.join(DATA_PATH, "hidden_label.csv"))


if __name__ == "__main__":
    data_train = DataImporter().data_a_train
