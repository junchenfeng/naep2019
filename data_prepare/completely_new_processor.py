import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from etl.raw_data import DATA_PATH
from constant import ITEM_LIST

MID_PATH = "data/mid"
TIME_PERIOD_DF_NAME = "time_period_df.csv"


class DataProcessor(object):
    def __init__(self, raw_df):
        self.raw_df = raw_df

    def process_df(self):
        df = self.raw_df.pipe(DataCleaner.clean_df)
        return df


class DataSeriesFeatureComputer(object):
    pass


class DataExporter(object):
    pass


if __name__ == "__main__":
    data_train = DataImporter().data_a_train
    test_data = DataProcessor(data_train).process_df()
    print("x")

