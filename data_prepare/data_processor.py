import pandas as pd

from data_prepare.raw_data import RawData


class DataProcessor(object):
    raw_data = RawData


class SimpleTestProcessor(DataProcessor):
    def __init__(self):
        self.train_data = self.raw_data().data_a_train
        self.result_data = self.raw_data().data_train_label
        self.predict_data = self.raw_data().data_a_hidden_30

    def get_time_period_df(self):
        self.train_data["EventTime"] = pd.to_datetime(self.train_data.EventTime)
        converted_data = (
            self.train_data.groupby(["STUDENTID", "AccessionNumber"])
            .apply(lambda x: max(x["EventTime"]) - min(x["EventTime"]))
            .reset_index()
            .rename(columns={0: "Duration"})
        )
        converted_data["Duration"] = converted_data["Duration"].dt.total_seconds()
        time_period_pivot_df = (
            converted_data.pivot("STUDENTID", "AccessionNumber", "Duration")
            .reset_index()
            .fillna(0)
        )
        return self.result_data.merge(time_period_pivot_df, on="STUDENTID", how="left")


if __name__ == "__main__":
    time_period_df = SimpleTestProcessor().get_time_period_df()
