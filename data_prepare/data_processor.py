import pandas as pd

from data_prepare.raw_data import RawData

MID_PATH = "data/mid"
TIME_PERIOD_DF_NAME = "time_period_df.csv"


class DataProcessor(object):
    raw_data = RawData


class SimpleTestProcessor(DataProcessor):
    def __init__(self):
        self.train_data = self.raw_data().data_a_train
        self.hidden_30 = self.raw_data().data_a_hidden_30
        self.result_data = self.raw_data().data_train_label
        self.predict_data = self.raw_data().data_a_hidden_30

    def get_train_time_period_df(self):
        time_period_pivot_df = self.chang_data_to_time_period_df(self.train_data)
        train_data = self.result_data.merge(
            time_period_pivot_df, on="STUDENTID", how="left"
        )
        return train_data

    def get_hidden_30_time_period_df(self):
        time_period_pivot_df = self.chang_data_to_time_period_df(self.hidden_30)
        return time_period_pivot_df

    @classmethod
    def chang_data_to_time_period_df(cls, df):
        df["EventTime"] = pd.to_datetime(df.EventTime)
        group_shift_judge: pd.Series = (
            df["STUDENTID"] != df.shift().fillna(method="bfill")["STUDENTID"]
        )
        group_index = group_shift_judge.astype(int).cumsum().rename("group")
        converted_data = (
            df.groupby(["STUDENTID", "AccessionNumber", group_index])
            .apply(lambda x: max(x["EventTime"]) - min(x["EventTime"]))
            .reset_index()
            .rename(columns={0: "Duration"})
        )
        converted_data["Duration"] = converted_data["Duration"].dt.total_seconds()
        return (
            converted_data.pivot("STUDENTID", "AccessionNumber", "Duration")
            .reset_index()
            .fillna(0)
        )


if __name__ == "__main__":
    SimpleTestProcessor().get_train_time_period_df()
