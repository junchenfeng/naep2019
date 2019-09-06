import numpy as np
import pandas as pd
from data_prepare.data_importer import DataImporter
from data_prepare.data_cleaner import DataCleaner, STUDENTID, EVENT_TIME, ACCESSION_NUMBER, ITEM_TYPE, OBSERVABLE


class DataSeriesFeatureComputer(object):
    @classmethod
    def get_series_feature(cls, df):
        df = df.pipe(DataCleaner.clean_df).pipe(cls.get_occur_time).pipe(cls.split_time_bin)
        return df

    @classmethod
    def get_occur_time(cls, df):
        start_time_df = df.groupby(STUDENTID, sort=False, as_index=False)[
            "EventTime"
        ].agg(np.min)
        df = df.merge(start_time_df, how="left", on=STUDENTID)
        # 这里现在只考虑精度到1秒。慢慢提高精度。
        verb_occur_second_series = (
            (df[f"{EVENT_TIME}_x"] - df[f"{EVENT_TIME}_y"])
            .dt.total_seconds()
            .astype(int)
        )
        df = df.assign(occur_second=verb_occur_second_series.values).drop(
            columns=[f"{EVENT_TIME}_x", f"{EVENT_TIME}_y"]
        )
        # 有人居然超时了，删除超时的记录。
        df = df.query("occur_second <= 1800")
        return df

    @classmethod
    def split_time_bin(cls, df):
        max_second_df = (
            df.groupby(STUDENTID, sort=False, as_index=False)["occur_second"]
            .max()
            .rename(columns={"occur_second": "max_second"})
        )
        second_num = 1800
        student_num = len(df[STUDENTID].unique())
        full_second_list = np.tile(np.arange(0, second_num + 1), student_num)
        full_student_list = np.repeat(df[STUDENTID].unique(), second_num + 1)
        full_index_df = pd.DataFrame(
            {STUDENTID: full_student_list, "occur_second": full_second_list}
        )
        df = full_index_df.merge(df, how="left", on=[STUDENTID, "occur_second"]).fillna(
            method="ffill"
        ).merge(max_second_df, how="left", on=STUDENTID)
        df = df.groupby([STUDENTID, "occur_second"], sort=False, as_index=False).last()
        for col in [ACCESSION_NUMBER, ITEM_TYPE, OBSERVABLE]:
            df.loc[df["occur_second"] > df["max_second"], col] = '0'
        df = df.drop(columns="max_second")
        return df


if __name__ == "__main__":
    test_data = DataImporter().data_a_train
    series_feature = DataSeriesFeatureComputer.get_series_feature(test_data)
