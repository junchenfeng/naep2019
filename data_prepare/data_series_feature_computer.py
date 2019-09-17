import numpy as np
import pandas as pd
from data_prepare.data_importer import DataImporter
from data_prepare.data_cleaner import (
    DataCleaner,
    STUDENTID,
    EVENT_TIME,
    ACCESSION_NUMBER,
    ITEM_TYPE,
    OBSERVABLE,
)

TIME_MAX_LENGTH = 1800


class BaseSeriesFeatureComputer(object):
    @classmethod
    def get_occur_time(cls, df, time_bin=1):
        start_time_df = df.groupby(STUDENTID, sort=False, as_index=False)[
            "EventTime"
        ].agg(np.min)
        df = df.merge(start_time_df, how="left", on=STUDENTID)
        verb_occur_series = (
            df[f"{EVENT_TIME}_x"] - df[f"{EVENT_TIME}_y"]
        ).dt.total_seconds() / time_bin
        # 这里现在只考虑精度到1秒。慢慢提高精度。
        verb_occur_second_series = verb_occur_series.astype(int)
        verb_duration_series = verb_occur_series.shift(-1) - verb_occur_series
        zero_duration_series = df[STUDENTID] == df[STUDENTID].shift(-1)
        verb_duration_series[~zero_duration_series] = 0
        df = df.assign(
            occur_second=verb_occur_second_series.values,
            verb_duration=verb_duration_series.values,
        ).drop(columns=[f"{EVENT_TIME}_x", f"{EVENT_TIME}_y"])
        # 有人居然超时了，删除超时的记录。
        df = df.query(f"occur_second <= {TIME_MAX_LENGTH / time_bin}")
        return df

    @classmethod
    def complete_time_and_mask_zero(cls, df, columns):
        max_second_df = (
            df.groupby(STUDENTID, sort=False, as_index=False)["occur_second"]
            .max()
            .rename(columns={"occur_second": "max_second"})
        )
        second_num = TIME_MAX_LENGTH
        student_num = len(df[STUDENTID].unique())
        full_second_list = np.tile(np.arange(0, second_num + 1), student_num)
        full_student_list = np.repeat(df[STUDENTID].unique(), second_num + 1)
        full_index_df = pd.DataFrame(
            {STUDENTID: full_student_list, "occur_second": full_second_list}
        )
        df = (
            full_index_df.merge(df, how="left", on=[STUDENTID, "occur_second"])
            .fillna(method="ffill")
            .merge(max_second_df, how="left", on=STUDENTID)
        )
        for col in columns:
            df.loc[df["occur_second"] > df["max_second"], col] = "0"
        df = df.drop(columns="max_second")
        return df


class DataWeightSeriesFeatureComputer(BaseSeriesFeatureComputer):
    @classmethod
    def get_series_feature(cls, df):
        df = (
            df.pipe(cls.get_occur_time, 1).pipe(cls.agg_time_bin)
            # .pipe(cls.complete_time_and_mask_zero, ["verb_weight"])
        )
        return df

    @classmethod
    def agg_time_bin(cls, df):
        """
        暂时先不动，之后可以考虑时间段内各个verb的比例等参数，目前的情况是这样会导致长度不好处理
        """
        # df = df.groupby([STUDENTID, "occur_second"], sort=False, as_index=False).mean()
        df.loc[:, "verb_weight"] = df.loc[:, "verb_weight"].astype(np.int32)
        return df


class DataOriginalSeriesFeatureComputer(BaseSeriesFeatureComputer):
    @classmethod
    def get_series_feature(cls, df):
        df = (
            df.pipe(DataCleaner.clean_df)
            .pipe(cls.get_occur_time)
            .pipe(cls.agg_time_bin)
            .pipe(
                cls.complete_time_and_mask_zero,
                [ACCESSION_NUMBER, ITEM_TYPE, OBSERVABLE],
            )
        )
        return df

    @classmethod
    def agg_time_bin(cls, df):
        df = df.groupby([STUDENTID, "occur_second"], sort=False, as_index=False).last()
        return df


if __name__ == "__main__":
    test_data = DataImporter().data_a_train
    series_feature = DataOriginalSeriesFeatureComputer.get_series_feature(test_data)
