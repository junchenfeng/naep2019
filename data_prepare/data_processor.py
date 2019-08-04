import pandas as pd
import numpy as np

from data_prepare.raw_data import RawData

MID_PATH = "data/mid"
TIME_PERIOD_DF_NAME = "time_period_df.csv"

# raw data constants
STUDENTID = "STUDENTID"
ACCESSION_NUMBER = "AccessionNumber"
OBSERVABLE = "Observable"
EVENT_TIME = "EventTime"
CLICK_PROGRESS_NAVIGATOR = "Click Progress Navigator"
BLOCK_REV = "BlockRev"
EOS_TIME_LFT = "EOSTimeLft"
HELP_MAT_8 = "HELPMAT8"
SEC_TIME_OUT = "SecTimeOut"


# custom constants
CUMSUM_GROUP_INDEX = "cumsum_group_index"
DURATION = "Duration"


class DataProcessor(object):
    raw_data = RawData


class SimpleTestProcessor(DataProcessor):
    def __init__(self):
        self.train_data = self.raw_data().data_a_train
        self.hidden_10 = self.raw_data().data_a_hidden_10
        self.hidden_20 = self.raw_data().data_a_hidden_20
        self.hidden_30 = self.raw_data().data_a_hidden_30
        self.hidden_label = self.raw_data().hidden_label
        self.result_data = self.raw_data().data_train_label
        self.predict_data = self.raw_data().data_a_hidden_30

    def get_train_time_period_df(self):
        feature_df = self.change_data_to_feature_df(self.train_data)
        train_data = self.result_data.merge(
            feature_df, on="STUDENTID", how="left"
        )
        return train_data

    def get_hidden_30_time_period_df(self):
        time_period_pivot_df = self.change_data_to_feature_df(self.hidden_30)
        return time_period_pivot_df

    def get_total_hidden_period_df(self):
        return pd.concat(
            [
                self.change_data_to_feature_df(self.hidden_10),
                self.change_data_to_feature_df(self.hidden_20),
                self.change_data_to_feature_df(self.hidden_30),
            ],
            axis=0,
            sort=False,
        ).fillna(0)

    @classmethod
    def change_data_to_feature_df(cls, df):
        df.loc[:, "EventTime"] = pd.to_datetime(df.EventTime)
        df = cls.simple_clean_df(df)
        df = cls.add_common_attempt_index(df)
        question_duration_features = cls.get_question_attempt_duration_features(
            df
        ).add_prefix("x_")
        question_common_parameter_features = cls.get_question_common_parameter_features(
            df
        ).add_prefix("y_")
        student_common_parameter_features = cls.get_student_common_parameter_features(
            df
        ).add_prefix("z_")
        return pd.concat(
            [
                question_duration_features,
                question_common_parameter_features,
                student_common_parameter_features,
            ],
            axis=1,
        )

    @classmethod
    def simple_clean_df(cls, df):
        return df.query(
            f"({OBSERVABLE} != '{CLICK_PROGRESS_NAVIGATOR}') & "
            f"({ACCESSION_NUMBER} != '{BLOCK_REV}') &"
            f"({ACCESSION_NUMBER} != '{EOS_TIME_LFT}') &"
            f"({ACCESSION_NUMBER} != '{HELP_MAT_8}') &"
            f"({ACCESSION_NUMBER} != '{SEC_TIME_OUT}')"
        )

    @classmethod
    def add_common_attempt_index(cls, df):
        group_shift_judge: pd.Series = (
            df[ACCESSION_NUMBER] != df.shift().fillna(method="bfill")[ACCESSION_NUMBER]
        )
        df_with_raw_rank = df.assign(
            cumsum_group_index=group_shift_judge.astype(int).cumsum()
        )
        df_with_raw_rank.loc[:, CUMSUM_GROUP_INDEX] = (
            df_with_raw_rank.sort_values(
                by=[STUDENTID, ACCESSION_NUMBER, CUMSUM_GROUP_INDEX, EVENT_TIME]
            )
            .groupby([STUDENTID, ACCESSION_NUMBER])[CUMSUM_GROUP_INDEX]
            .rank("dense")
        )
        df_with_raw_rank = df_with_raw_rank.sort_values(
            by=[STUDENTID, EVENT_TIME, ACCESSION_NUMBER, CUMSUM_GROUP_INDEX]
        )
        return df_with_raw_rank

    @classmethod
    def get_question_attempt_duration(cls, df):
        converted_data = (
            df.groupby([STUDENTID, ACCESSION_NUMBER, CUMSUM_GROUP_INDEX])
            .apply(lambda x: max(x[EVENT_TIME]) - min(x[EVENT_TIME]))
            .reset_index()
            .rename(columns={0: DURATION})
        )
        converted_data[DURATION] = converted_data[DURATION].dt.total_seconds()
        return converted_data

    @classmethod
    def get_question_attempt_duration_features(cls, df):
        converted_data = cls.get_question_attempt_duration(df)

        return converted_data.pivot_table(
            values=DURATION,
            index=STUDENTID,
            columns=[converted_data.AccessionNumber, converted_data.cumsum_group_index],
        ).fillna(0)

    @classmethod
    def get_question_parameters(cls, sub_df):
        valid_record = sub_df[sub_df[DURATION] > 1]
        valid_attempt_count = valid_record.shape[0]
        mean = np.mean(valid_record[DURATION]) if not valid_record.empty else 0
        return pd.Series(
            [valid_attempt_count, mean],
            index=["valid_attempt_times", "valid_attempt_mean"],
        )

    @classmethod
    def get_question_parameter(cls, df):
        converted_data = cls.get_question_attempt_duration(df)
        common_parameters = converted_data.groupby([STUDENTID, ACCESSION_NUMBER]).apply(
            cls.get_question_parameters
        )
        return common_parameters.reset_index()

    @classmethod
    def get_question_common_parameter_features(cls, df):
        question_common_parameters = cls.get_question_parameter(df)
        return question_common_parameters.pivot_table(
            values=["valid_attempt_times", "valid_attempt_mean"],
            index=STUDENTID,
            columns=ACCESSION_NUMBER,
        ).fillna(0)

    @classmethod
    def get_student_parameters(cls, sub_df, all_questions):
        valid_record = sub_df[sub_df[DURATION] > 1]
        all_question_count = all_questions.shape[0]
        valid_question_attempt_percent = (
            valid_record[ACCESSION_NUMBER].drop_duplicates().shape[0]
            / all_question_count
        )
        mean = np.mean(valid_record[DURATION]) if not valid_record.empty else 0
        std = np.std(valid_record[DURATION], ddof=1)
        return pd.Series(
            [valid_question_attempt_percent, mean, std],
            index=[
                "valid_question_attempt_percent",
                "valid_question_attempt_mean",
                "valid_question_attempt_std",
            ],
        )

    @classmethod
    def get_student_common_parameter_features(cls, df):

        all_questions = df[ACCESSION_NUMBER].drop_duplicates()
        converted_data = cls.get_question_attempt_duration(df)
        common_parameters = converted_data.groupby([STUDENTID]).apply(
            cls.get_student_parameters, all_questions
        )
        return common_parameters


if __name__ == "__main__":
    SimpleTestProcessor().get_train_time_period_df()
