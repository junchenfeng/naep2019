import os

import pandas as pd
import numpy as np

from etl.raw_data import DATA_PATH

MID_PATH = "data/mid"
TIME_PERIOD_DF_NAME = "time_period_df.csv"

# raw data constants
INDEX_VAR = "STUDENTID"
ACCESSION_NUMBER = "AccessionNumber"
OBSERVABLE = "Observable"
EVENT_TIME = "EventTime"
CLICK_PROGRESS_NAVIGATOR = "Click Progress Navigator"
BLOCK_REV = "BlockRev"
EOS_TIME_LFT = "EOSTimeLft"
HELP_MAT_8 = "HELPMAT8"
SEC_TIME_OUT = "SecTimeOut"
VERB_CHAIN = "verb_chain"


# custom constants
CUMSUM_GROUP_INDEX = "cumsum_group_index"
DURATION = "Duration"


class FeatureProcessor(object):
    @classmethod
    def change_data_to_feature_df(cls, df):
        df.loc[:, "EventTime"] = pd.to_datetime(df.EventTime)
        df = cls.simple_clean_df(df)
        df = cls.add_common_attempt_index(df)
        question_duration_features = cls.get_question_attempt_duration_features(
            df
        ).add_prefix("x_")
        question_verb_chain_features = cls.get_question_verb_chain_features(df)
        question_common_parameter_features = cls.get_question_common_parameter_features(
            df
        ).add_prefix("y_")
        student_common_parameter_features = cls.get_student_common_parameter_features(
            df
        ).add_prefix("z_")
        return pd.concat(
            [
                question_verb_chain_features,
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
        group_shift_judge = pd.Series(
            df[ACCESSION_NUMBER] != df.shift().fillna(method="bfill")[ACCESSION_NUMBER]
        )
        df_with_raw_rank = df.assign(
            cumsum_group_index=group_shift_judge.astype(int).cumsum()
        )
        df_with_raw_rank.loc[:, CUMSUM_GROUP_INDEX] = (
            df_with_raw_rank.sort_values(
                by=[INDEX_VAR, ACCESSION_NUMBER, CUMSUM_GROUP_INDEX, EVENT_TIME]
            )
            .groupby([INDEX_VAR, ACCESSION_NUMBER])[CUMSUM_GROUP_INDEX]
            .rank("dense")
        )
        df_with_raw_rank = df_with_raw_rank.sort_values(
            by=[INDEX_VAR, EVENT_TIME, ACCESSION_NUMBER, CUMSUM_GROUP_INDEX]
        )
        return df_with_raw_rank

    @classmethod
    def get_question_attempt_duration(cls, df):
        converted_data = (
            df.groupby([INDEX_VAR, ACCESSION_NUMBER, CUMSUM_GROUP_INDEX])
            .apply(lambda x: max(x[EVENT_TIME]) - min(x[EVENT_TIME]))
            .reset_index()
            .rename(columns={0: DURATION})
        )
        converted_data[DURATION] = converted_data[DURATION].dt.total_seconds()
        return converted_data

    @classmethod
    def get_question_attempt_verb_chain(cls, df):
        def question_verb_chain(sub_df):
            group_shift_judge = pd.Series(
                sub_df[OBSERVABLE] != sub_df.shift().fillna(method="bfill")[OBSERVABLE]
            )
            df_with_raw_rank = sub_df.assign(
                cumsum_group_index=group_shift_judge.astype(int).cumsum()
            )
            verb_group = (
                df_with_raw_rank.groupby(
                    [OBSERVABLE, "cumsum_group_index"], sort=False
                )[EVENT_TIME]
                .count()
                .reset_index()
                .rename(columns={EVENT_TIME: "verb_count"})
            )
            verb_chain_series = (
                verb_group["cumsum_group_index"].astype(str)
                + "_"
                + verb_group[OBSERVABLE]
                + "("
                + verb_group["verb_count"].astype(str)
                + ")"
            )
            return "=>".join(verb_chain_series.to_list())

        converted_data = (
            df.groupby([INDEX_VAR, ACCESSION_NUMBER, CUMSUM_GROUP_INDEX])
            .apply(question_verb_chain)
            .reset_index()
            .rename(columns={0: VERB_CHAIN})
        )
        return converted_data

    @classmethod
    def get_question_attempt_duration_features(cls, df):
        converted_data = cls.get_question_attempt_duration(df)

        return converted_data.pivot_table(
            values=DURATION,
            index=INDEX_VAR,
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
        common_parameters = converted_data.groupby([INDEX_VAR, ACCESSION_NUMBER]).apply(
            cls.get_question_parameters
        )
        return common_parameters.reset_index()

    @classmethod
    def get_question_common_parameter_features(cls, df):
        question_common_parameters = cls.get_question_parameter(df)
        return question_common_parameters.pivot_table(
            values=["valid_attempt_times", "valid_attempt_mean"],
            index=INDEX_VAR,
            columns=ACCESSION_NUMBER,
        ).fillna(0)

    @classmethod
    def get_question_verb_chain_features(cls, df):
        question_convert_chain = cls.get_question_attempt_verb_chain(df)
        return (
            question_convert_chain.assign(tmp=1)
            .pivot_table(
                values=["tmp"], index=INDEX_VAR, columns=[ACCESSION_NUMBER, VERB_CHAIN]
            )
            .fillna(0)
            .shape
        )

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
        common_parameters = converted_data.groupby([INDEX_VAR]).apply(
            cls.get_student_parameters, all_questions
        )
        return common_parameters


if __name__ == "__main__":
    data_a_train_10 = pd.read_csv(os.path.join(DATA_PATH, "data_a_train_10.csv"))
    FeatureProcessor().change_data_to_feature_df(data_a_train_10)
