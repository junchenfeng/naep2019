import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

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
VERB_MAP = {
    "Calculator Buffer": "Calculator",
    "Open Calculator": "Calculator",
    "Close Calculator": "Calculator",
    "Move Calculator": "Calculator",
    "Clear Scratchwork": "Scratchwork",
    "Scratchwork Draw Mode On": "Scratchwork",
    "Draw": "Scratchwork",
    "Erase": "Scratchwork",
    "Scratchwork Erase Mode On": "Scratchwork",
    "Scratchwork Highlight Mode On": "Scratchwork",
    "Scratchwork Mode Off": "Scratchwork",
    "Scratchwork Mode On": "Scratchwork",
    "Show Timer": "Timer",
    "Hide Timer": "Timer",
    "Vertical Item Scroll": "Scroll",
    "Horizontal Item Scroll": "Scroll",
    "Open Equation Editor": "Equation Editor",
    "Equation Editor Button": "Equation Editor",
    "Close Equation Editor": "Equation Editor",
    "Change Theme": "Theme",
    "Decrease Zoom": "Zoom",
    "Increase Zoom": "Zoom",
    "First Text Change": "Text Change",
    "Last Text Change": "Text Change",
}

# custom constants
ADJACENT_GROUP_INDEX = "adjacent_group_index"
VERB_ADJACENT_GROUP_INDEX = "verb_adjacent_group_index"
DURATION = "Duration"
VERB_DURATION = "verb_duration"
# verb_map


class FeatureProcessor(object):
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
        question_verb_chain_features = cls.get_question_verb_chain_features(
            df
        ).add_prefix("m_")
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
        df_with_accession_rank = cls.get_adjacent_index(
            df, ACCESSION_NUMBER, [INDEX_VAR, ACCESSION_NUMBER], ADJACENT_GROUP_INDEX
        )
        df_with_accession_and_verb_rank = cls.get_adjacent_index(
            df_with_accession_rank,
            OBSERVABLE,
            [INDEX_VAR, ACCESSION_NUMBER, ADJACENT_GROUP_INDEX],
            VERB_ADJACENT_GROUP_INDEX,
        )
        df_with_accession_and_verb_rank = df_with_accession_and_verb_rank.sort_values(
            by=[INDEX_VAR, EVENT_TIME, ACCESSION_NUMBER, ADJACENT_GROUP_INDEX]
        )
        return df_with_accession_and_verb_rank

    @classmethod
    def get_adjacent_index(cls, df, target_column, index_columns, rank_column_name):
        group_shift_judge = pd.Series(
            df[target_column] != df.shift().fillna(method="bfill")[target_column]
        )
        df_with_raw_rank = df.assign(
            cumsum_group_index=group_shift_judge.astype(int).cumsum()
        )
        df_with_raw_rank = (
            df_with_raw_rank.assign(
                tmp_column=df_with_raw_rank.groupby(index_columns, sort=False)[
                    "cumsum_group_index"
                ]
                .rank("dense")
                .values
            )
            .drop("cumsum_group_index", axis=1)
            .rename(columns={"tmp_column": rank_column_name})
        )
        return df_with_raw_rank

    @classmethod
    def get_question_attempt_duration(cls, df):
        converted_data = (
            df.groupby([INDEX_VAR, ACCESSION_NUMBER, ADJACENT_GROUP_INDEX], sort=False)[
                EVENT_TIME
            ]
            .agg(["min", "max"])
            .diff(axis=1)
            .drop("min", axis=1)
            .rename(columns={"max": DURATION})
            .reset_index()
        )
        converted_data[DURATION] = converted_data[DURATION].dt.total_seconds()
        return converted_data

    @classmethod
    def get_question_attempt_verb_chain(cls, df):
        sort_index = [
            INDEX_VAR,
            ACCESSION_NUMBER,
            ADJACENT_GROUP_INDEX,
            VERB_ADJACENT_GROUP_INDEX,
        ]
        group_index = sort_index + [OBSERVABLE]
        converted_data = (
            df.sort_values(by=sort_index)
            .groupby(group_index)[EVENT_TIME]
            .count()
            .reset_index(level=[OBSERVABLE])
        )
        converted_data = converted_data.assign(
            verb_with_times=converted_data[OBSERVABLE]
            + "("
            + converted_data[EVENT_TIME].astype(str)
            + ")"
        ).drop([OBSERVABLE, EVENT_TIME], axis=1)
        converted_data = converted_data.groupby(level=[0, 1, 2]).agg(
            lambda x: x.to_list()
        )
        converted_data["verb_with_times"] = converted_data["verb_with_times"].str.join(
            "=>"
        )
        converted_data = converted_data.rename(columns={"verb_with_times": VERB_CHAIN})
        return converted_data

    @classmethod
    def get_question_attempt_duration_features(cls, df):
        converted_data = cls.get_question_attempt_duration(df)

        return converted_data.pivot_table(
            values=DURATION,
            index=INDEX_VAR,
            columns=[
                converted_data.AccessionNumber,
                converted_data[ADJACENT_GROUP_INDEX],
            ],
        ).fillna(0)

    @classmethod
    def get_question_parameter(cls, df):
        converted_data = cls.get_question_attempt_duration(df)
        common_parameters = (
            converted_data.query(f"{DURATION}>1")
            .groupby([INDEX_VAR, ACCESSION_NUMBER])[DURATION]
            .agg(["count", np.mean])
            .rename(
                columns={"count": "valid_attempt_times", "mean": "valid_attempt_mean"}
            )
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
        question_convert_chain = question_convert_chain.reset_index(
            level=[ADJACENT_GROUP_INDEX, ACCESSION_NUMBER]
        )
        question_convert_chain = question_convert_chain.assign(
            verb_chain=question_convert_chain[ACCESSION_NUMBER]
            + "_"
            + question_convert_chain[ADJACENT_GROUP_INDEX].astype(str)
            + "_"
            + +question_convert_chain[VERB_CHAIN]
        ).drop(ACCESSION_NUMBER, axis=1)
        return pd.get_dummies(question_convert_chain).groupby(level=[0]).sum()

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

    @classmethod
    def get_duration_level(cls, df, variance_index, target_column):
        variance = (
            df.groupby(variance_index)[target_column]
            .std(1)
            .reset_index()
            .rename(columns={target_column: "duration_variance"})
        )
        tmp_var_df = df.merge(variance.reset_index(), on=variance_index, how="left")
        tmp_duration_level = (
            tmp_var_df[target_column]
            .floordiv(tmp_var_df["duration_variance"])
            .fillna(0)
        )
        duration_level_df = df.assign(duration_level=tmp_duration_level.values)
        return duration_level_df

    @classmethod
    def get_verb_attempt_duration(cls, df):
        converted_data = df.assign(shift_date=df[EVENT_TIME].shift(-1))
        converted_data = converted_data.assign(
            verb_duration=converted_data["shift_date"] - converted_data[EVENT_TIME]
        )
        converted_data[VERB_DURATION] = converted_data[VERB_DURATION].dt.total_seconds()
        converted_data.loc[
            ~(df[ACCESSION_NUMBER] == df[ACCESSION_NUMBER].shift(-1)), VERB_DURATION
        ] = 0
        converted_data = cls.get_duration_level(
            converted_data, [INDEX_VAR, ACCESSION_NUMBER], VERB_DURATION
        )
        return converted_data

    @classmethod
    def add_time_n_sigma(cls, df):
        df.loc[:, "EventTime"] = pd.to_datetime(df.EventTime)
        df = cls.simple_clean_df(df)
        df = cls.add_common_attempt_index(df)
        df = df.groupby(
            [
                "STUDENTID",
                ACCESSION_NUMBER,
                "ItemType",
                "adjacent_group_index",
                "verb_adjacent_group_index",
            ],
            sort=False,
        ).head(1)
        return cls.get_verb_attempt_duration(df)

    @classmethod
    def change_data_to_verb_time_chain(cls, df):
        """
        想要获取动作按顺序配合duration分级的动作序列，用于提升纯动作序列的效果
        TODO: 目前只进行了原始数据获取，数据格式并未进行具体整理。
        """
        verb_attempt_duration = cls.add_time_n_sigma(df)
        verb_attempt_duration = verb_attempt_duration.assign(
            verb_attempt_with_duration_level=verb_attempt_duration[ACCESSION_NUMBER]
            + "_"
            + verb_attempt_duration[OBSERVABLE]
            + "_"
            + "("
            + verb_attempt_duration["duration_level"].astype(int).astype(str)
            + ")"
        )
        verb_attempt_chain = verb_attempt_duration.groupby([INDEX_VAR], sort=False)[
            "verb_attempt_with_duration_level"
        ].agg(list)
        max_chain_length = verb_attempt_chain.apply(lambda x: len(x)).max()
        verb_attempt_chain = verb_attempt_chain.agg(
            lambda x: x + np.zeros(max_chain_length - len(x), str).tolist()
        )
        return (
            verb_attempt_chain.apply(pd.Series)
            .stack()
            .reset_index()
            .drop("level_1", axis=1)
            .rename(columns={0: "verbs"})
        )

    def get_verb_time_chain_pivot(self, df, ln):
        raw_data = self.change_data_to_verb_time_chain(df)
        raw_data["verbs"] = ln.transform(raw_data["verbs"])
        raw_data = raw_data.assign(
            verb_index=raw_data.reset_index()
            .groupby("STUDENTID", sort=False)["index"]
            .rank("min")
        )
        pivot_table = raw_data.pivot_table(
            values="verbs", index="STUDENTID", columns="verb_index"
        )
        return pivot_table

    @classmethod
    def get_multi_dim_verb(cls, df, question_type_ln, verb_ln):
        df[OBSERVABLE] = df[OBSERVABLE].apply(
            lambda x: VERB_MAP[x] if x in VERB_MAP else x
        )
        verb_attempt_duration = cls.add_time_n_sigma(df)
        verb_attempt_duration["duration_level"] = (
            verb_attempt_duration["duration_level"] + 1
        )
        max_chain_length = (
            verb_attempt_duration.groupby([INDEX_VAR], sort=False)
            .agg(
                {"Observable": "count", "ItemType": "count", "duration_level": "count"}
            )
            .max()
            .max()
        )
        verb_attempt_duration["Observable"] = verb_ln.transform(
            verb_attempt_duration["Observable"]
        ) + 1
        verb_attempt_duration["ItemType"] = question_type_ln.transform(
            verb_attempt_duration["ItemType"]
        ) + 1
        verb_attempt_chain = verb_attempt_duration.groupby([INDEX_VAR], sort=False).agg(
            {"Observable": list, "ItemType": list, "duration_level": list}
        )
        verb_attempt_chain["Observable"] = verb_attempt_chain["Observable"].agg(
            lambda x: x + np.zeros(max_chain_length - len(x)).tolist()
        )
        verb_attempt_chain["ItemType"] = verb_attempt_chain["ItemType"].agg(
            lambda x: x + np.zeros(max_chain_length - len(x)).tolist()
        )
        verb_attempt_chain["duration_level"] = verb_attempt_chain["duration_level"].agg(
            lambda x: x + np.zeros(max_chain_length - len(x)).tolist()
        )
        # 第一维度学生，第二维度时间步，第三维度，特征
        time_step_array = np.transpose(
            np.concatenate(
                verb_attempt_chain.apply(lambda x: list(x), axis=1).values
            ).reshape((verb_attempt_chain.shape[0], 3, max_chain_length)),
            (0, 2, 1),
        )
        return verb_attempt_chain.index, time_step_array


if __name__ == "__main__":
    data_a_hidden = pd.concat(
        [
            pd.read_csv(os.path.join(DATA_PATH, "data_a_hidden_10.csv")),
            pd.read_csv(os.path.join(DATA_PATH, "data_a_hidden_20.csv")),
            pd.read_csv(os.path.join(DATA_PATH, "data_a_hidden_30.csv")),
        ],
        axis=0,
        sort=False,
    )
    data_a_hidden[OBSERVABLE] = data_a_hidden[OBSERVABLE].apply(
        lambda x: VERB_MAP[x] if x in VERB_MAP else x
    )
    tmp_question_type_ln = LabelEncoder()
    tmp_verb_ln = LabelEncoder()
    tmp_verb_ln.fit(data_a_hidden["Observable"])
    tmp_question_type_ln.fit(data_a_hidden["ItemType"])
    train_verb_chain = FeatureProcessor().get_multi_dim_verb(
        data_a_hidden, tmp_question_type_ln, tmp_verb_ln
    )

    print("x")
