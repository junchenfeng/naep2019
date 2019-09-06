import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from etl.raw_data import DATA_PATH

from constant import ITEM_LIST, INDEX_VAR, EVENT_TIME, DURATION

MID_PATH = "data/mid"
TIME_PERIOD_DF_NAME = "time_period_df.csv"

# raw data constants

ACCESSION_NUMBER = "AccessionNumber"
OBSERVABLE = "Observable"

CLICK_PROGRESS_NAVIGATOR = "Click Progress Navigator"
BLOCK_REV = "BlockRev"
EOS_TIME_LFT = "EOSTimeLft"
HELP_MAT_8 = "HELPMAT8"
SEC_TIME_OUT = "SecTimeOut"
VERB_CHAIN = "verb_chain"


# custom constants
ADJACENT_GROUP_INDEX = "adjacent_group_index"
VERB_ADJACENT_GROUP_INDEX = "verb_adjacent_group_index"

SCALE_DURATION = "SCALED_DURATION"
VERB_DURATION = "verb_duration"


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


class FeatureProcessor(object):
    @classmethod
    def _base_clean_df(cls, df):
        # TODO: keep the click progress navigator in the feature set.
        return df.query(
            f"({OBSERVABLE} != '{CLICK_PROGRESS_NAVIGATOR}') & "
            f"({ACCESSION_NUMBER} != '{BLOCK_REV}') &"
            f"({ACCESSION_NUMBER} != '{EOS_TIME_LFT}') &"
            f"({ACCESSION_NUMBER} != '{HELP_MAT_8}') &"
            f"({ACCESSION_NUMBER} != '{SEC_TIME_OUT}')"
        )

    @classmethod
    def _get_adjacent_index(cls, df, target_column, index_columns, rank_column_name):
        # Detect if the item changes. In simple clean step, the confounding navigator has been deleted
        group_shift_judge = pd.Series(
            df[target_column] != df.shift().fillna(method="bfill")[target_column]
        )
        df_with_raw_rank = df.assign(
            cumsum_group_index=group_shift_judge.astype(int).cumsum()
        )
        df_with_raw_rank = (
            df_with_raw_rank.assign(
                tmp_column=df_with_raw_rank.groupby(index_columns)[
                    "cumsum_group_index"
                ].rank("dense")
            )
            .drop("cumsum_group_index", axis=1)
            .rename(columns={"tmp_column": rank_column_name})
        )
        return df_with_raw_rank

    @classmethod
    def _add_common_attempt_index(cls, df):
        df_with_accession_rank = cls._get_adjacent_index(
            df, ACCESSION_NUMBER, [INDEX_VAR, ACCESSION_NUMBER], ADJACENT_GROUP_INDEX
        )
        df_with_accession_and_verb_rank = cls._get_adjacent_index(
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
    def preprocess(cls, df):
        df.loc[:, "EventTime"] = pd.to_datetime(df.EventTime)
        df = cls._base_clean_df(df)
        # common attempt = repeated verb in the same attempt of an item
        df = cls._add_common_attempt_index(df)

        return df


class TimeFeatureProcessor(FeatureProcessor):
    @classmethod
    def change_data_to_feature_df(cls, df):
        clean_df = cls.preprocess(df)
        # 每个学生、每个元素、每次尝试
        question_duration_features = cls.get_question_attempt_duration_features(
            clean_df
        ).add_prefix("x_")
        # 每个学生、每个元素
        question_common_parameter_features = cls.get_question_common_parameter_features(
            clean_df
        ).add_prefix("y_")
        # 每个学生
        student_common_parameter_features = cls.get_student_common_parameter_features(
            clean_df
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
    def _calc_each_attempt_duration(cls, df):
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
    def get_question_attempt_duration_features(cls, df):
        """
        每个学生对于每道题的每次尝试的时长数据
        """
        converted_data = cls._calc_each_attempt_duration(df)

        return converted_data.pivot_table(
            values=DURATION,
            index=INDEX_VAR,
            columns=[
                converted_data.AccessionNumber,
                converted_data[ADJACENT_GROUP_INDEX],
            ],
        ).fillna(0)

    @classmethod
    def get_question_common_parameter_features(cls, df):
        """
        每个学生、每个元素的尝试次数、平均答题时间和最小答题时间
        """
        converted_data = cls._calc_each_attempt_duration(df)
        question_common_parameters = (
            converted_data.groupby([INDEX_VAR, ACCESSION_NUMBER])[DURATION]
            .agg(["count", np.mean, np.min])
            .rename(
                columns={
                    "count": "attempt_times",
                    "mean": "attempt_mean_time",
                    "amin": "attempt_min_time",
                }
            )
        ).reset_index()
        return question_common_parameters.pivot_table(
            values=["attempt_times", "attempt_mean_time", "attempt_min_time"],
            index=INDEX_VAR,
            columns=ACCESSION_NUMBER,
        ).fillna(0)

    @classmethod
    def _make_student_parameters(cls, sub_df):
        # 获取学生的平均答题数据
        valid_record = sub_df[sub_df[SCALE_DURATION] > -1.5]  # 如果尝试时间比1.5个标准差小，定义为无效尝试
        all_question_count = len(ITEM_LIST)
        valid_question_attempt_percent = (
            valid_record[ACCESSION_NUMBER].drop_duplicates().shape[0]
            / all_question_count
        )
        mean = np.mean(sub_df[DURATION])  # natural time
        std = np.std(sub_df[DURATION], ddof=1)
        scaled_mean = np.mean(sub_df[SCALE_DURATION])  # scaled time compared to peers
        scaled_std = np.std(sub_df[SCALE_DURATION])
        return pd.Series(
            [valid_question_attempt_percent, mean, std, scaled_mean, scaled_std],
            index=[
                "valid_question_attempt_percent",
                "question_attempt_mean_time",
                "question_attempt_std_time",
                "scaled_question_attempt_mean_time",
                "scaled_question_attempt_std_time",
            ],
        )

    @classmethod
    def get_student_common_parameter_features(cls, df):
        """
        学生纬度答题的有效尝试比例、平均答题时间、平均答题时间标准差、平均标准化答题时间、平均标准化答题时间
        """
        # 首先仅提取有效题目数据
        df = df[df[ACCESSION_NUMBER].isin(ITEM_LIST)]

        converted_data = cls._calc_each_attempt_duration(df)
        converted_data[SCALE_DURATION] = 0
        # demean and standardize
        # TODO: use the same scaler to transform hidden data
        for question in ITEM_LIST:
            duration_data = converted_data[
                converted_data[ACCESSION_NUMBER] == question
            ]["Duration"]
            converted_data.loc[
                converted_data[ACCESSION_NUMBER] == question, SCALE_DURATION
            ] = (duration_data - duration_data.mean()) / duration_data.std()

        common_parameters = converted_data.groupby([INDEX_VAR]).apply(
            cls._make_student_parameters
        )
        return common_parameters


class BehaviorFeatureProcessor(FeatureProcessor):
    @classmethod
    def change_data_to_feature_df(cls, df):
        clean_df = cls.preprocess(df)
        question_verb_chain_features = cls.get_question_verb_chain_features(
            clean_df
        ).add_prefix(
            "m_"
        )  # verb stream of student attempt

        return question_verb_chain_features

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
        ).drop(EVENT_TIME, axis=1)
        group_data = converted_data.groupby(level=[0, 1, 2]).agg(
            lambda x: ",".join(sorted((set(x.to_list()))))
        )
        g = (
            group_data.reset_index()
            .groupby([ACCESSION_NUMBER, OBSERVABLE], group_keys=False)
            .count()["STUDENTID"]
            .groupby(level=0, group_keys=False)
        )
        valid_item_verb = (
            g.apply(lambda x: x.sort_values(ascending=False).head(10))
            .reset_index()
            .query("STUDENTID>20")
            .drop("STUDENTID", axis=1)
            .assign(valid=True)
        )  # take top 10 of each verbs and requires it to have at least 20 occurance.
        merged_verb_chain = pd.merge(
            group_data.reset_index(),
            valid_item_verb,
            left_on=["AccessionNumber", "Observable"],
            right_on=["AccessionNumber", "Observable"],
            how="left",
        )
        merged_verb_chain.loc[
            merged_verb_chain["valid"].isnull(), "Observable"
        ] = "OtherActions"
        converted_data = merged_verb_chain.drop(["valid"], axis=1).rename(
            columns={OBSERVABLE: VERB_CHAIN}
        )
        return converted_data

    @classmethod
    def get_question_verb_chain_features(cls, df):
        # 将学生每次做答的行为流所涉及的行为类别作为特征
        question_convert_chain = cls.get_question_attempt_verb_chain(df)
        question_convert_chain = question_convert_chain.assign(
            verb_chain=question_convert_chain[ACCESSION_NUMBER]
            + "_"
            + question_convert_chain[ADJACENT_GROUP_INDEX].astype(str)
            + "_"
            + question_convert_chain[VERB_CHAIN]
        ).drop([ACCESSION_NUMBER, ADJACENT_GROUP_INDEX], axis=1)
        verb_feature = pd.get_dummies(question_convert_chain).groupby("STUDENTID").sum()
        filtered_verb_feature = verb_feature.drop(
            verb_feature.columns[verb_feature.sum() < 10], axis=1
        )  # 因为还有一次尝试次数的拼接，稀释了常见动作的频率，因此需要再清洗一次
        return filtered_verb_feature


class ResponseFeatureProcessor(object):
    @classmethod
    def change_data_to_feature_df(cls, df, batch_id: str):
        # has to be in the format of response_{task}_{batch_id}.csv
        df.rename(columns={"sid": INDEX_VAR}, inplace=True)
        df = df.set_index(INDEX_VAR)
        if batch_id == "10":
            response_item = ["VH098519", "VH098808", "VH098759", "VH098740", "VH098783"]
        filter_df = df[response_item]
        enc = OneHotEncoder(handle_unknown="ignore")
        enc.fit(filter_df)
        feature_df = pd.DataFrame(
            enc.transform(filter_df).toarray(),
            index=filter_df.index.tolist(),
            columns=enc.get_feature_names(),
        )
        return feature_df


class TimeFrameFeatureProcessor(FeatureProcessor):
    @classmethod
    def add_duration_level(cls, df, variance_index, target_column):
        if variance_index:
            variance = (
                (df.groupby(variance_index, sort=False)[target_column].std(1) / 10)
                .reset_index()
                .rename(columns={target_column: "duration_variance"})
            )
            tmp_var_df = df.merge(variance.reset_index(), on=variance_index, how="left")
        else:
            variance = df[target_column].std() / 2000
            tmp_var_df = df.assign(duration_variance=variance)
        tmp_duration_level = (
            tmp_var_df[target_column]
            .floordiv(tmp_var_df["duration_variance"])
            .fillna(0)
        )
        duration_level_df = df.assign(duration_level=tmp_duration_level.values)
        return duration_level_df

    @classmethod
    def add_verb_attempt_duration(cls, df):
        converted_data = df.assign(shift_date=df[EVENT_TIME].shift(-1))
        converted_data = converted_data.assign(
            verb_duration=converted_data["shift_date"] - converted_data[EVENT_TIME]
        )
        converted_data[VERB_DURATION] = converted_data[VERB_DURATION].dt.total_seconds()
        converted_data.loc[
            ~(df[ACCESSION_NUMBER] == df[ACCESSION_NUMBER].shift(-1)), VERB_DURATION
        ] = 0
        return converted_data.drop("shift_date", axis=1)

    @classmethod
    def add_verb_occur_minute(cls, df):
        start_time_df = df.groupby("STUDENTID", sort=False, as_index=False)[
            "EventTime"
        ].agg(np.min)
        df = df.merge(start_time_df, how="left", on="STUDENTID")
        verb_occur_minute_series = (
            (df[f"{EVENT_TIME}_x"] - df[f"{EVENT_TIME}_y"]).dt.seconds / 60
        ).astype(int)
        df = (
            df.assign(occur_minute=verb_occur_minute_series.values)
            .drop(f"{EVENT_TIME}_y", axis=1)
            .rename(columns={f"{EVENT_TIME}_x": EVENT_TIME})
        )
        return df

    @classmethod
    def add_time_n_sigma(cls, df):
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
        return cls.add_verb_attempt_duration(df)

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
    def get_multi_dim_verb(cls, df):
        df[OBSERVABLE] = df[OBSERVABLE].apply(
            lambda x: VERB_MAP[x] if x in VERB_MAP else x
        )
        df.loc[:, "EventTime"] = pd.to_datetime(df.EventTime)
        # 此df在原始数据基础上添加题目进入序号，动作序号，动作duration, 动作duration_level, 时间occur_minute, 动作在每个题目中的duration_level
        middle_df = (
            df.pipe(cls.preprocess)
            .pipe(cls.add_verb_occur_minute)
            .pipe(cls.add_verb_attempt_duration)
            .pipe(cls.add_duration_level, None, VERB_DURATION)
        )
        middle_df = middle_df.query("duration_level<10")
        verb_attempt_chain = middle_df.groupby(
            [INDEX_VAR, "occur_minute"], sort=False
        ).agg({"Observable": list, "ItemType": list, "duration_level": "sum"})
        verb_attempt_chain = verb_attempt_chain.assign(
            verb_count=verb_attempt_chain["Observable"].agg(lambda x: len(x)),
            unique_verb_count=verb_attempt_chain["Observable"].agg(
                lambda x: len(set(x))
            ),
            unique_verb_chain=verb_attempt_chain["Observable"].agg(
                lambda x: json.dumps(sorted(list((set(x)))))
            ),
        )
        useful_df = verb_attempt_chain.iloc[:, 2:].reset_index()
        unique_student = useful_df[INDEX_VAR].unique()
        full_occur_minute = list(range(useful_df["occur_minute"].max() + 1))
        full_index = pd.DataFrame(
            {
                INDEX_VAR: np.repeat(unique_student, len(full_occur_minute)),
                "occur_minute": np.tile(full_occur_minute, len(unique_student)),
            }
        )
        full_df = full_index.merge(
            useful_df, on=[INDEX_VAR, "occur_minute"], how="left"
        )
        full_df = full_df.fillna(0)
        full_df["unique_verb_chain"] = full_df["unique_verb_chain"].astype(str)
        duration_level_array = full_df["duration_level"].values.reshape(
            len(unique_student), len(full_occur_minute), 1
        )
        verb_count_array = full_df["verb_count"].values.reshape(
            len(unique_student), len(full_occur_minute), 1
        )
        unique_verb_count_array = full_df["unique_verb_count"].values.reshape(
            len(unique_student), len(full_occur_minute), 1
        )
        unique_verb_chain_array = full_df["unique_verb_chain"].values.reshape(
            len(unique_student), len(full_occur_minute), 1
        )
        return (
            useful_df[INDEX_VAR].unique(),
            np.concatenate(
                [
                    duration_level_array,
                    verb_count_array,
                    unique_verb_count_array,
                    unique_verb_chain_array,
                ],
                axis=2,
            ),
        )

    @classmethod
    def change_data_to_feature_df(cls, task_name: str):
        data_a_hidden = pd.concat(
            [
                pd.read_csv(os.path.join(DATA_PATH, f"data_a_{task_name}_10.csv")),
                pd.read_csv(os.path.join(DATA_PATH, f"data_a_{task_name}_20.csv")),
                pd.read_csv(os.path.join(DATA_PATH, f"data_a_{task_name}_30.csv")),
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
        return cls.get_multi_dim_verb(data_a_hidden)


if __name__ == "__main__":
    data_a_train_10 = pd.read_csv(os.path.join(DATA_PATH, "data_a_train_10.csv"))
