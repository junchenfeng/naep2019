import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, KBinsDiscretizer
from sklearn.linear_model import LogisticRegression
from data_prepare.data_importer import DataImporter
from data_prepare.data_cleaner import (
    ACCESSION_NUMBER,
    ITEM_TYPE,
    OBSERVABLE,
    STUDENTID,
    DataCleaner,
)
from data_prepare.data_series_feature_computer import (
    DataOriginalSeriesFeatureComputer,
    DataWeightSeriesFeatureComputer,
)

FEATURE_LIST = [ACCESSION_NUMBER, ITEM_TYPE, OBSERVABLE]
VERB_WEIGHT = "verb_weight"
VERB_DURATION = "verb_duration"
WEIGHT_LIST = [VERB_WEIGHT]
DURATION_LIST = [VERB_DURATION]


class BaseDataEncoder(object):
    @classmethod
    def transform_column(cls, df, columns):
        pivot_list = []
        for column in columns:
            pivot_table = df.pivot(
                values=column, columns="occur_second", index=STUDENTID
            )
            pivot_list.append(pivot_table)
        return pivot_list


class DataObservationWeightEncoder(BaseDataEncoder):
    @classmethod
    def transform_train_and_hidden(cls, train, hidden):
        observable_weight_map = cls.get_observable_weight_map()
        train = (
            train.pipe(DataCleaner.clean_df)
            .pipe(cls.encode_df, observable_weight_map)
            .pipe(DataWeightSeriesFeatureComputer.get_series_feature)
        )
        hidden = (
            hidden.pipe(DataCleaner.clean_df)
            .pipe(cls.encode_df, observable_weight_map)
            .pipe(DataWeightSeriesFeatureComputer.get_series_feature)
        )
        train_pivot_list, hidden_pivot_list = cls.transform_both_columns(
            train, hidden, WEIGHT_LIST + DURATION_LIST
        )
        return train_pivot_list, hidden_pivot_list

    @classmethod
    def transform_both_columns(cls, train, hidden, columns):
        def get_pad_list(df):
            return df.groupby(STUDENTID, sort=False).agg(
                lambda x: list(
                    np.pad(
                        list(x),
                        (0, max_step_length - len(list(x))),
                        "constant",
                        constant_values=0,
                    )
                )
            )

        train_pivot_list = []
        hidden_pivot_list = []
        max_step_length = max(
            train.groupby(STUDENTID)["verb_weight"].count().max(),
            hidden.groupby(STUDENTID)["verb_weight"].count().max(),
        )
        return_columns = list(range(max_step_length))
        tmp_train_group_values = get_pad_list(train)
        tmp_hidden_group_values = get_pad_list(hidden)
        for column in columns:
            tmp_train_values = np.array(tmp_train_group_values[column].values.tolist())
            tmp_hidden_values = np.array(tmp_hidden_group_values[column].values.tolist())
            train_pivot_list.append(
                pd.DataFrame(
                    tmp_train_values, index=tmp_train_group_values.index, columns=return_columns
                )
            )
            hidden_pivot_list.append(
                pd.DataFrame(
                    tmp_hidden_values, index=tmp_hidden_group_values.index, columns=return_columns
                )
            )
        return train_pivot_list, hidden_pivot_list

    @classmethod
    def encode_df(cls, df, observable_weight_map):
        df = df.merge(
            observable_weight_map, how="left", on=[ACCESSION_NUMBER, OBSERVABLE]
        ).drop(columns=FEATURE_LIST)
        # 使用无影响4代替所有空值
        df["verb_weight"] = df["verb_weight"].fillna(4)
        return df

    @classmethod
    def get_observable_weight_map(cls):
        # 拿到train所有数据
        total_clean_train = DataCleaner.clean_df(DataImporter().data_a_train)
        total_train_label = DataImporter().data_train_label
        # 数据合并，整理
        total_train_label_df = total_clean_train.merge(
            total_train_label, how="left", on="STUDENTID"
        )
        total_train_label_df = total_train_label_df.assign(
            verb_values=total_train_label_df[ACCESSION_NUMBER]
            + "&&"
            + total_train_label_df[OBSERVABLE]
        )
        # one hot 编码
        one_hot_encoder = OneHotEncoder()
        one_hot_encoder.fit(total_train_label_df["verb_values"].values.reshape(-1, 1))
        verb_values = one_hot_encoder.transform(
            total_train_label_df["verb_values"].values.reshape(-1, 1)
        ).toarray()
        verb_hidden = total_train_label_df["EfficientlyCompletedBlockB"].values
        # 逻辑回归拟合，获取所有参数的系数
        clf = LogisticRegression(
            random_state=0,
            solver="liblinear",
            multi_class="ovr",
            class_weight="balanced",
        )
        clf.fit(verb_values, verb_hidden)
        verb_weight_map = pd.DataFrame(
            {
                "verb_values": one_hot_encoder.categories_[0],
                "verb_weight": clf.coef_.ravel(),
            }
        )
        est = KBinsDiscretizer(n_bins=6, encode="ordinal", strategy="uniform")
        est.fit(verb_weight_map["verb_weight"].values.reshape(-1, 1))
        # encode 之后统一加1，因为0将会用作mask
        verb_weight_map.loc[:, "verb_weight"] = (
            est.transform(verb_weight_map["verb_weight"].values.reshape(-1, 1)).ravel()
            + 1
        )
        # 拆分accession与observable
        accession_number_with_observable = (
            verb_weight_map["verb_values"]
            .str.split("&&", expand=True)
            .rename(columns={0: ACCESSION_NUMBER, 1: OBSERVABLE})
        )
        verb_weight_map = verb_weight_map.drop(columns="verb_values").join(
            accession_number_with_observable
        )
        return verb_weight_map


class DataOriginalLabelEncoder(BaseDataEncoder):
    @classmethod
    def transform_train_and_hidden(cls, train, hidden):
        train = DataOriginalSeriesFeatureComputer.get_series_feature(train)
        hidden = DataOriginalSeriesFeatureComputer.get_series_feature(hidden)
        train, hidden, dim_list = cls.encode_train_and_hidden(train, hidden)
        train_pivot_list = cls.transform_column(train, FEATURE_LIST)
        hidden_pivot_list = cls.transform_column(hidden, FEATURE_LIST)
        return train_pivot_list, hidden_pivot_list, dim_list

    @classmethod
    def encode_train_and_hidden(cls, train, hidden):
        """
        这里用了一个encoder，但是这个encoder不能选择筛选，所以利用了排序的时候'0'会被排到前边来
        实现不encode '0'，之后必须重写
        """
        train, hidden, accession_number_dim = cls.encode_column(
            train, hidden, ACCESSION_NUMBER
        )
        train, hidden, item_type_dim = cls.encode_column(train, hidden, ITEM_TYPE)
        train, hidden, observable_dim = cls.encode_column(train, hidden, OBSERVABLE)
        return train, hidden, [accession_number_dim, item_type_dim, observable_dim]

    @classmethod
    def encode_column(cls, train, hidden, column):
        encoder = LabelEncoder()
        encoder.fit(train[column].to_list() + hidden[column].to_list())
        encoder_dim = len(encoder.classes_)
        train.loc[:, column] = encoder.transform(train[column])
        hidden.loc[:, column] = encoder.transform(hidden[column])
        return train, hidden, encoder_dim


if __name__ == "__main__":
    train_data = DataImporter().data_a_train
    hidden_data = DataImporter().data_a_hidden_10
    train_df_list_tmp, hidden_df_list_tmp = DataObservationWeightEncoder.transform_train_and_hidden(
        train_data, hidden_data
    )
