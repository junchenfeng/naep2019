from sklearn.preprocessing import LabelEncoder
from data_prepare.data_importer import DataImporter
from data_prepare.data_cleaner import ACCESSION_NUMBER, ITEM_TYPE, OBSERVABLE, STUDENTID
from data_prepare.data_series_feature_computer import DataSeriesFeatureComputer

FEATURE_LIST = [ACCESSION_NUMBER, ITEM_TYPE, OBSERVABLE]


class DataEncoder(object):
    @classmethod
    def transform_train_and_hidden(cls, train, hidden):
        train = DataSeriesFeatureComputer.get_series_feature(train)
        hidden = DataSeriesFeatureComputer.get_series_feature(hidden)
        train, hidden, dim_list = cls.encode_train_and_hidden(train, hidden)
        train_pivot_list = cls.transform_column(train, FEATURE_LIST)
        hidden_pivot_list = cls.transform_column(hidden, FEATURE_LIST)
        return train_pivot_list, hidden_pivot_list, dim_list

    @classmethod
    def transform_column(cls, df, columns):
        pivot_list = []
        for column in columns:
            pivot_table = df.pivot_table(
                values=column, columns="occur_second", index=STUDENTID
            )
            pivot_list.append(pivot_table)
        return pivot_list

    @classmethod
    def encode_train_and_hidden(cls, train, hidden):
        """
        这里用了一个encoder，但是这个encoder不能选择筛选，所以利用了排序的时候'0'会被排到前边来
        实现不encode '0'，之后必须重写
        """
        train, hidden, accession_number_dim = cls.encode_column(train, hidden, ACCESSION_NUMBER)
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
    train_series_feature = DataSeriesFeatureComputer.get_series_feature(train_data)
    hidden_series_feature = DataSeriesFeatureComputer.get_series_feature(hidden_data)
    train_df_list_tmp, hidden_df_list_tmp, dim_list_tmp = DataEncoder.transform_train_and_hidden(
        train_series_feature, hidden_series_feature
    )
