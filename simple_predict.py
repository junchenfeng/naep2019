import os
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from data_prepare.data_processor import SimpleTestProcessor

RESULT_DIR = "data/result"
HIDDEN_30_NAME = "hidden_30_result.csv"
TOTAL_HIDDEN_NAME = "total_hidden.csv"


class SimplePredict(object):
    def __init__(self):
        self.train_data = SimpleTestProcessor().get_train_time_period_df()
        self.hidden_data = SimpleTestProcessor().get_total_hidden_period_df()
        self.hidden_label = SimpleTestProcessor().hidden_label
        self.model = self.init_model()

    @classmethod
    def init_model(cls):
        return RandomForestClassifier(n_estimators=1000, random_state=0, verbose=1)

    def train_predict_model(self):
        train_data, hidden_data = self.balance_train_and_hidden(self.train_data.iloc[:, 2:], self.hidden_data)
        x = train_data.values
        y = self.train_data.iloc[:, 1].astype(int).values
        self.model.fit(x, y)
        result = self.model.predict_proba(hidden_data.values)
        pd.DataFrame(
            result, columns=["False", "True"], index=self.hidden_data["STUDENTID"]
        )
        result_df = pd.DataFrame(
            result, columns=["False", "True"], index=self.hidden_data["STUDENTID"]
        )
        result_df = self.hidden_label.merge(result_df, on="STUDENTID", how="left")
        result_df[["True"]].to_csv(
            os.path.join(RESULT_DIR, TOTAL_HIDDEN_NAME),
            index=False,
            header=False,
            line_terminator=",\n",
        )

    @classmethod
    def balance_train_and_hidden(cls, train, hidden):
        common_columns = set(train.columns) & set(hidden.columns)
        return train[list(common_columns)], hidden[common_columns]


if __name__ == "__main__":
    SimplePredict().train_predict_model()
