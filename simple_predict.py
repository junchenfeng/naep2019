import os
from datetime import date
from sklearn.metrics import cohen_kappa_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score
import pandas as pd
import numpy as np
from data_prepare.data_processor import SimpleTestProcessor

RESULT_DIR = "data/result"
HIDDEN_30_NAME = "hidden_30_result.csv"
TOTAL_HIDDEN_NAME = f"total_hidden_{date.today()}.csv"
RANDOM_STATE = 1
DEFAULT_SCORE_METHOD = "roc_auc"
DEFAULT_CV = 3
DEFAULT_VERBOSE = 1
DEFAULT_N_JOBS = 1


class SimplePredict(object):
    def __init__(self):
        self.train_data_and_train_label = SimpleTestProcessor().get_train_time_period_df()
        self.hidden_data = SimpleTestProcessor().get_total_hidden_period_df()
        self.hidden_label = SimpleTestProcessor().hidden_label
        self.model = self.init_model()

    @classmethod
    def init_model(cls):
        return RandomForestClassifier(n_estimators=1000, random_state=0, verbose=1)

    def train_predict_model(self):
        train_data, hidden_data = self.balance_train_and_hidden(
            self.train_data_and_train_label.iloc[:, 2:], self.hidden_data
        )
        x = train_data.values
        y = self.train_data_and_train_label.iloc[:, 1].astype(int).values
        self.model.fit(x, y)
        result = self.model.predict_proba(hidden_data.values)
        result_df = pd.DataFrame(
            result, columns=["False", "True"], index=self.hidden_data.index
        )
        result_df = self.hidden_label.merge(result_df, on="STUDENTID", how="left")
        result_df[["True"]].to_csv(
            os.path.join(RESULT_DIR, TOTAL_HIDDEN_NAME),
            index=False,
            header=False,
            line_terminator=",\n",
        )

    def get_model_auc(self) -> float:
        train_data, hidden_data = self.balance_train_and_hidden(
            self.train_data_and_train_label.iloc[:, 2:], self.hidden_data
        )
        x = train_data.values
        y = self.train_data_and_train_label.iloc[:, 1].astype(int).values
        cv_results = cross_val_score(
            self.model, x, y, scoring=DEFAULT_SCORE_METHOD, cv=DEFAULT_CV
        )
        return 2 * float(np.mean(cv_results)) - 0.5

    def get_model_kappa(self) -> float:
        train_data, hidden_data = self.balance_train_and_hidden(
            self.train_data_and_train_label.iloc[:, 2:], self.hidden_data
        )
        x = train_data.values
        y = self.train_data_and_train_label.iloc[:, 1].astype(int).values
        y_predict = cross_val_predict(self.model, x, y, cv=DEFAULT_CV)
        return cohen_kappa_score(y, y_predict)

    @classmethod
    def balance_train_and_hidden(cls, train, hidden):
        common_columns = set(train.columns) & set(hidden.columns)
        return train[list(common_columns)], hidden[common_columns]


if __name__ == "__main__":
    SimplePredict().train_predict_model()
