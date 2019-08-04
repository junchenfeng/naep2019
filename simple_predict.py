import os
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from data_prepare.data_processor import SimpleTestProcessor

RESULT_DIR = "data/result"
HIDDEN_30_NAME = "hidden_30_result.csv"


class SimplePredict(object):
    def __init__(self):
        self.train_data = SimpleTestProcessor().get_train_time_period_df()
        self.hidden_data = SimpleTestProcessor().get_hidden_30_time_period_df()
        self.model = self.init_model()

    @classmethod
    def init_model(cls):
        return RandomForestClassifier(n_estimators=200, random_state=0, verbose=1)

    def train_predict_model(self):
        x = self.train_data.iloc[:, 2:].values
        y = self.train_data.iloc[:, 1].astype(int).values
        self.model.fit(x, y)
        result = self.model.predict_proba(self.hidden_data.iloc[:, 1:].values)
        pd.DataFrame(result, columns=[False, True], index=self.hidden_data["STUDENTID"])
        result_df = pd.DataFrame(result, columns=[False, True], index=self.hidden_data["STUDENTID"])
        result_df.to_csv(os.path.join(RESULT_DIR, HIDDEN_30_NAME))


if __name__ == "__main__":
    SimplePredict().train_predict_model()
