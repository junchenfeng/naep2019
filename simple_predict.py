from sklearn.ensemble import RandomForestClassifier

from data_prepare.data_processor import SimpleTestProcessor


class SimplePredict(object):
    def __init__(self):
        self.train_data = SimpleTestProcessor().get_train_time_period_df()
        self.hidden_data = SimpleTestProcessor().get_hidden_10_time_period_df()
        self.model = self.init_model()

    @classmethod
    def init_model(cls):
        return RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0, verbose=1)

    def train_predict_model(self):
        x = self.train_data.iloc[:, 2:].values
        y = self.train_data.iloc[:, 1].astype(int).values
        self.model.fit(x, y)
        result = self.model.predict_proba(self.hidden_data.iloc[:, 1:].values)
        print(result)


if __name__ == "__main__":
    SimplePredict().train_predict_model()
