from sklearn.ensemble import RandomForestClassifier

from data_prepare.data_processor import SimpleTestProcessor


class SimplePredict(object):
    def __init__(self):
        self.data = SimpleTestProcessor().get_time_period_df()
        self.model = self.init_model()

    @classmethod
    def init_model(cls):
        return RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0, verbose=1)

    def train_model(self):
        x = self.data.iloc[:, 2:].values
        y = self.data.iloc[:, 1].astype(int).values
        self.model.fit(x, y)


if __name__ == "__main__":
    SimplePredict().train_model()
