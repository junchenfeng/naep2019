from abc import ABC
from collections import namedtuple

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import cross_val_predict, cross_val_score

Metrics = namedtuple("Metrics", "auc kappa adj_score")

RANDOM_STATE = 1
DEFAULT_SCORE_METHOD = "roc_auc"
DEFAULT_CV = 3
DEFAULT_VERBOSE = 0
DEFAULT_N_JOBS = 1


class BaseModel(ABC):
    def __init__(
        self,
        label_df: pd.DataFrame,
        feature_df: pd.DataFrame,
        predict_feature: pd.DataFrame,
    ):
        self._train_feature, self._predict_feature = self.balance_train_and_hidden(
            feature_df, predict_feature
        )
        self._train_label = label_df
        self._model = self.classifer()

    def train(self):
        pass

    def predict(self, df: pd.DataFrame):
        pass

    @classmethod
    def model(cls):
        pass

    @property
    def metrics(self):
        pass

    @classmethod
    def classifer(cls):
        pass

    @classmethod
    def balance_train_and_hidden(cls, train, hidden):
        # SORT OF CHEATING!
        common_columns = set(train.columns) & set(hidden.columns)
        return train[list(common_columns)], hidden[common_columns]


class RandForest(BaseModel):
    def train(self):
        x = self._train_feature.values
        y = self._train_label.astype(int).values
        self._model.fit(x, y)

    def predict(self, is_train: bool):
        if is_train:
            feature = self._train_feature
        else:
            feature = self._predict_feature.fillna(0)

        result = self._model.predict_proba(feature.values)
        result_df = pd.DataFrame(result, columns=["False", "True"], index=feature.index)
        return result_df

    @classmethod
    def classifer(cls):
        return RandomForestClassifier(n_estimators=1000, random_state=0, verbose=1, max_features=0.7)

    @property
    def metrics(self):
        auc = self._get_model_auc()
        kappa = self._get_model_kappa()
        adj_auc = 2 * (auc - 0.5) if auc > 0.5 else 0
        adj_kappa = max(kappa, 0)
        adj_score = adj_auc + adj_kappa

        return Metrics(auc, kappa, adj_score)

    def _get_model_auc(self) -> float:
        x = self._train_feature.values
        y = self._train_label.astype(int).values
        cv_results = cross_val_score(
            self._model, x, y, scoring=DEFAULT_SCORE_METHOD, cv=DEFAULT_CV
        )
        return np.mean(cv_results)

    def _get_model_kappa(self) -> float:
        x = self._train_feature.values
        y = self._train_label.astype(int).values
        y_predict = cross_val_predict(self._model, x, y, cv=DEFAULT_CV)
        return cohen_kappa_score(y, y_predict)
