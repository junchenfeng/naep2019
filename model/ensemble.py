from abc import ABCMeta, abstractmethod
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


class BaseModel(metaclass=ABCMeta):
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

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame):
        pass

    @classmethod
    def model(cls):
        pass

    @property
    @abstractmethod
    def metrics(self):
        pass

    @classmethod
    @abstractmethod
    def classifier(cls):
        pass

    @classmethod
    def balance_train_and_hidden(cls, train, hidden):
        # SORT OF CHEATING!
        common_columns = set(train.columns) & set(hidden.columns)
        return train[list(common_columns)], hidden[common_columns]


class RandForest(BaseModel):
    def __init__(
        self,
        label_df: pd.DataFrame,
        feature_df: pd.DataFrame,
        predict_feature: pd.DataFrame,
    ):
        super().__init__(label_df, feature_df, predict_feature)
        self._model = self.classifier(
            n_estimators=3000, random_state=0, verbose=1, max_features=0.7, n_jobs=2
        )

    def train(self):

        min_samples_leaf_list = list(range(1, 100, 30))
        max_features_list = np.arange(0.1, 1, 0.3)
        adj_scores = np.zeros((len(min_samples_leaf_list), len(max_features_list)))
        for min_index, min_samples_leaf in enumerate(min_samples_leaf_list):
            for max_index, max_features in enumerate(max_features_list):
                self._model.min_samples_leaf = min_samples_leaf
                self._model.max_features = max_features
                self._train()
                adj_scores[min_index, max_index] = self.metrics.adj_score
        print(adj_scores)
        best_arg_index = np.unravel_index(np.argmax(adj_scores), adj_scores.shape)
        best_min_sample_leaf = min_samples_leaf_list[int(best_arg_index[0])]
        best_max_features_list = max_features_list[int(best_arg_index[1])]
        self._model.min_samples_leaf = best_min_sample_leaf
        self._train()
        print(f"best score: {self.metrics}")
        print(f"best min leaf: {best_min_sample_leaf}")
        print(f"best max features: {best_max_features_list}")

    def _train(self):
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
    def classifier(
        cls, n_estimators=5000, random_state=0, verbose=1, max_features=0.7, n_jobs=2
    ):
        return RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            verbose=verbose,
            max_features=max_features,
            n_jobs=n_jobs,
        )

    @property
    def metrics(self):
        auc = self._get_model_auc()
        kappa = self._get_model_kappa()
        adj_auc = 2 * (auc - 0.5) if auc > 0.5 else 0
        adj_kappa = max(kappa, 0)
        adj_score = adj_auc + adj_kappa

        return Metrics(adj_auc, adj_kappa, adj_score)

    def _get_model_auc(self) -> np.array:
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
