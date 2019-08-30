from abc import ABCMeta, abstractmethod
from collections import namedtuple
import itertools

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
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
        self._model = self._get_model()

    @abstractmethod
    def _get_model(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @classmethod
    def model(cls):
        pass

    @classmethod
    @abstractmethod
    def classifier(cls):
        pass

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
    def balance_train_and_hidden(cls, train, hidden):
        # SORT OF CHEATING!
        common_columns = set(train.columns) & set(hidden.columns)
        return train[list(common_columns)], hidden[common_columns]

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


class RandForest(BaseModel):
    def _get_model(self):
        return self.classifier()

    def train(self):
        # Parameter Space
        num_tree_list = [500, 1500, 3000]  # 3
        max_features_list = np.arange(0.05, 0.66, 0.05)  #

        # cross validate
        best_score = 0
        best_param_set = None
        for param_set in itertools.product(num_tree_list, max_features_list):

            num_tree, max_features = param_set
            self._model.n_estimators = num_tree
            self._model.max_features = max_features
            self._train()

            if self.metrics.adj_score > best_score:
                best_score = self.metrics.adj_score
                best_param_set = param_set
            print(
                f"n {num_tree}, feature frac {max_features}: score {self.metrics.adj_score}"
            )

        # set optimal and train again
        best_num_tree, best_max_features = best_param_set
        self._model.n_estimators = best_num_tree
        self._model.max_features = best_max_features
        self._train()
        print(f"best score: {self.metrics}")
        print(f"best num tree: {best_num_tree}")
        print(f"best max features: {best_max_features}")

    @classmethod
    def classifier(
        cls, n_estimators=3000, random_state=0, verbose=1, max_features=0.3, n_jobs=3
    ):
        return RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            # verbose=verbose,
            max_features=max_features,
            n_jobs=n_jobs,
        )


class AdaBoost(BaseModel):
    def _get_model(self):
        return self.classifier(n_estimators=3000)

    def train(self):
        n_estimators = list(range(500, 3000, 500))
        adj_scores = np.zeros(len(n_estimators))
        for n_index, n_estimator in enumerate(n_estimators):
            self._model.n_estimators = n_estimator
            self._train()
            adj_scores[n_index] = self.metrics.adj_score
        print(adj_scores)
        best_arg_index = np.argmax(adj_scores)
        best_n_estimators = n_estimators[int(best_arg_index)]
        self._model.best_n_estimators = best_n_estimators
        self._train()
        print(f"best score: {self.metrics}")
        print(f"best n estimators: {best_n_estimators}")

    @classmethod
    def classifier(cls, n_estimators=5000):
        return AdaBoostClassifier(
            LogisticRegression(solver="lbfgs", max_iter=200, verbose=True, n_jobs=2),
            n_estimators=n_estimators,
        )
