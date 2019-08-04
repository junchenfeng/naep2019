from abc import ABC

import pandas as pd
from data_prepare.raw_data import (
    DATA_A_TRAIN_10_FILE_NAME,
    DATA_A_TRAIN_20_FILE_NAME,
    DATA_A_TRAIN_30_FILE_NAME,
    DATA_A_HIDDEN_10_FILE_NAME,
    DATA_A_HIDDEN_20_FILE_NAME,
    DATA_A_HIDDEN_30_FILE_NAME,
    DATA_TRAIN_LABEL_FILE_NAME,
)

BATCH_NAME_REF = {
    "10": {"train": DATA_A_TRAIN_10_FILE_NAME, "predict": DATA_A_HIDDEN_10_FILE_NAME},
    "20": {"train": DATA_A_TRAIN_20_FILE_NAME, "predict": DATA_A_HIDDEN_20_FILE_NAME},
    "30": {"train": DATA_A_TRAIN_30_FILE_NAME, "predict": DATA_A_HIDDEN_30_FILE_NAME},
}

BATCH_IDS = ["10", "20", "30"]


class BaseModel(ABC):
    def predict(self, df: pd.DataFrame):
        pass

    @property
    def metrics(self):
        return


# Step 1
def feature_extraction(batch_id: str, task_name: str):
    """

    :param batch_id: 10, 20 or 30
    :param task_name: train or predict
    :return:
    """

    df = pd.read_csv(BATCH_NAME_REF[batch_id][task_name])

    """
    DO SOMETHING HERE
    """

    feature_df = pd.DataFrame()

    return feature_df


# Step 2
def model_training(batch_id, model_name: str):
    label_df = pd.read_csv(DATA_TRAIN_LABEL_FILE_NAME)
    feature_df = feature_extraction(batch_id, "train")

    train_data = pd.merge(label_df)  # make sure they are aligned by student id

    """
    DO SOMETHING HERE
    """
    trained_model = BaseModel()

    return trained_model


# Step 3


def label_prediction(batch_id, model):
    feature_df = feature_extraction(batch_id, "test")
    label_prob = model.predict(feature_df)
    return label_prob


def routine(model_name: str):

    model_repo = {}

    for batch_id in BATCH_IDS:
        model_repo[batch_id] = model_training(batch_id, model_name)

    predicted_labels = (
        label_prediction("10", model_repo["10"])
        + label_prediction("20", model_repo["20"])
        + label_prediction("30", model_repo["30"])
    )
    # TODO: make sure it sorts by label order
    return predicted_labels
