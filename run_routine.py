import os
import pickle

import pandas as pd

from data_prepare.raw_data import (
    DATA_A_TRAIN_10_FILE_NAME,
    DATA_A_TRAIN_20_FILE_NAME,
    DATA_A_TRAIN_30_FILE_NAME,
    DATA_A_HIDDEN_10_FILE_NAME,
    DATA_A_HIDDEN_20_FILE_NAME,
    DATA_A_HIDDEN_30_FILE_NAME,
    DATA_TRAIN_LABEL_FILE_NAME,
    HIDDEN_LABEL_FILE_NAME,
    DATA_PATH,
)
from data_prepare.data_processor import FeatureProcessor, INDEX_VAR
from model.ensemble import RandForest

BATCH_NAME_REF = {
    "10": {"train": DATA_A_TRAIN_10_FILE_NAME, "predict": DATA_A_HIDDEN_10_FILE_NAME},
    "20": {"train": DATA_A_TRAIN_20_FILE_NAME, "predict": DATA_A_HIDDEN_20_FILE_NAME},
    "30": {"train": DATA_A_TRAIN_30_FILE_NAME, "predict": DATA_A_HIDDEN_30_FILE_NAME},
}

BATCH_IDS = ["10", "20", "30"]

RESULT_DIR = "data/result"
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

def feature_extraction(batch_id: str, task_name: str):
    """

    :param batch_id: 10, 20 or 30
    :param task_name: train or predict
    :return:
    """

    df = pd.read_csv(os.path.join(DATA_PATH, BATCH_NAME_REF[batch_id][task_name]))
    feature_df = FeatureProcessor().change_data_to_feature_df(df)

    return feature_df


def model_training(batch_id, model_name: str):
    label_df = pd.read_csv(
        os.path.join(DATA_PATH, DATA_TRAIN_LABEL_FILE_NAME)
    ).set_index(INDEX_VAR)
    train_feature_df = feature_extraction(batch_id, "train")
    hidden_feature_df = feature_extraction(batch_id, "predict")
    train_data = pd.merge(
        train_feature_df, label_df, left_index=True, right_index=True
    )  # make sure they are aligned by student id

    if model_name == "random_forest":
        trained_model = RandForest(
            train_data.iloc[:, -1], train_data.iloc[:, :-1], hidden_feature_df
        )
    else:
        raise Exception("Unknown model")

    trained_model.train()
    print(trained_model.metrics)

    return trained_model


def main(model_name: str):
    cache_file_path = os.path.join(RESULT_DIR, f"{model_name}_cache.p")
    if os.path.exists(cache_file_path):
        model_repo = pickle.load(open(cache_file_path, 'rb'))
    else:
        model_repo = {}
        for batch_id in BATCH_IDS:
            model_repo[batch_id] = model_training(batch_id, model_name)
        pickle.dump(model_repo, open(cache_file_path,'wb'))

    predicted_labels = pd.concat(
        [
            model_repo["10"].predict(False),
            model_repo["20"].predict(False),
            model_repo["30"].predict(False),
        ],
        axis=0,
        sort=False,
    )
    predict_frame = pd.read_csv(
        os.path.join(DATA_PATH, HIDDEN_LABEL_FILE_NAME)
    ).set_index(INDEX_VAR)
    output_labels = pd.merge(
        predict_frame, predicted_labels, how="left", right_index=True, left_index=True
    )

    return (
        output_labels[["True"]]
        .fillna(0)
        .to_csv(
            os.path.join(RESULT_DIR, f"{model_name}.txt"),
            index=False,
            header=False,
            line_terminator=",\n",
        )
    )


if __name__ == "__main__":
    main("random_forest")
