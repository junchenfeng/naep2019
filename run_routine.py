import os
import pickle

import pandas as pd


from constant import TASK_TRAIN, TASK_HIDDEN
from etl.raw_data import (
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
from etl.data_processor import TimeFeatureProcessor, BehaviorFeatureProcessor, INDEX_VAR
from model.ensemble import RandForest


BATCH_NAME_REF = {
    "10": {
        TASK_TRAIN: DATA_A_TRAIN_10_FILE_NAME,
        TASK_HIDDEN: DATA_A_HIDDEN_10_FILE_NAME,
    },
    "20": {
        TASK_TRAIN: DATA_A_TRAIN_20_FILE_NAME,
        TASK_HIDDEN: DATA_A_HIDDEN_20_FILE_NAME,
    },
    "30": {
        TASK_TRAIN: DATA_A_TRAIN_30_FILE_NAME,
        TASK_HIDDEN: DATA_A_HIDDEN_30_FILE_NAME,
    },
}

BATCH_IDS = ["10", "20", "30"]

RESULT_DIR = "data/result"
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)


def feature_extraction(batch_id: str, task_name: str):
    df = pd.read_csv(os.path.join(DATA_PATH, BATCH_NAME_REF[batch_id][task_name]))
    time_feature_df = TimeFeatureProcessor().change_data_to_feature_df(df)  # time
    all_feature = time_feature_df
    return all_feature


def model_training(batch_id, model_name: str):
    label_df = pd.read_csv(
        os.path.join(DATA_PATH, DATA_TRAIN_LABEL_FILE_NAME)
    ).set_index(INDEX_VAR)
    train_feature_df = feature_extraction(batch_id, TASK_TRAIN)
    hidden_feature_df = feature_extraction(batch_id, TASK_HIDDEN)
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
    model_repo = {}
    for batch_id in BATCH_IDS:
        # cache by each batch
        cache_file_path = os.path.join(RESULT_DIR, f"{model_name}_{batch_id}_cache.p")
        if os.path.exists(cache_file_path):
            model = pickle.load(open(cache_file_path, "rb"))
        else:
            model = model_training(batch_id, model_name)
            pickle.dump(model, open(cache_file_path, "wb"))
        model_repo[batch_id] = model

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

