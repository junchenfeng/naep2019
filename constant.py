import os
TASK_TRAIN = "train"
TASK_HIDDEN = "hidden"

NO_ANS = "N_A"
NOT_DONE = "N_D"

ITEM_LIST = [
    "VH098810",
    "VH098519",
    "VH098808",
    "VH139047",
    "VH098759",
    "VH098740",
    "VH134366",
    "VH098753",
    "VH134387",
    "VH098783",
    "VH098812",
    "VH139196",
    "VH134373",
    "VH098839",
    "VH098597",
    "VH098556",
    "VH098779",
    "VH098834",
    "VH098522",
]

INDEX_VAR = "STUDENTID"
EVENT_TIME = "EventTime"
DURATION = "Duration"

DATA_PATH = os.path.abspath("data")
DATA_A_TRAIN_10_FILE_NAME = "data_a_train_10.csv"
DATA_A_TRAIN_20_FILE_NAME = "data_a_train_20.csv"
DATA_A_TRAIN_30_FILE_NAME = "data_a_train_30.csv"
DATA_TRAIN_LABEL_FILE_NAME = "data_train_label.csv"

DATA_A_HIDDEN_10_FILE_NAME = "data_a_hidden_10.csv"
DATA_A_HIDDEN_20_FILE_NAME = "data_a_hidden_20.csv"
DATA_A_HIDDEN_30_FILE_NAME = "data_a_hidden_30.csv"
HIDDEN_LABEL_FILE_NAME = "hidden_label.csv"