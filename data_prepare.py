import pandas as pd
import os


ROOT_PATH = "data"
TRAIN_DATA_PATH = os.path.join(ROOT_PATH, "data_a_train.csv")


if __name__ == "__main__":
    train_data = pd.read_csv(TRAIN_DATA_PATH)
    print("x")
