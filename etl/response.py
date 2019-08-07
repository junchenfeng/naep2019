import json

from collections import defaultdict


import luigi
import pandas as pd
from tqdm import tqdm


from etl.util import (
    ItemMCSS,
    ItemFillBlanks,
    ItemMatchMS,
    ItemMultipleBlanks,
    ItemCompositeCR,
    NOT_DONE,
)


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


ITEM_REPO = {
    "VH098810": ItemMCSS(4),
    "VH098519": ItemMCSS(2),
    "VH098808": ItemMCSS(3),
    "VH139047": ItemMatchMS(),
    "VH098759": ItemMCSS(1),
    "VH098740": ItemMCSS(4),
    "VH134366": ItemMultipleBlanks(),
    "VH098753": ItemMCSS(4),
    "VH134387": ItemFillBlanks(["60", "X=60", "60 DEGREES"]),
    "VH098783": ItemMCSS(2),
    "VH098812": ItemMCSS(2),
    "VH139196": ItemCompositeCR(),
    "VH134373": ItemFillBlanks(["35", "35C"]),
    "VH098839": ItemMCSS(4),
    "VH098597": ItemMCSS(5),
    "VH098556": ItemMCSS(4),
    "VH098779": ItemMCSS(4),
    "VH098834": ItemMCSS(1),
    "VH098522": ItemMCSS(4),
}


class ConvertStreamTable2Json(luigi.Task):
    batch_id = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(f"data/mid/raw_{self.batch_id}.json")

    def run(self):
        student_repo = dict()
        label_data = pd.read_csv("data/raw/data_train_label.csv")
        for j in range(label_data.shape[0]):
            sid = str(label_data.iloc[j]["STUDENTID"])
            label = int(label_data.iloc[j]["EfficientlyCompletedBlockB"])
            student_repo[sid] = {"label": label, "logs": defaultdict(list)}

        log_data = pd.read_csv(f"data/raw/data_a_train_{self.batch_id}.csv")
        for i in tqdm(range(log_data.shape[0])):
            sid = str(log_data.iloc[i]["STUDENTID"])
            qid = log_data.iloc[i]["AccessionNumber"]
            action = log_data.iloc[i]["Observable"]
            info = log_data.iloc[i]["ExtendedInfo"]
            time = log_data.iloc[i]["EventTime"]

            if action == "Enter Item":
                current_qid = qid  # click Progress会改变qid

            student_repo[sid]["logs"][current_qid].append((action, info, time))

        with self.output().open("w") as f:
            json.dump(student_repo, f)


class ConvertStream2Response(luigi.Task):
    batch_id = luigi.Parameter()

    def requires(self):
        return ConvertStreamTable2Json(self.batch_id)

    def output(self):
        return luigi.LocalTarget(f"data/mid/response_{self.batch_id}.csv")

    def run(self):
        student_repo = json.load(self.input().open())

        response_table = []
        for sid, obj in student_repo.items():
            response_line = [sid, obj["label"]]
            for qid in ITEM_LIST:
                if qid in obj["logs"]:
                    response = ITEM_REPO[qid].answer(obj["logs"][qid])
                    response_line.append(response)
                else:
                    response_line.append(NOT_DONE)
            response_table.append(response_line)

        pd.DataFrame(response_table, columns=["sid", "label"] + ITEM_LIST).to_csv(
            self.output().path, index=False
        )


if __name__ == "__main__":
    luigi.run()
