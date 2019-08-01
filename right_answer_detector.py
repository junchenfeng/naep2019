from collections import defaultdict
import json

import luigi
import pandas as pd
from tqdm import tqdm

class MCSS_ITEM(object):
    def __init__(self, right_result):
        self.right_result = right_result

    def _separate(self, logs):
        attempts = []
        for log in logs:
            action, result, time = log
            if action == "Enter Item":
                attempts.append([])
            elif action == "Exit Item":
                continue
            else:
                if not attempts:
                    # 部分脏数据
                    continue
                attempts[-1].append((action, result))
        return attempts

    def _judge_attempt(self, attempt):
        results = [log[1] for log in attempt if log[0] == "Click Choice"]
        return int(results[-1] == self.right_result) if results else -1

    def judge(self, logs):
        attempts = self._separate(logs)
        scores = [self._judge_attempt(attempt) for attempt in attempts]

        # 最后一次尝试可能没有做任何变动（跳过时）
        # 或者每次都跳过无有效应答
        last_score = None
        if scores:
            for score in scores:
                last_score = score if score in [0, 1] else last_score
        return last_score

'''
# MCSS: 单选题
ALL data
"""
VH098783	1331	VH098783_2:checked	947	0.711495116
VH098522	1097	VH098522_4:checked	724	0.659981768
VH098519	1409	VH098519_2:checked	919	0.652235628
VH098597	1238	VH098597_5:checked	773	0.624394184

VH098779	1302	VH098779_4:checked	664	0.509984639
VH098810	2134	VH098810_4:checked	1057	0.495313964
VH098808	1433	VH098808_3:checked	686	0.47871598
VH098556	1359	VH098556_4:checked	489	0.3598234
VH098834	1193	VH098834_1:checked	412	0.345347863
VH098759	1405	VH098759_1:checked	484	0.344483986
VH098812	2035	VH098812_2:checked	680	0.334152334
VH098839	1305	VH098839_4:checked	428	0.327969349
VH098740	1878	VH098740_2:checked	589	0.313631523
VH098753	1426	VH098753_2:checked	430	0.301542777
"""

Conditional on All correct for Top 4

VH098759	VH098759_1:checked	271	348	78% 
VH098779	VH098779_4:checked	320	474	68% 
VH098808	VH098808_3:checked	287	456	63% 
VH098810	VH098810_4:checked	421	705	60% 

VH098812	VH098812_2:checked	315	581	54%
VH098839	VH098839_4:checked	227	430	53%
VH098834	VH098834_1:checked	230	438	53%
VH098556	VH098556_4:checked	229	480	48%
VH098740	VH098740_4:checked	240	606	40% [Diff]
VH098753	VH098753_2:checked	132	446	30% 


Conditional on All correct for Top 8

VH098839	VH098839_4:checked	96	138	69.6%
VH098834	VH098834_1:checked	93	133	69.9%
VH098812	VH098812_2:checked	114	171	66.7%
VH098556	VH098556_4:checked	92	144	63.9%

VH098740	VH098740_4:checked	89	195	45.6%
VH098753	VH098753_4:checked	49	137	35.8% [Diff]

Conditional on All correct for Top 12

VH098740	VH098740_4:checked	40	87	46.0%
VH098753	VH098753_4:checked	28	58	48.3%


# FillInBlank
# VH134373
{"numericIdentifier":"1","partId":"","contentMathML":"<math xmlns=\"http://www.w3.org/1998/Math/MathML\"/>","contentLaTeX":"$$","code":"Digit3"}
{"numericIdentifier":"1","partId":"","contentMathML":"<math xmlns=\"http://www.w3.org/1998/Math/MathML\"><mn>3</mn></math>","contentLaTeX":"$3$","code":"Digit5"}
# VH134387
{"numericIdentifier":"1","partId":"","contentMathML":"<math xmlns=\"http://www.w3.org/1998/Math/MathML\"/>","contentLaTeX":"$$","code":"Digit6"}
{"numericIdentifier":"1","partId":"","contentMathML":"<math xmlns=\"http://www.w3.org/1998/Math/MathML\"><mn>6</mn></math>","contentLaTeX":"$6$","code":"Digit0"}

# MultipleFillInBlank
# VH134366 [5空]

# MatchMS：连线题
# VH139047

# CompositeCR:复合开放题
# VH139196
'''

ITEM_LIST = [
    "VH098783",
    "VH098522",
    "VH098519",
    "VH098597",
    "VH098779",
    "VH098810",
    "VH098808",
    "VH098556",
    "VH098834",
    "VH098759",
    "VH098812",
    "VH098839",
    "VH098740",
    "VH098753",
    "VH134373",
    "VH134387",
    "VH134366",
    "VH139047",
    "VH139196",
]
ITEM_REPO = {
    "VH098783": MCSS_ITEM("VH098783_2:checked"),
    "VH098522": MCSS_ITEM("VH098522_4:checked"),
    "VH098597": MCSS_ITEM("VH098597_5:checked"),
    "VH098519": MCSS_ITEM("VH098519_2:checked"),
    "VH098759": MCSS_ITEM("VH098759_1:checked"),
    "VH098779": MCSS_ITEM("VH098779_4:checked"),
    "VH098808": MCSS_ITEM("VH098808_3:checked"),
    "VH098810": MCSS_ITEM("VH098810_4:checked"),
    "VH098839": MCSS_ITEM("VH098839_4:checked"),
    "VH098834": MCSS_ITEM("VH098834_1:checked"),
    "VH098812": MCSS_ITEM("VH098812_2:checked"),
    "VH098556": MCSS_ITEM("VH098556_4:checked"),
    #"VH098740": MCSS_ITEM("VH098740_4:checked"),
    #"VH098753": MCSS_ITEM("VH098753_4:checked"),
}




class ConvertCsv2Json(luigi.Task):
    def output(self):
        return luigi.LocalTarget("data/mid/raw.json")

    def run(self):
        student_repo = dict()
        label_data = pd.read_csv("data/data_train_label.csv")
        for j in range(label_data.shape[0]):
            sid = str(label_data.iloc[j]["STUDENTID"])
            label = int(label_data.iloc[j]["EfficientlyCompletedBlockB"])
            student_repo[sid] = {"label": label, "logs": defaultdict(list)}

        log_data = pd.read_csv("data/data_a_train.csv")
        for i in tqdm(range(log_data.shape[0])):
            sid = str(log_data.iloc[i]["STUDENTID"])
            qid = log_data.iloc[i]["AccessionNumber"]
            action = log_data.iloc[i]["Observable"]
            info = log_data.iloc[i]["ExtendedInfo"]
            time = log_data.iloc[i]["EventTime"]

            student_repo[sid]["logs"][qid].append((action, info, time))
        with self.output().open("w") as f:
            json.dump(student_repo, f)


class ConvertJson2Score(luigi.Task):
    def requires(self):
        return ConvertCsv2Json()

    def output(self):
        return luigi.LocalTarget("data/mid/score.csv")

    def run(self):



        student_repo = json.load(self.input().open())

        score_table = []
        for sid, obj in student_repo.items():
            score_log = [sid, obj["label"]]
            for qid in ITEM_LIST:
                if qid in obj["logs"]:
                    if qid in ITEM_REPO:
                        score_log.append(ITEM_REPO[qid].judge(obj["logs"][qid]))
                    else:
                        score_log.append(-1)
                else:
                    score_log.append(None)
            score_table.append(score_log)

        pd.DataFrame(score_table, columns=["sid", "label"] + ITEM_LIST).to_csv(
            self.output().path, index=False
        )


class FilterTopCandidates(luigi.Task):
    def requires(self):
        return ConvertJson2Score()

    def output(self):
        return luigi.LocalTarget("data/mid/top_candidates.csv")

    def run(self):
        score_df = pd.read_csv(self.input().path)

        filter_items = [x for x in ITEM_REPO.keys()]

        candidates = score_df[
            (score_df[filter_items[0]] == 1)
            & (score_df[filter_items[1]] == 1)
            & (score_df[filter_items[2]] == 1)
            & (score_df[filter_items[3]] == 1)
            & (score_df[filter_items[4]] == 1)
            & (score_df[filter_items[5]] == 1)
            & (score_df[filter_items[6]] == 1)
            & (score_df[filter_items[7]] == 1)
            & (score_df[filter_items[8]] == 1)
            & (score_df[filter_items[9]] == 1)
            & (score_df[filter_items[10]] == 1)
            & (score_df[filter_items[11]] == 1)
        ]["sid"]
        """
        all_data = pd.read_csv("data/data_a_train.csv")
        
        mcss_stat = (
            all_data[
                (all_data["ItemType"] == "MCSS")
                & (all_data["Observable"] == "Click Choice")
                & (all_data["STUDENTID"].isin(candidates))
            ]
            .groupby(["AccessionNumber", "ExtendedInfo"])
            .agg({"STUDENTID": "count"})
            .reset_index()
            .rename(columns={"STUDENTID": "count"})
        )

        right_idx = mcss_stat.groupby(['AccessionNumber'])['count'].transform(max) == mcss_stat['count']
        mcss_right_answer = mcss_stat[right_idx]
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(mcss_stat)
            print(mcss_right_answer)
        """
        candidates.to_csv(self.output().path, index=False)


class FilterTopBehavior(luigi.Task):
    def requires(self):
        return FilterTopCandidates()

    def output(self):
        return luigi.LocalTarget('data/mid/top_behavior.csv')

    def run(self):
        all_data = pd.read_csv("data/data_a_train.csv")
        top_candidates = pd.read_csv(self.input().path, header=None)

        all_data[all_data['STUDENTID'].isin(top_candidates[0])].to_csv(self.output().path, index=False)

if __name__ == "__main__":
    luigi.run()
