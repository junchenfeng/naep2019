from collections import defaultdict
import json

import luigi
import pandas as pd
from tqdm import tqdm


class Item(object):
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


class ItemMCSS(Item):
    def __init__(self, right_result):
        self.right_result = right_result

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


def translate(key_strokes, input_str):
    output = input_str
    for key in key_strokes:
        if key in ["Backspace", "Delete"]:
            if output:
                output = output[:-1]
        elif "Digit" in key:
            output += key.strip("Digit")
        elif key == "Period":
            output += "."
        elif "Key" in key:
            output += key.strip("Key")
        elif key == "Space":
            output += " "
        elif key == "Backslash":
            output += "/"
        elif key == "Slash":
            output += "\\"
        elif key == "Enter":
            output += "\n"
        elif key == "Equal":
            output += "="
        elif key == "ArrowRight":
            output += ">"
        elif key == "ArrowLeft":
            output += "<"
        elif key == "Minus":
            output += "-"
        elif key == "Comma":
            output += ","
        elif key == "Quote":
            output += "'"
        elif key == "Semicolon":
            output += ";"
        elif key == "BracketRight":
            output += "}"
        elif (
            "Shift" in key
            or "ControlLeft" in key
            or key
            in [
                "CapsLock",
                "Tab",
                "ArrowDown",
                "AltLeft",
                "AltRight",
                "ArrowUp",
                "Insert",
            ]
        ):
            # TODO: shift + key
            continue
        else:
            print(key)
    return output


class ItemFillBlanks(Item):
    def __init__(self, right_ans_list):
        self.right_ans_list = right_ans_list

    def _judge_attempt(self, attempt, input_str):
        key_strokes = [
            json.loads(log[1])["code"] for log in attempt if log[0] == "Math Keypress"
        ]
        output_str = translate(key_strokes, input_str)

        return output_str

    def judge(self, logs):
        attempts = self._separate(logs)

        output_str = self._judge_attempt(attempts[-1], "")
        if len(attempts) > 1:
            for j in range(1, len(attempts)):
                output_str = self._judge_attempt(attempts[j], output_str)

        return int(output_str in self.right_ans_list)


class ItemMultipleBlanks(Item):
    def __init__(self, right_ans_list):
        self.right_ans_list = right_ans_list

    def _judge_attempt(self, attempt, input_str_list):

        key_strokes = [[] for j in range(5)]
        for log in attempt:
            if log[0] != "Math Keypress":
                continue
            key_obj = json.loads(log[1])

            blank_idx = int(key_obj["numericIdentifier"]) - 1
            key_strokes[blank_idx].append(key_obj["code"])

        output_str_list = [
            translate(key_strokes[j], input_str_list[j]) for j in range(5)
        ]

        return output_str_list

    def judge(self, logs):
        attempts = self._separate(logs)
        output_str_list = ["" for j in range(5)]
        if attempts:
            output_str_list = self._judge_attempt(attempts[-1], output_str_list)
            if len(attempts) > 1:
                for j in range(1, len(attempts)):
                    output_str_list = self._judge_attempt(attempts[j], output_str_list)

        return [int(output_str_list[j] in self.right_ans_list[j]) for j in range(5)]


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
# 95F ? C. Ans = 35
# VH134387
_\_，外角150度，求Y截距的角度


# MultipleFillInBlank
# VH134366 [5空]
['3.75', '5', '6.25', '7.50', '8.75']
TODO: 研究2333267625

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
    "VH098783": ItemMCSS("VH098783_2:checked"),
    "VH098522": ItemMCSS("VH098522_4:checked"),
    "VH098597": ItemMCSS("VH098597_5:checked"),
    "VH098519": ItemMCSS("VH098519_2:checked"),
    "VH098759": ItemMCSS("VH098759_1:checked"),
    "VH098779": ItemMCSS("VH098779_4:checked"),
    "VH098808": ItemMCSS("VH098808_3:checked"),
    "VH098810": ItemMCSS("VH098810_4:checked"),
    "VH098839": ItemMCSS("VH098839_4:checked"),
    "VH098834": ItemMCSS("VH098834_1:checked"),
    "VH098812": ItemMCSS("VH098812_2:checked"),
    "VH098556": ItemMCSS("VH098556_4:checked"),
    "VH098740": ItemMCSS("VH098740_4:checked"),
    "VH098753": ItemMCSS("VH098753_4:checked"),
    "VH134373": ItemFillBlanks(["35", "35C"]),
    "VH134387": ItemFillBlanks(["60", "X=60", "60 DEGREES"]),
    "VH134366": ItemMultipleBlanks(
        [["3.75"], ["5", "5.0", "5.00"], ["6.25"], ["7.50", "7.5"], ["8.75"]]
    ),
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
        return luigi.LocalTarget("data/mid/top_behavior.csv")

    def run(self):
        all_data = pd.read_csv("data/data_a_train.csv")
        top_candidates = pd.read_csv(self.input().path, header=None)

        all_data[all_data["STUDENTID"].isin(top_candidates[0])].to_csv(
            self.output().path, index=False
        )


if __name__ == "__main__":
    luigi.run()
