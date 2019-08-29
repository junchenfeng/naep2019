import luigi
import pandas as pd

from etl.response import ConvertStream2Response, ITEM_REPO
from constant import ITEM_LIST, NOT_DONE, NO_ANS


"""
# MCSS: 单选题
ALL data

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
5个连线题

# CompositeCR:复合开放题
# VH139196
3题1空

"""


class GradePaper(luigi.Task):
    batch_id = luigi.Parameter()
    task = luigi.Parameter()

    def requires(self):
        return ConvertStream2Response(batch_id=self.batch_id, task=self.task)

    def output(self):
        return luigi.LocalTarget(f"data/mid/score_{self.task}_{self.batch_id}.csv")

    def run(self):

        response_df = pd.read_csv(self.input().path)

        score_table = []
        for i in range(response_df.shape[0]):
            sid = response_df.iloc[i, 0]
            label = response_df.iloc[i, 1]
            score_log = [sid, label]
            for j in range(19):
                resp = response_df.iloc[i, 2 + j]
                if resp in [NOT_DONE, NO_ANS]:
                    score = resp
                else:
                    qid = ITEM_LIST[j]
                    if isinstance(resp, str):
                        resp = resp.strip("\\").strip("\n")
                    score = ITEM_REPO[qid].judge(resp)
                score_log.append(score)
            score_table.append(score_log)

        pd.DataFrame(score_table, columns=["sid", "label"] + ITEM_LIST).to_csv(
            self.output().path, index=False
        )


class Grade(luigi.Task):
    def requires(self):
        yield [GradePaper(batch_id=x, task="train") for x in ["10", "20", "30"]]
        yield [GradePaper(batch_id=x, task="hidden") for x in ["10", "20", "30"]]


if __name__ == "__main__":
    luigi.run()
