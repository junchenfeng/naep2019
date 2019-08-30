import json
from typing import List

from constant import NO_ANS


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
        # TODO: 375<<. ArrowRight/ArrowLeft用于迁移输入数位
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
        elif key == "Backquote":
            output += "`"
        elif key == "Semicolon":
            output += ";"
        elif key == "BracketRight":
            output += "}"
        elif key == "BracketLeft":
            output += "{"
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
                "",
            ]
        ):
            # TODO: shift + key
            # TODO: ""说明有latex
            continue
        else:
            print(key)
    return output.strip('\n')


def score_list(user_list, right_list):
    num_blank = len(right_list)
    if all([x == [] for x in user_list]):
        return NO_ANS
    else:
        return [int(user_list[j] in right_list[j]) for j in range(num_blank)]


def extract_choice(choice_str):
    return int(choice_str.split(":")[0][-1])


def separate_attempt(logs: List):
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


class Item(object):
    pass


class ItemMCSS(Item):
    def __init__(self, right_result):
        self.right_result = right_result

    @classmethod
    def _write(cls, attempt):
        results = [
            extract_choice(log[1]) for log in attempt if log[0] == "Click Choice"
        ]
        return results[-1] if results else None

    def answer(self, logs):
        # what would be a paper exam looks like
        attempts = separate_attempt(logs)
        responses = [self._write(attempt) for attempt in attempts]
        valid_responses = [x for x in responses if x is not None]
        return valid_responses[-1] if valid_responses else NO_ANS

    def judge(self, response: str):
        return int(int(response) == self.right_result)


class ItemFillBlanks(Item):
    def __init__(self, right_ans_list):
        self.right_ans_list = right_ans_list

    def _write(self, attempt, input_str):
        key_strokes = [
            json.loads(log[1])["code"] for log in attempt if log[0] == "Math Keypress"
        ]
        output_str = translate(key_strokes, input_str)

        return output_str

    def answer(self, logs):
        attempts = separate_attempt(logs)
        output_str = self._write(attempts[-1], "")
        if len(attempts) > 1:
            for j in range(1, len(attempts)):
                output_str = self._write(attempts[j], output_str)
        return output_str if output_str else NO_ANS

    def judge(self, resp: str):
        return int(resp in self.right_ans_list) if resp else NO_ANS


class ItemMultipleBlanks(Item):
    def __init__(self):
        self.right_ans_list = [
            ["3.75"],
            ["5", "5.0", "5.00", "5.000"],
            ["6.25"],
            ["7.50", "7.5"],
            ["8.75"],
        ]
        self.num_blank = len(self.right_ans_list)

    def answer(self, logs):
        attempts = separate_attempt(logs)
        output_str_list = ["" for j in range(self.num_blank)]
        if attempts:
            output_str_list = self._write(attempts[0], output_str_list)
            if len(attempts) > 1:
                for j in range(1, len(attempts)):
                    output_str_list = self._write(attempts[j], output_str_list)

        return (
            output_str_list if any([x is not None for x in output_str_list]) else NO_ANS
        )

    def _write(self, attempt, input_str_list):

        key_strokes = [[] for j in range(self.num_blank)]
        for log in attempt:
            if log[0] != "Math Keypress":
                continue
            key_obj = json.loads(log[1])

            blank_idx = int(key_obj["numericIdentifier"]) - 1
            key_strokes[blank_idx].append(key_obj["code"])

        return [
            translate(key_strokes[j], input_str_list[j]) for j in range(self.num_blank)
        ]

    def judge(self, resp: str):
        user_list = eval(resp)
        return sum(score_list(user_list, self.right_ans_list))


class ItemCompositeCR(Item):
    def __init__(self):
        self.right_ans_list = [
            ["0.8", ".8", "0.80", ".80"],
            ["1.4", "1.40"],
            ["1.10", "1.1"],
        ]
        self.num_blank = len(self.right_ans_list)

    def answer(self, logs: List):
        attempts = separate_attempt(logs)
        output_str_list = ["" for j in range(self.num_blank)]
        if attempts:
            output_str_list = self._write(attempts[0], output_str_list)
            if len(attempts) > 1:
                for j in range(1, len(attempts)):
                    output_str_list = self._write(attempts[j], output_str_list)

        return (
            output_str_list if any([x is not None for x in output_str_list]) else NO_ANS
        )

    def _write(self, attempt, input_str_list):

        part_to_blank_ref = {"A": 0, "B": 1, "C": 2}

        key_strokes = [[] for j in range(self.num_blank)]
        for log in attempt:
            if log[0] != "Math Keypress":
                continue
            key_obj = json.loads(log[1])

            blank_idx = part_to_blank_ref[key_obj["partId"]]
            key_strokes[blank_idx].append(key_obj["code"])

        output_str_list = [
            translate(key_strokes[j], input_str_list[j]) for j in range(self.num_blank)
        ]

        return output_str_list

    def judge(self, resp):
        user_list = eval(resp)
        return sum(score_list(user_list, self.right_ans_list))


class ItemMatchMS(Item):
    def __init__(self):

        self.right_ans_list = [2, 3, 1, 4]
        self.num_blank = 4

    def answer(self, logs: List):
        attempts = separate_attempt(logs)
        user_ans_list = [None for i in range(self.num_blank)]
        if attempts:
            user_ans_list = self._write(attempts[0], user_ans_list)
            if len(attempts) > 1:
                for j in range(1, len(attempts)):
                    user_ans_list = self._write(attempts[j], user_ans_list)
        return user_ans_list if any([x is not None for x in user_ans_list]) else NO_ANS

    def _write(self, attempt, user_ans_list):
        for log in attempt:
            if log[0] != "DropChoice":
                continue
            for choice in json.loads(log[1]):
                user_ans_list[int(choice["source"]) - 1] = choice["target"]
        return user_ans_list

    def judge(self, resp: str):
        user_ans_list = eval(resp)
        return sum([user_ans_list[i] == self.right_ans_list[i] for i in range(4)])
