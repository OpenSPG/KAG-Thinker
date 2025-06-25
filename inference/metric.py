import json
import re
import string
import os


def find_all_file(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            yield fullname


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


class ExactMatch:
    r"""Exact match measure whether the predicted answer is completely consistent
    with the standard answer.

    """

    metric_name = "em"

    def __init__(self):
        self.is_regex = False

    def calculate_em(self, prediction: str, golden_answers: list) -> float:
        if isinstance(golden_answers, str):
            golden_answers = [golden_answers]
        normalized_prediction = normalize_answer(prediction)
        score = 0.0
        for golden_answer in golden_answers:
            if self.is_regex:
                golden_answer = re.compile(golden_answer, re.IGNORECASE)
                match = re.fullmatch(golden_answer, normalized_prediction)
                if match is not None:
                    score = 1.0
                    break
            else:
                golden_answer = normalize_answer(golden_answer)
                if golden_answer == normalized_prediction:
                    score = 1.0
                    break
        return score

    def calculate_metric(self, pred_list, golden_answers_list):

        metric_score_list = [
            self.calculate_em(pred, golden_answers)
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        em_score = sum(metric_score_list) / len(metric_score_list)

        return {"em": "%.3f" % em_score}, metric_score_list


def evaluate(dataset_names):
    exact_match = ExactMatch()
    for dataset_name in dataset_names:
        files = find_all_file(dataset_name)
        pred_list = []
        golden_answers_list = []
        for file in files:
            with open(file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    try:
                        data = json.loads(line)
                    except:
                        print(line)
                        continue
                    golden_answers = data["golden_answers"]
                    pred = data["predict"]
                    pred_list.append(pred)
                    golden_answers_list.append(golden_answers)
        try:
            print("{} predict len = {}".format(dataset_name, len(pred_list)))
            results, metric_score_list = exact_match.calculate_metric(
                pred_list, golden_answers_list
            )
            print(results)
        except:
            pass


if __name__ == "__main__":
    dataset_names = [
        "nq",
        "triviaqa",
        "popqa",
        "hotpotqa",
        "2wikimultihopqa",
        "musique",
        "bamboogle",
    ]
    evaluate(dataset_names)
