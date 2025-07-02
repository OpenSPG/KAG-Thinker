import sys
import requests
import re
from metric import ExactMatch
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import os
import argparse
import random

random.seed(1234)


def get_clarification_prompt(question):
    base_prompt = """You are an expert in function calls, capable of accurately understanding function definitions and precisely decompose user queries to select appropriate functions to solve problems. The functions are as follows:\n\nFunction Name: Retrieval\nDescription: Search for SPO information. S stands for subject, O stands for object, represented as variable_name:entity_type[entity_name], where entity_name is an optional parameter required when there is a specific query entity; P represents predicate, i.e., relation or property, indicated as variable_name:edge_type or attribute_type. A unique variable name is assigned to each variable for subsequent reference. Note that S, P, O should not appear repeatedly within the same expression. When the variable refers to a previously defined variable, the variable name must match exactly, and only the variable name needs to be provided, with the entity type specified only upon first introduction.\nFunction Usage: Retrieval(s=s_alias:type['name'], p=p_alias:edge, o=o_alias:type['name'], p.prop='value', s.prop='value', o.prop='value')\n\nFunction Name: Math\nDescription: Perform calculations, which include set operations such as numerical calculations or sorting and counting. Content provides input information, which can be text or a referenced variable name. The target is the computational objective, usually the current subproblem. Math_alia is a variable name that represents its calculation result and can be referenced in subsequent actions.\nFunction Usage: Math(content=[`XXX` or `o_alias/s_alias`], target=`XXX`)->math_alias\n\nFunction Name: Deduce\nDescription: Inference refers to the process of inferring search or calculation results to answer questions. op=judgement | entailment | rule | choice | multiChoice respectively represents true or false questions, implication reasoning (such as sentencing), fragment extraction, multiple choice questions, and multiple-choice questions. Content refers to questions, historical conversations, or search results, which can be text fragments or referred to by variable names. The target is the inference objective.\nFunction Usage: Deduce(op=judgement|entailment|extract|choice|multiChoice, content=[`XXX` or `o_alias/s_alias`], target=`XXX`)->deduce_alias\n\nFunction Name: Output\nDescription: Directly output A, B, ... as the answer, where A and B are variable names referencing previous retrieval or calculation results.\nFunction Usage: Output(A,B,...)\n\nPlease, based on the definition of the above function, decompose the user question into one or multiple logical steps, outputting the execution plan for each step along with the corresponding action. Please note:\nStep: Accurately point out the logical thinking process of the question, and use #1 to refer to the solution result of Step1, #2 to refer to the solution result of Step2, and so on\nAction: Indicate exactly the function you selected and its parameters.\n\nQuestion:\n    {question}\nOutput:\n    """
    return base_prompt.replace("{question}", question)


def get_subquestion_user_v2(question):
    return (
        "Can you answer the following questions step by step? If you can, wrap your answer with <answer>\\boxed{your answer}</answer>. If you can't, just reply that based on my internal knowledge, I can't answer this question, I need to retrieve external knowledge. \nQuestion: "
        + question
    )


def get_summary_user(question):
    return (
        "Refer to the reasoning process to get the final answer to the question and wrap your answer with <answer>\\boxed{your answer}</answer>. Question: "
        + question
    )


def get_summary_assistant(answer):
    return (
        "<think>I think I can answer this question now</think>\n\n<answer>\\boxed{"
        + answer
        + "}</answer>."
    )


def get_system_prompt():
    return """As you answer each question, you must provide a thought process and insert it between <think> and </think>."""


def get_output_prompt(question, sub_questions, last_question):
    base_prompt = """Answer the last question based on question, the first n sub-questions and their answers (indicated by #n), and wrap the result with <answer>\\boxed{your answer}</answer>. 
question:
{question}
sub-questions:
{sub_questions}
last-question:
{last_question}
answer:
"""
    base_prompt = (
        base_prompt.replace("{question}", question)
        .replace("{sub_questions}", sub_questions)
        .replace("{last_question}", last_question)
    )
    return base_prompt


def get_recalls(query, top_n=3, return_score=True):
    url = "http://0.0.0.0:12356/search"
    response = requests.post(
        url, json={"query": query, "top_n": top_n, "return_score": return_score}
    )
    results = response.json()
    try:
        docs = results[0]
        scores = results[1]
        recalls = []
        for doc, score in zip(docs, scores):
            recall = copy.deepcopy(doc)
            recall["score"] = score
            recalls.append(recall)
    except:
        print("recall fail!!!")
        print(query)
        recalls = []
    return recalls


def service(model, messages):
    if model == "kag-thinker-multiturn":
        url = "http://127.0.0.1:12389/v1/chat/completions"
    else:
        raise Exception("Online Model Not Supported!")
    data = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 0.95,
        "max_tokens": 8192,
        "stop_token_ids": [151643, 151645],
        "seed": 42,
    }
    attempts = 0
    while attempts < 3:
        try:
            response = requests.post(url, json=data, timeout=100)
            output = response.json()
            answer = output["choices"][0]["message"]["content"].strip()
            return answer
        except:
            attempts += 1
    return "None"


class KAGThinker:
    def __init__(self):
        pass

    def extract_box_answer(self, text):
        pattern = r"\\boxed\{([^}]*)\}"
        extracted_answers = re.findall(pattern, text)
        if len(extracted_answers) == 0:
            return ""
        else:
            return extracted_answers[0]

    def extract_answer(self, text):
        pattern = r"(?i)`.*?`"
        matches = re.findall(pattern, text)
        extracted_answers = []
        for match in matches:
            answer = re.search(r"`(.*?)`", match, re.IGNORECASE).group(1)
            extracted_answers.append(answer)
        if len(extracted_answers) == 0:
            return ""
        else:
            return extracted_answers[0]

    def search_plan_extraction(self, text):
        text = text.replace("\n", "")
        pattern = r"(?i)<search.*?>.*?</search>"
        matches = re.findall(pattern, text)

        extracted_plans = []
        for match in matches:
            plan = re.search(r"<search.*?>(.*?)</search>", match, re.IGNORECASE).group(
                1
            )
            extracted_plans.append(plan)
        if len(extracted_plans) == 0:
            return ""
        else:
            return extracted_plans[0].strip()

    def generate(self, messages):
        response = service("kag-thinker-multiturn", messages)
        return response

    def question_clear(self, question):
        question = question.split("Action")[0].strip()[6:].strip()
        return question

    def process(self, item):
        question = item.output["question"]
        messages = [{"role": "system", "content": get_system_prompt()}]
        decomposition_user = {
            "role": "user",
            "content": get_clarification_prompt(question),
        }
        messages.append(decomposition_user)
        logic_form = self.generate(messages)
        decomposition_assistant = {
            "role": "assistant",
            "content": logic_form,
        }
        logic_form = (
            logic_form.split("</think>")[-1]
            .strip()
            .replace("<answer>", "")
            .replace("</answer>", "")
            .strip()
        )
        messages.append(decomposition_assistant)
        sub_questions = [
            "Step" + x.strip() for x in logic_form.split("Step") if len(x) > 0
        ]
        answer_dict = {}
        history_question = ""
        for index, sub_question in enumerate(sub_questions[:-1]):
            for key, value in answer_dict.items():
                if key in sub_question and len(key.strip()) > 0:
                    sub_question = sub_question.replace(key, value)
            subquestion_user = {
                "role": "user",
                "content": get_subquestion_user_v2(sub_question),
            }
            messages.append(subquestion_user)
            num_turns = 0
            while num_turns < 10:
                num_turns += 1
                subquestion_response = self.generate(messages)
                if "<answer>" in subquestion_response:
                    subquestion_assistant = {
                        "role": "assistant",
                        "content": subquestion_response,
                    }
                    messages.append(subquestion_assistant)
                    predict = self.extract_box_answer(subquestion_response)
                    answer_dict["#" + str(index + 1)] = predict
                    history_question += self.question_clear(sub_question) + "\n"
                    history_question += "#" + str(index + 1) + ": " + predict + "\n"
                    break
                else:
                    if "<search>" in subquestion_response:
                        search = self.search_plan_extraction(subquestion_response)
                        search = self.question_clear(search)
                        recalls = get_recalls(search)
                        recall_str = ""
                        for recall in recalls:
                            recall_str += recall["contents"] + "\n\n"
                        recall_str = "<references>" + recall_str + "</references>"
                        subquestion_assistant = {
                            "role": "assistant",
                            "content": subquestion_response,
                        }
                        messages.append(subquestion_assistant)
                        subquestion_user = {
                            "role": "user",
                            "content": recall_str,
                        }
                        messages.append(subquestion_user)
                    else:
                        subquestion_assistant = {
                            "role": "assistant",
                            "content": subquestion_response,
                        }
                        messages.append(subquestion_assistant)

        last_question = sub_questions[-1]
        user_content = get_output_prompt(question, history_question, last_question)

        summary_user = {
            "role": "user",
            "content": user_content,
        }
        messages.append(summary_user)
        summary_response = self.extract_box_answer(self.generate(messages))
        summary_assistant = {
            "role": "assistant",
            "content": get_summary_assistant(summary_response),
        }
        messages.append(summary_assistant)
        golden_answers = item.output["golden_answers"]
        scores, _ = exact_match.calculate_metric([summary_response], [golden_answers])
        item.update_output("predict", summary_response)
        item.update_output("scores", scores)
        item.update_output("messages", messages)

    def run(self, data_list):
        with ThreadPoolExecutor(max_workers=200) as executor:
            futures = [executor.submit(self.process, item) for item in data_list]
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Inference: "
            ):
                future.result()
        return data_list


class Item:
    def __init__(self, item_dict):
        self.output = {}
        for k, v in item_dict.items():
            self.output[k] = v

    def update_output(self, key, value):
        self.output[key] = value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str)
    args = parser.parse_args()

    if not os.path.exists("save"):
        os.mkdir("save")
    wf = open("save" + "/predict_en.jsonl", "w", encoding="utf-8")

    kagThinker = KAGThinker()
    exact_match = ExactMatch()
    questions = []
    golden_answers_list = []
    pred_list = []
    datas = []
    with open(args.filename) as f:
        lines = f.readlines()
        for line in lines:
            datas.append(line.strip())
    random.shuffle(datas)
    data_list = []
    for data in datas:
        data = json.loads(data)
        data_item = Item(data)
        data_list.append(data_item)
    data_list = kagThinker.run(data_list)
    for data in data_list:
        pred_list.append(data.output["predict"])
        golden_answers_list.append(data.output["golden_answers"])
        wf.write(json.dumps(data.output, ensure_ascii=False) + "\n")
        wf.flush()
    results, metric_score_list = exact_match.calculate_metric(
        pred_list, golden_answers_list
    )
    print(results)
