import sys
import requests
import re
from metric import ExactMatch
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import argparse
import os
import random

random.seed(1234)


def get_clarification_prompt(question):
    base_prompt = """您是一位函数调用专家，能够准确理解函数定义，并精准地将用户查询分解为适当的函数以解决问题。\n\n#以下是相关函数的描述：\n*函数名称：Retrieval\n描述：Retrieval函数用于搜索S、P和O信息。S代表主体，O代表客体，表示为 variable_name:实体类型[`实体名`]，其中实体名是可选参数，但当问题中有特定的查询实体时，实体名称是必要填写的；P代表谓词，即关系或属性，表示为：变量名:边类型或属性类型。每个变量都会被赋予唯一的变量名，以便后续引用。注意，S、P和O不应在同一个表达式中重复出现。当需要引用先前定义好的变量时，变量名必须完全匹配，并且只需提供变量名，实体类型仅在首次引入时指定。\n注意如果约束在P上，可以直接放在retrieval中。如果约束或者定语在实体S或者O上，则需要进行步骤拆解，通过多个Retrieval进行检索和过滤。\n``之间的 `名称` 或者 `值` 以及实体类型和边的名称需要根据输入问题的语言类型进行填充，一般使用中文。其余部分使用中文\n函数用法：Retrieval(s=s_alias:类型[`名称`], p=p_alias:边, o=o_alias:类型[`名称`], p.属性=`值`)\n\n*函数名称：Math\n描述：Math函数用于执行计算，包括集合运算、数值计算、排序和计数等。在Math函数中，content提供输入信息，这些信息可以是文本，也可以是引用的变量名。target是计算目标，通常是当前子问题。Math_alia是其计算结果的变量名，可在后续操作中引用。\n除了``之间的内容使用中文，其他使用英文。\n函数用法： Math(content=[`XXX` or `o_alias/s_alias`], target=`XXX`)->math_alias\n\n*函数名称：Deduce\n描述：推理是指通过推导搜索或计算结果来回答问题的过程。op=judgement | entailment | rule | choice | multiChoice 分别表示是非题、蕴含推理（如推断语句）、片段提取、单选题和多选题。content 指问题、历史对话或搜索结果，可以是文本片段或引用变量名。target 是推理目标。\n除了``之间的内容使用中文，其他使用英文。\n函数用法：Deduce(op=judgement|entailment|extract|choice|multiChoice, content=[`XXX` or `o_alias/s_alias`], target=`XXX`)->deduce_alias\n\n*函数名称：Output\n描述：直接输出 A、B 等作为答案，其中 A 和 B 是引用先前检索或计算结果的变量名。\n函数用法：Output(A,B,...)\n\n#请根据上述函数定义，将用户问题分解为一个或多个逻辑步骤，输出每一步的执行计划及相应的动作。请注意以下几点：\n1. 问题中的每个约束条件都必须使用。使用 Retrieval 时，如果约束在P上，可以直接放在retrieval中。如果约束或者定语在实体S或者O上，则需要进行步骤拆解，通过多个Retrieval进行检索和过滤。\n2. 确保问题进行合理的拆解，进行函数调用时，语义需要和拆解的step一致。如果进行多个retrieval，一般需要后面使用 Deduce 函数进行推理（entailment），判定（judgement），选择（choice）或者使用 Math 函数进行计算。\n3. 拆解的最后一个步骤，需要使用 Output函数，进行结果的输出。\n4. 请确保拆解成后，能够解决原始问题.\n5. 每一个拆解步骤包含 step与Action:\n    -- Step:确指出问题的逻辑思维过程，并使用 #1 引用步骤1的解决结果，#2 引用步骤2的解决结果，依此类推。请使用中文。\n    -- Action: 明确指出您选择的函数及其参数。\n\nQuestion:\n    {question}\n"""
    return base_prompt.replace("{question}", question)


def get_subquestion_user_v2(question):
    return (
        "你能一步步回答下面的问题吗？如果可以，请用<answer>\\boxed{你的答案}</answer>的格式包裹你的答案。如果不行，就回复说基于我的内部知识，我无法回答这个问题，我需要获取外部知识。 \n问题: \n"
        + question
    )


def get_summary_assistant(answer):
    return (
        "<think>我现在已经能回答这个问题了，问题的答案是</think>\n\n<answer>\\boxed{"
        + answer
        + "}</answer>."
    )


def get_system_prompt():
    return """当你回答每一个问题时，你必须提供一个思考过程，并将其插入到<think>和</think>之间"""


def get_output_prompt(question, sub_questions, last_question):
    base_prompt = """根据问题、前n个子问题及其答案（由#n表示）来回答最后一个子问题。请用<answer>\\boxed{你的答案}</answer>的格式包裹你的答案。 
问题:
{question}
子问题:
{sub_questions}
最后一个子问题:
{last_question}
输出:
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
        "repetition_penalty": 1.05,
        "stop_token_ids": [126081],
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
            while num_turns < 4:
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
    wf = open("save" + "/predict_ch.jsonl", "w", encoding="utf-8")

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
