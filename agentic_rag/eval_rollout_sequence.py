import argparse
import json
import re
from collections import Counter
from typing import List, Tuple, Callable, Dict, Union

import numpy as np
import loguru
import string
import os
import jieba

def normalize_answer(answer: str) -> str:
    """
    Normalize a given string by applying the following transformations:
    1. Convert the string to lowercase.
    2. Remove punctuation characters.
    3. Remove the articles "a", "an", and "the".
    4. Normalize whitespace by collapsing multiple spaces into one.

    Args:
        answer (str): The input string to be normalized.

    Returns:
        str: The normalized string.
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(answer))))

def calculate_em_scores(gold_answers: List[List[str]], predicted_answers: List[str],
                            aggregation_fn: Callable = np.max) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """
    Calculates the Exact Match (EM) score.

    Args:
        gold_answers (List[List[str]]): List of lists containing ground truth answers.
        predicted_answers (List[str]): List of predicted answers.
        aggregation_fn (Callable): Function to aggregate scores across multiple gold answers (default: np.max).

    Returns:
        Tuple[Dict[str, float], List[Dict[str, float]]]:
            - A dictionary with the averaged EM score.
            - A list of dictionaries with EM scores for each example.
    """
    assert len(gold_answers) == len(
        predicted_answers), "Length of gold answers and predicted answers should be the same."

    example_eval_results = []
    total_em = 0

    for gold_list, predicted in zip(gold_answers, predicted_answers):
        em_scores = [1.0 if normalize_answer(gold) == normalize_answer(predicted) else 0.0 for gold in gold_list]
        if len(em_scores) == 0:
            em_scores = [0.0]
        aggregated_em = aggregation_fn(em_scores)
        example_eval_results.append({"ExactMatch": aggregated_em})
        total_em += aggregated_em

    avg_em = total_em / len(gold_answers) if gold_answers else 0.0
    pooled_eval_results = {"ExactMatch": avg_em}

    return pooled_eval_results, example_eval_results

def calculate_f1_scores(gold_answers: List[List[str]], predicted_answers: List[str], aggregation_fn: Callable = np.max) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """
    Calculates the F1 score.

    Args:
        gold_answers (List[List[str]]): List of lists containing ground truth answers.
        predicted_answers (List[str]): List of predicted answers.
        aggregation_fn (Callable): Function to aggregate scores across multiple gold answers (default: np.max).

    Returns:
        Tuple[Dict[str, float], List[Dict[str, float]]]:
            - A dictionary with the averaged F1 score.
            - A list of dictionaries with F1 scores for each example.
    """
    assert len(gold_answers) == len(predicted_answers), "Length of gold answers and predicted answers should be the same."
    eval_lang = os.environ.get("EVAL_LANG", "en")
    # detect chinese characters in gold_answers
    if any('\u4e00' <= char <= '\u9fff' for gold in gold_answers for char in gold[0]):
        if eval_lang != "zh":
            loguru.logger.warning("Detected Chinese characters in gold answers, carefully check EVAL_LANG")

    def compute_f1(gold: str, predicted: str) -> float:
        gold_tokens = normalize_answer(gold).split()
        predicted_tokens = normalize_answer(predicted).split()
        common = Counter(predicted_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            return 0.0

        precision = 1.0 * num_same / len(predicted_tokens)
        recall = 1.0 * num_same / len(gold_tokens)
        return 2 * (precision * recall) / (precision + recall)

    def compute_f1_zh(gold: str, predicted: str) -> float:
        gold_tokens = ' '.join(jieba.cut(normalize_answer(gold)))
        predicted_tokens = ' '.join(jieba.cut(normalize_answer(predicted)))
        return compute_f1(gold_tokens, predicted_tokens)

    example_eval_results = []
    total_f1 = 0.0

    for gold_list, predicted in zip(gold_answers, predicted_answers):
        if eval_lang == "zh":
            f1_scores = [compute_f1_zh(gold, predicted) for gold in gold_list]
        elif eval_lang == "en":
            f1_scores = [compute_f1(gold, predicted) for gold in gold_list]
        else:
            raise ValueError(f"Unsupported evaluation language: {eval_lang}")
        aggregated_f1 = aggregation_fn(f1_scores)
        example_eval_results.append({"F1": aggregated_f1})
        total_f1 += aggregated_f1

    avg_f1 = total_f1 / len(gold_answers) if gold_answers else 0.0
    pooled_eval_results = {"F1": avg_f1}

    return pooled_eval_results, example_eval_results

def extract_answer(solution_str: str, remove_boxed_answer=False) -> str:
    if "<|im_start|>assistant\n" in solution_str:
        solution_str_split = solution_str.split("<|im_start|>assistant\n")
    else:
        solution_str_split = solution_str.split("Assistant:")

    response = solution_str_split[1]
    text = response.strip()

    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return ""

    answer_part = match.group(1)
    if remove_boxed_answer:
        try:
            answer = remove_boxed(last_boxed_only_string(answer_part))
            return answer
        except Exception as e:
            return ""
    else:
        return answer_part


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval

def evaluate_rollout_sequence(gold_answers, answers, return_all=False):

    overall_qa_em_result, example_qa_em_results = calculate_em_scores(
        gold_answers=gold_answers, predicted_answers=answers,
        aggregation_fn=np.max)
    overall_qa_f1_result, example_qa_f1_results = calculate_f1_scores(
        gold_answers=gold_answers, predicted_answers=answers,
        aggregation_fn=np.max)

    overall_qa_em_result.update(overall_qa_f1_result)
    overall_qa_results = overall_qa_em_result
    overall_qa_results = {k: round(float(v)*100, 2) for k, v in overall_qa_results.items()}
    # loguru.logger.info(f"Evaluation results for QA: {overall_qa_results}")
    if return_all:
        return overall_qa_results, example_qa_em_results, example_qa_f1_results
    else:
        return overall_qa_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout_dir", type=str, default="/checkpoint_load/graph-agent-search-73/rollout", help="Path to the rollout directory")
    parser.add_argument("--step", type=int, default=50, help="Step number to evaluate")

    args = parser.parse_args()
    rollout_dir = args.rollout_dir
    step = args.step

    rollout_file = f"{rollout_dir}/val_{step}.jsonl"
    with open(rollout_file, "r") as f:
        all_rollout_data = [json.loads(line) for line in f]
    loguru.logger.info(f"Loaded {len(all_rollout_data)} rollout data from {rollout_file}")
    # classify by datasource
    data_source_data = {}
    for data in all_rollout_data:
        data_source = data.get("data_source", "unknown")
        if data_source not in data_source_data:
            data_source_data[data_source] = []
        data_source_data[data_source].append(data)

    for data_source, rollout_data in data_source_data.items():
        loguru.logger.info(f"Evaluating {len(rollout_data)} examples from data source: {data_source}")
        gold_answer_keys = ["ground_truth", "gold_answers", "answer"]
        gold_answer_key = ""
        for key in gold_answer_keys:
            if key in rollout_data[0]:
                gold_answer_key = key
                break

        gold_answers = [data[gold_answer_key] for data in rollout_data]
        if "sequences_str" in rollout_data[0]:
            sequences_str = [data["sequences_str"] for data in rollout_data]
            answers = [extract_answer(data) for data in sequences_str]
        elif "answer" in rollout_data[0]:
            answers = [data["answer"] for data in rollout_data]

        overall_qa_results = evaluate_rollout_sequence(gold_answers, answers)
        print(overall_qa_results)
