import re
import sys
import string
from typing import Union, List
from collections import Counter
import os

import jieba
import loguru

loguru.logger.info(f"Evaluation language is set to: {os.environ.get('EVAL_LANG', 'en')}")

def validate_format(text: str) -> tuple[bool, str]:
    # check if <think></think>, <answer></answer> is paired
    if text.count('<think>') != text.count('</think>'):
        return False, "<think> </think> not paired"
    
    if text.count('<think>') == 0 or text.count('</think>') == 0:
        return False, "<think> or </think> not found"
    
    if text.count('<answer>') != 1 or text.count('</answer>') != 1:
        return False, "<answer> or </answer> not found"        
    
    # check the order of search/result
    current_pos = 0
    start_patterns = ['<chunk_search>', '<graph_search>', '<get_adjacent_passages>', '<web_search>', '<browse_url>', '<local_search_agent>', '<web_search_agent>', '<all_search_agent>']
    end_patterns = ['</chunk_search>', '</graph_search>', '</get_adjacent_passages>', '</web_search>', '</browse_url>', '</local_search_agent>', '</web_search_agent>', '</all_search_agent>']
    while True:
        all_search_pos = [text.find(sp, current_pos) for sp in start_patterns if text.find(sp, current_pos) != -1]
        search_pos = min(all_search_pos) if all_search_pos else -1
        if search_pos == -1:
            break
            
        result_pos = text.find('<result>', search_pos)
        all_search_end_pos = [text.find(ep, search_pos) for ep in end_patterns if text.find(ep, search_pos) != -1]
        search_end_pos = min(all_search_end_pos) if all_search_end_pos else -1
        result_end_pos = text.find('</result>', result_pos)
        
        if -1 in (result_pos, search_end_pos, result_end_pos):
            return False, "search/result tags are incomplete"
            
        if not (search_pos < search_end_pos < result_pos < result_end_pos):
            return False, "search/result tags are nested in the wrong order"
        if len(all_search_pos) > 1:
            all_search_pos = sorted(all_search_pos)
            if not (all_search_pos[0] < search_end_pos < result_pos < result_end_pos < all_search_pos[1]):
                return False, "multiple search tags are nested in the wrong order"
            
        current_pos = result_end_pos
    
    # check if \boxed{} is in the answer
    answer_start = text.find('<answer>')
    answer_end = text.find('</answer>')
    if answer_start > answer_end:
        return False, "<answer> must be before </answer>"
    # answer_content = text[answer_start:answer_end]
    # if '\\boxed{' not in answer_content or '}' not in answer_content:
    #     return False, "answer is missing \\boxed{} format"
    
    return True, "format is correct"

def extract_answer(text: str):
    text = text.strip()

    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None
    
    return match.group(1)

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

def get_f1_score(prediction: str, ground_truths: Union[str, List[str]], eval_lang):
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    
    final_metric = {"f1": 0, "precision": 0, "recall": 0}

    for ground_truth in ground_truths:
        normalized_prediction = normalize_answer(prediction)
        normalized_ground_truth = normalize_answer(ground_truth)

        if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue
        
        if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue

        if eval_lang == "zh":
            normalized_prediction = " ".join(jieba.cut(normalized_prediction))
            normalized_ground_truth = " ".join(jieba.cut(normalized_ground_truth))
        elif eval_lang == "en":
            pass
        else:
            raise ValueError(f"Unsupported eval_lang: {eval_lang}")
        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        
        final_metric["precision"] = max(precision, final_metric["precision"])
        final_metric["recall"] = max(recall, final_metric["recall"])
        final_metric["f1"] = max(f1, final_metric["f1"])
    
    return final_metric['f1']


def get_local_web_score(solution_str: str) -> float:
    """
    The fist 3 search tool calls must be local search, including <chunk_search>, <graph_search>, <get_adjacent_passages>
    If the first 3 search tool calls are not local search, return -1.0.
    Discourage the use of web search tool calls (<web_search>, <browse_url>) between 4th and 6th search tool calls.
    Discourage the use of web search tool calls (<web_search>, <browse_url>) after 6th search tool calls mildly.
    """
    start_patterns = ['<chunk_search>', '<graph_search>', '<get_adjacent_passages>', '<web_search>', '<browse_url>']
    local_tool_calls = ['<chunk_search>', '<graph_search>', '<get_adjacent_passages>']
    web_tool_calls = ['<web_search>', '<browse_url>']
    # get all tool calls and their positions
    all_search_pos = {}
    for sp in start_patterns:
        # find all occurrences of the pattern in the solution string
        pos = solution_str.find(sp)
        while pos != -1:
            if sp not in all_search_pos:
                all_search_pos[sp] = []
            all_search_pos[sp].append(pos)
            pos = solution_str.find(sp, pos + 1)
    tool_call_with_pos = []
    for sp, positions in all_search_pos.items():
        for pos in positions:
            tool_call_with_pos.append((sp, pos))
    tool_call_with_pos.sort(key=lambda x: x[1])  # sort by position
    # check the first 3 tool calls
    first_three_tool_calls = [x[0] for x in tool_call_with_pos[:3]]
    if any(x in web_tool_calls for x in first_three_tool_calls):
        return -1.0
    # check the 4th to 6th tool calls
    fourth_to_sixth_tool_calls = [x[0] for x in tool_call_with_pos[3:6]]
    if any(x in web_tool_calls for x in fourth_to_sixth_tool_calls):
        penalty = -0.05  # default penalty for using web search tool calls
        for i, x in enumerate(fourth_to_sixth_tool_calls):
            if x in web_tool_calls:
                penalty = -0.05 * (4 - i)
                break
        return penalty

    # check the 7th to 8th tool calls
    ninth_to_eighth_tool_calls = [x[0] for x in tool_call_with_pos[6:8]]
    if any(x in web_tool_calls for x in ninth_to_eighth_tool_calls):
        penalty = -0.015
        for i, x in enumerate(ninth_to_eighth_tool_calls):
            if x in web_tool_calls:
                penalty = -0.015 * (3 - i)
                break
        return penalty
    # check the 9th and later tool calls
    later_tool_calls = [x[0] for x in tool_call_with_pos[8:]]
    if any(x in web_tool_calls for x in later_tool_calls):
        return -0.01  # mild penalty for using web search tool calls after the 8th call

    return 0.0  # all good


def tool_call_comprehensiveness(prompt: str, solution_str: str) -> float:
    """
    Check if the solution string contains at least one of the local search tool calls (<chunk_search>, <graph_search>, <get_adjacent_passages>)
    If not, return -1.0.
    """
    all_tool_calls = ['<chunk_search>', '<graph_search>', '<get_adjacent_passages>', '<web_search>', '<browse_url>', '<local_search_agent>', '<web_search_agent>', '<all_search_agent>']
    valid_tool_calls = [x for x in all_tool_calls if x in prompt]
    used_tool_calls = [x for x in all_tool_calls if x in solution_str]
    return 0.1 * len(set(used_tool_calls) & set(valid_tool_calls)) / len(set(valid_tool_calls)) if valid_tool_calls else 0.1


def compute_score(tokenizer, solution_str, ground_truth, eval_lang) -> tuple[float, str]:
    # handling both the base model and the instruction-tuned model
    if "<|im_start|>assistant\n" in solution_str:
        solution_str_split = solution_str.split("<|im_start|>assistant\n")
    else:
        solution_str_split = solution_str.split("Assistant:")

    prompt = solution_str_split[0].strip()
    response = solution_str_split[1]
    valid_template, reason = validate_format(response)
    if not valid_template:
        return 0, f'bad format: {reason}'

    if response.endswith(tokenizer.eos_token):
        response = response[:-len(tokenizer.eos_token)]
    else:
        return 0, f'over length'

    answer_part = extract_answer(response)
    # if answer_part is not None:
    #     try:
    #         answer = remove_boxed(last_boxed_only_string(answer_part))
    #     except Exception as e:
    #         return 0, f'find box error: {e}'
    # else:
    #     return 0, f'cannot extract answer'

    answer = answer_part.strip()
    f1_score = get_f1_score(answer, ground_truth, eval_lang)

    # penalty activated if environment variable LOCAL_WEB_REWARD is set to TRUE
    local_web_reward = os.environ.get('LOCAL_WEB_REWARD', 'FALSE').upper()
    if local_web_reward == 'TRUE':
        f1_score += get_local_web_score(solution_str)

    tool_call_comp = tool_call_comprehensiveness(prompt, response)
    if f1_score > 0:
        return f1_score, f'correct answer, get f1 score: {f1_score}'
    else:
        return tool_call_comp, f'wrong answer but good format: {answer}'



