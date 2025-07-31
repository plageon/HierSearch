import argparse
import os
from functools import partial
from typing import Union, List, Dict
import openai
import loguru
import time
import random
import json
import re

import requests
from tqdm import tqdm
from transformers import AutoTokenizer
import concurrent.futures
import sys
sys.path.append('./')
from agentic_rag.eval_rollout_sequence import evaluate_rollout_sequence, extract_answer
from agentic_rag.templates import prompt_template_dict

def parse_agent_rollout_sequence(sequence: str, tag: str, attach_reasoning=True) -> List[Dict[str, str]]:
    """
    Parse the agent rollout sequence to extract the prompt and response.

    Args:
        sequence (str): The agent rollout sequence.

    Returns:
        Dict[str, str]: A dictionary containing the prompt and response.
    """
    if sequence == "":
        return []
    rollout_sequence = ""
    if "<|im_start|>assistant\n" in sequence:
        parts = sequence.split("<|im_start|>assistant\n")
        if len(parts) != 2:
            raise ValueError("Invalid sequence format")
        prompt = parts[0].strip()
        rollout_sequence = parts[1].strip()
    else:
        return "Sequence does not contain <|im_start|>assistant"

    start_patterns = ['<chunk_search>', '<graph_search>', '<get_adjacent_passages>', '<web_search>', '<browse_url>']
    end_patterns = ['</chunk_search>', '</graph_search>', '</get_adjacent_passages>', '</web_search>',
                    '</browse_url>']
    local_tool_calls = ['<chunk_search>', '<graph_search>', '<get_adjacent_passages>']
    web_tool_calls = ['<web_search>', '<browse_url>']
    other_tags = ['<think>', '<answer>', '<result>']
    evidence_sources = {
        '<chunk_search>': 'Local Chunk Corpus',
        '<graph_search>': 'Local Knowledge Graph',
        '<get_adjacent_passages>': 'Local Chunk Corpus',
        '<web_search>': 'Search Engine',
        '<browse_url>': 'Web Page'
    }

    infos = []
    # contents between <result> and </result> are evidences
    # contents between <think> and </think> are hypotheses
    # contents between <answer> and </answer> are conclusions
    # get all tool calls and their positions
    text = rollout_sequence
    current_pos = 0
    while True:
        think_pos = text.find('<think>', current_pos)
        think_end_pos = text.find('</think>', current_pos)
        if think_pos != -1 and think_end_pos != -1:
            # extract hypotheses
            hypothesis = text[think_pos + len('<think>'):think_end_pos].strip()
            if attach_reasoning:
                infos.append({
                    "contents": hypothesis,
                    "source": f"{tag} Agent Hypothesis",
                    "start_pos": think_pos + len('<think>'),
                    "end_pos": think_end_pos
                })
        else:
            hypothesis = ""

        all_search_pos = [(text.find(sp, current_pos), sp) for sp in start_patterns if
                          text.find(sp, current_pos) != -1]
        all_search_pos.sort(key=lambda x: x[0])  # Sort by position
        start_pattern = all_search_pos[0][1] if all_search_pos else None
        search_pos = all_search_pos[0][0] if all_search_pos else -1
        all_search_pos = [pos for pos, _ in all_search_pos]
        if search_pos == -1:
            break

        result_pos = text.find('<result>', search_pos)
        all_search_end_pos = [text.find(ep, search_pos) for ep in end_patterns if text.find(ep, search_pos) != -1]
        search_end_pos = min(all_search_end_pos) if all_search_end_pos else -1
        result_end_pos = text.find('</result>', result_pos)
        search_query = text[search_pos + len(start_pattern):search_end_pos].strip()

        if -1 in (result_pos, search_end_pos, result_end_pos):
            break

        if not (think_pos < think_end_pos < search_pos < search_end_pos < result_pos < result_end_pos):
            break
        if len(all_search_pos) > 1:
            all_search_pos = sorted(all_search_pos)
            if not (all_search_pos[0] < search_end_pos < result_pos < result_end_pos < all_search_pos[1]):
                break

        current_pos = result_end_pos

        # extract evidences
        all_results = [r.strip() for r in text[result_pos + len('<result>'):result_end_pos].split("\n\n") if
                       r.strip()]
        for i, result in enumerate(all_results):
            infos.append({
                "contents": result,
                "source": evidence_sources[start_pattern],
                "start_pos": result_pos + len('<result>'),
                "end_pos": result_end_pos
            })

    answer_start = text.find('<answer>')
    answer_end = text.find('</answer>')
    if answer_start != -1 and answer_end != -1:
        # extract conclusion
        if attach_reasoning:
            infos.append({
                "contents": text[answer_start + len('<answer>'):answer_end].strip(),
                "source": f"{tag} Agent Conclusion",
                "start_pos": answer_start + len('<answer>'),
                "end_pos": answer_end
            })
    return infos

def handle_ablation(info_list, local_rollout, web_rollout):
    evidence_sources = {
        '<chunk_search>': 'Local Chunk Corpus',
        '<graph_search>': 'Local Knowledge Graph',
        '<get_adjacent_passages>': 'Local Chunk Corpus',
        '<web_search>': 'Search Engine',
        '<browse_url>': 'Web Page'
    }

    if ablation == 'local':
        # remove local info from info_list
        ablation_info_list = []
        for i in range(len(info_list)):
            if "Local" not in info_list[i]["source"]:
                ablation_info_list.append(info_list[i])
        return ablation_info_list
    elif ablation == 'web':
        # remove web info from info_list
        ablation_info_list = []
        for i in range(len(info_list)):
            if "Web Page" not in info_list[i]["source"] and "Search Engine" not in info_list[i]["source"]:
                ablation_info_list.append(info_list[i])
        return ablation_info_list
    elif ablation == "refiner":
        # remove local and web info from info_list
        ablation_info_list = parse_agent_rollout_sequence(local_rollout, tag="Local") + parse_agent_rollout_sequence(web_rollout, tag="Web")
        return ablation_info_list
    elif ablation == "refiner-reasoning":
        # remove local and web info from info_list
        ablation_info_list = parse_agent_rollout_sequence(local_rollout, tag="Local", attach_reasoning=False) + parse_agent_rollout_sequence(web_rollout, tag="Web", attach_reasoning=False)
        return ablation_info_list
    else:
        raise ValueError(f"Invalid ablation study name {ablation}")


def search_agent(func_url, query: Union[str, List[str]], sample_id: Union[str, List[str]], data_source: List[str],
                 destination: str, return_rollouts: bool=False) -> List[str]:
    if len(query) == 0:
        return 'invalid query'
    # url = f'{self.config.web_search_url}/web_search'
    if isinstance(query, str):
        query = [query]
        sample_id = [sample_id]
        data_source = [data_source]

    result_list = []

    data = {'query': query, 'data_source': data_source, 'sample_id': sample_id, 'return_rollouts': return_rollouts, 'filter_ratio': filter_ratio}
    url = f"{func_url}/{destination}"
    response = requests.post(url, json=data, timeout=1000)
    response_jsons = response.json()
    for i in range(len(query)):
        response_json = response_jsons[i]

        # loguru.logger.info(f"Requesting search agent url {url} with data: {data}")
        if "Agent Failed" in json.dumps(response_json) and "Not necessary" not in json.dumps(query[i]) and "不需要" not in json.dumps(query[i]):
            loguru.logger.warning(f"Agent Failed for question: {query[i]}, sample_id: {sample_id[i]}")
            result_list.append(f"Agent Failed for question: {query[i]}")
        if return_rollouts:
            if destination == 'local_search_agent':
                # pop the last line from response_json
                local_agent_rollout = response_json.pop()
                web_agent_rollout = {"source": "Web Agent Rollout", "contents": ""}
                assert local_agent_rollout["source"] == "Local Agent Rollout", f"Invalid source: {local_agent_rollout['source']}"
                local_agent_rollouts[sample_id[i]].append(local_agent_rollout["contents"])
                # loguru.logger.info(local_agent_rollout["contents"])
            elif destination == 'web_search_agent':
                local_agent_rollout = {"source": "Local Agent Rollout", "contents": ""}
                web_agent_rollout = response_json.pop()
                assert web_agent_rollout["source"] == "Web Agent Rollout", f"Invalid source: {web_agent_rollout['source']}"
                web_agent_rollouts[sample_id[i]].append(web_agent_rollout["contents"])
            elif destination == 'all_search_agent':
                web_agent_rollout = response_json.pop()
                local_agent_rollout = response_json.pop()
                assert local_agent_rollout["source"] == "Local Agent Rollout", f"Invalid source: {local_agent_rollout['source']}"
                assert web_agent_rollout["source"] == "Web Agent Rollout", f"Invalid source: {web_agent_rollout['source']}"
                local_agent_rollouts[sample_id[i]].append(local_agent_rollout["contents"])
                web_agent_rollouts[sample_id[i]].append(web_agent_rollout["contents"])
                # loguru.logger.info(local_agent_rollout["contents"])
            else:
                raise ValueError(f"Invalid destination: {destination}")

            if ablation:
                response_json = handle_ablation(response_json, local_agent_rollout["contents"], web_agent_rollout["contents"])
        curr_result = ''
        for line in response_json:
            source = line["source"]
            contents = line["contents"]
            curr_result += f"{source}: {contents}\n\n"
        result_list.append(curr_result.strip())
    return result_list

def batch_search(func_url, query: Union[str, List[str]], top_n=5) -> List[str]:
    if len(query) == 0:
        return 'invalid query'

    url = f'{func_url}/batch_search'
    if isinstance(query, str):
        query = [query]
    data = {'query': query, 'top_n': top_n}
    response = requests.post(url, json=data)

    result_list = []
    for item in response.json():
        curr_result = ''
        for line in item:
            contents = " ".join([c.strip() for c in line["contents"].split("\n")]).strip()
            curr_result += f"{contents}\n\n"
        result_list.append(curr_result.strip())

    return result_list

def graph_search(func_url, query: Union[str, List[str]], top_n=5) -> List[str]:
    if isinstance(query, str):
        query = [query]
    if len(query) == 0:
        return 'invalid query'

    url = f'{func_url}/graph_search'
    data = {'query': query, 'top_n': top_n, 'return_score': False, }
    response = requests.post(url, json=data)
    result_list = []
    for item in response.json():
        curr_result = ''
        for line in item:
            curr_result += f"[Subject]: {line['subject_phrase']} [Predicate]: {line['predicate_phrase']} [Object]: {line['object_phrase']}\n\n"
        result_list.append(curr_result.strip())
    return result_list

def get_adjacent_passages(func_url, query: List[str], top_n=5) -> List[str]:
    if isinstance(query, str):
        query = [query]
    if query == '':
        return 'invalid query'
    dense_search_url = f"{func_url}/get_adjacent_passages"
    data = {'query': query, 'top_n': top_n, }
    response = requests.post(dense_search_url, json=data)
    result_list = []
    for item in response.json():
        curr_result = ''
        for line in item:
            contents = " ".join([c.strip() for c in line["contents"].split("\n")]).strip()
            curr_result += f"{contents}\n\n"
        result_list.append(curr_result.strip())
    return result_list

def batch_web_search(func_url, query: Union[str, List[str]], sample_id: Union[str, List[str]]=None, top_n=5) -> List[str]:
    if len(query) == 0:
        return 'invalid query'
    url = f'{func_url}/web_search'
    if isinstance(query, str):
        query = [query]
    if sample_id is None:
        sample_id = ['default'] * len(query)
    if isinstance(sample_id, str):
        sample_id = [sample_id]

    result_list = []
    for i in range(len(query)):
        data = {'query': query[i], 'sample_id': sample_id[i], 'top_n': top_n}
        # loguru.logger.info(f"Requesting web search url {url} with data: {data}")
        response = requests.post(url, json=data)
        if "Invalid sample id" in response.text:
            loguru.logger.error(f"Invalid sample id: {sample_id[i]}")
            result_list.append(f"Invalid sample id: {sample_id[i]}")
        curr_result = ''
        for line in response.json():
            contents = " ".join([c.strip() for c in line["contents"].split("\n")]).strip()
            curr_result += f"{contents}\n\n"
        result_list.append(curr_result.strip())
    return result_list

def webpage_browse(func_url, query: Union[str, List[str]], browse_url: Union[str, List[str]], sample_id: Union[str, List[str]]=None, top_n=5) -> List[str]:
    if len(query) == 0:
        return 'invalid query'

    url = f'{func_url}/batch_webpage_browse'
    if isinstance(query, str):
        query = [query]
    if sample_id is None:
        sample_id = ['default'] * len(query)
    if isinstance(browse_url, str):
        browse_url = [browse_url]

    result_list = []
    data = {'query': query, 'sample_id': sample_id, 'url': browse_url, 'top_n': top_n}
    # loguru.logger.info(f"Requesting batch webpage browse with data: {data}")
    response = requests.post(url, json=data).json()
    for i in range(len(query)):
        item_resp = response[i]
        if "Invalid sample id" in json.dumps(item_resp):
            loguru.logger.error(f"Invalid sample id: {sample_id[i]}")
            result_list.append(f"Invalid sample id: {sample_id[i]}")
        elif "Invalid URL" in json.dumps(item_resp):
            loguru.logger.error(f"Invalid URL: {browse_url[i]}")
            result_list.append(f"Invalid URL: {browse_url[i]}")
        curr_result = ''
        for line in item_resp:
            curr_result += f"{line['contents']}\n\n"
        result_list.append(curr_result.strip())
    return result_list



def sample_response(llm_client, serve_model_name, prompts, stop_words, temperature=0.3):
    _response = None
    while True:
        try:
            _response = llm_client.completions.create(
                prompt=prompts,
                model=serve_model_name,
                temperature=temperature,
                max_tokens=512,
                stop=stop_words,
            )
            break
        except Exception as e:
            loguru.logger.warning(f"Error in response: {e}")
            if "maximum context" in str(e):
                return "Error: maximum context length exceeded"
            time.sleep(random.randint(1, 5))
            continue
    return _response


def parse_response(sample_id, completion, stop_reason, finish_reason):
    functions_dict = {
        '<chunk_search>': partial(batch_search, remote_retriever_url, top_n=5),
        '<graph_search>': partial(graph_search, remote_retriever_url, top_n=5),
        '<get_adjacent_passages>': partial(get_adjacent_passages, remote_retriever_url),
        '<web_search>': partial(batch_web_search, remote_web_retriever_url, sample_id=sample_id, top_n=5),
        '<browse_url>': partial(webpage_browse, remote_web_browse_url, sample_id=sample_id, top_n=5),
        '<all_search_agent>': partial(search_agent, remote_agent_url, sample_id=sample_id, data_source=dataset_name, destination='all_search_agent', return_rollouts=True),
        '<local_search_agent>': partial(search_agent, remote_agent_url, sample_id=sample_id, data_source=dataset_name, destination='local_search_agent', return_rollouts=True),
        '<web_search_agent>': partial(search_agent, remote_agent_url, sample_id=sample_id, data_source=dataset_name, destination='web_search_agent', return_rollouts=True),
    }
    def extract_search_content(start_tag, end_tag, text: str) -> str:
        try:
            end_pos = text.rindex(end_tag)
            start_pos = text.rindex(start_tag, 0, end_pos)
            return text[start_pos + len(start_tag):end_pos].strip()
        except ValueError:
            return ""

    if finish_reason == 'stop' and isinstance(stop_reason, str) and any(
            [p in stop_reason for p in end_patterns]):
        ## process the search
        for start_pattern, end_pattern in zip(start_patterns, end_patterns):
            if start_pattern in completion:
                completion += end_pattern
                search_content = extract_search_content(start_pattern, end_pattern, completion)
                if search_content.strip() == '':
                    continue
                search_function = functions_dict[start_pattern]
                if start_pattern == "<browse_url>":
                    url, query = "", ""
                    for comma in ['|', '｜', ',', '，']:
                        parsing_success = False
                        try:
                            split_item = search_content.split(comma)
                            assert split_item[0].strip() != '' and split_item[1].strip() != ''
                            url, query = split_item[0].strip(), split_item[1].strip()
                            assert "http" in url or "https" in url
                            parsing_success = True
                            break
                        except Exception as e:
                            pass
                    if not parsing_success:
                        loguru.logger.warning(f"Error in parsing search content: {search_content}")
                        search_result = [f"Error in parsing search content: {search_content}"]
                    else:
                        search_result = search_function(query=query, browse_url=url, sample_id=sample_id)
                else:
                    search_result = search_function(search_content)
                return completion, f" <result>\n{search_result[0]}\n</result>"
    return completion, ""



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running exp")
    parser.add_argument("--method_name", type=str, default="")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--dataset_name", type=str, default="musique")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--save_note", type=str, default='your-save-note-for-identification')
    parser.add_argument("--remote_llm_url", type=str, default="http://172.24.88.76:18011/v1")
    parser.add_argument("--remote_retriever_url", type=str, default="http://172.24.88.76:18009/")
    parser.add_argument("--remote_web_retriever_url", type=str, default="http://")
    parser.add_argument("--remote_web_browse_url", type=str, default="http://127.0.0.1:15005")
    parser.add_argument("--remote_agent_url", type=str, default="http://")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--serve_model_name", type=str, default="")
    parser.add_argument("--sys_template_name", type=str, default="hybrid_graph_search_agent_template_sys",)
    parser.add_argument("--max_turns", type=int, default=8, help="maximum number of turns for the conversation")
    parser.add_argument("--ablation", type=str, default="", help="ablation study")
    parser.add_argument("--filter_ratio", type=float, default=0.0, help="filter ratio for the conversation")

    args = parser.parse_args()
    method_name = args.method_name
    data_dir = args.data_dir
    dataset_name = args.dataset_name
    split = args.split
    save_note = args.save_note if args.save_note else args.method_name
    remote_llm_url = args.remote_llm_url
    remote_web_retriever_url = args.remote_web_retriever_url
    remote_web_browse_url = args.remote_web_browse_url
    remote_agent_url = args.remote_agent_url
    model_path = args.model_path
    serve_model_name = args.serve_model_name
    sys_template_name = args.sys_template_name
    ablation = args.ablation
    max_turns = args.max_turns
    loguru.logger.info(f"configs: {args}")

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm_client = openai.OpenAI(
        api_key="EMPTY",
        base_url=remote_llm_url,
    )

    sys_template = prompt_template_dict[args.sys_template_name] if args.sys_template_name in prompt_template_dict \
        else prompt_template_dict['web_graph_search_agent_template_sys']

    # test sample response
    # _response = sample_response("Hello, ", ["\n"])
    # loguru.logger.info(_response)

    all_start_patterns = ['<chunk_search>', '<graph_search>', '<get_adjacent_passages>', '<browse_url>', '<web_search>', '<all_search_agent>', '<local_search_agent>', '<web_search_agent>']
    all_end_patterns = ['</chunk_search>', '</graph_search>', '</get_adjacent_passages>', '</browse_url>', '</web_search>', '</all_search_agent>', '</local_search_agent>', '</web_search_agent>']

    # supported tools
    supported_tools = ['chunk_search', 'graph_search', 'get_adjacent_passages', 'web_search', 'browse_url',
                       'all_search_agent', 'local_search_agent', 'web_search_agent']
    start_patterns, end_patterns = [], []
    for start_pattern, end_pattern in zip(all_start_patterns, all_end_patterns):
        tool_name = start_pattern.split('<')[1].split('>')[0]
        if tool_name in supported_tools:
            start_patterns.append(start_pattern)
            end_patterns.append(end_pattern)
    stop_words = end_patterns

    remote_retriever_urls = {
        "musique": "http://127.0.0.1:18007",
        "omnieval": "http://127.0.0.1:18009",
        "bioasq": "http://127.0.0.1:18010",
        "nq": "http://127.0.0.1:18011",
        "hotpotqa": "http://127.0.0.1:18012",
        "pubmedqa": "http://127.0.0.1:18013",
    }
    filter_ratios = {
        "musique": 0.8,
        "omnieval": 0.8,
        "bioasq": 0.4,
        "nq": 0.4,
        "hotpotqa": 0.6,
        "pubmedqa": 0.6,
    }

    dataset_names = ["musique", "omnieval", "bioasq", "nq", "hotpotqa", "pubmedqa"]
    # dataset_names = ["musique", "omnieval", "pubmedqa"]
    for dataset_name in dataset_names:
        if dataset_name == "omnieval":
            os.environ["EVAL_LANG"] = "zh"
        else:
            os.environ["EVAL_LANG"] = "en"
        loguru.logger.info(f"Setting EVAL_LANG to {os.environ['EVAL_LANG']}")
        # read data
        data_path = f'{data_dir}/{dataset_name}/{split}.jsonl'
        save_path = f'{data_dir}/{dataset_name}/rollout/{method_name}_{split}.jsonl'
        with open(data_path, 'r') as f:
            data = [json.loads(line) for line in f]
        # data = data[:4]

        remote_retriever_url = remote_retriever_urls[dataset_name]
        if args.filter_ratio == 0.0:
            filter_ratio = filter_ratios[dataset_name]
        else:
            filter_ratio = args.filter_ratio
        loguru.logger.info(f"Using remote retriever url: {remote_retriever_url}, filter_ratio: {filter_ratio}")

        # eval rollout sequence
        gold_answers = [item['golden_answers'] for item in data]
        for _gold_answers in gold_answers:
            if len(_gold_answers) == 0:
                _gold_answers.append("")
                loguru.logger.info(f"Empty gold answers, set to empty string")

        answers = ["" for _ in range(len(data))]
        rollout_sequences = ["" for _ in range(len(data))]
        sample_ids = [item['id'] for item in data]

        local_agent_rollouts, web_agent_rollouts = {sid: [] for sid in sample_ids}, {sid: [] for sid in sample_ids}

        def rollout_thread(idx):
            question = data[idx]['question']
            sample_id = data[idx]['id']
            # loguru.logger.info(f"Question: {question}")

            original_messages = [
                {"role": "system", "content": sys_template},
                {"role": "user", "content": question},
            ]
            original_input = tokenizer.apply_chat_template(original_messages, tokenize=False, add_generation_prompt=True)
            # loguru.logger.info(f"Original input: {original_input}")

            # online eval
            round_cnt = 0
            model_input = original_input
            while round_cnt < max_turns:
                round_cnt += 1
                # loguru.logger.info(f"Round {round_cnt}")
                response = sample_response(llm_client, serve_model_name, model_input, stop_words)
                # loguru.logger.info(f"Round {round_cnt}, response: {response}")
                completion = response.choices[0].text
                stop_reason = response.choices[0].stop_reason
                finish_reason = response.choices[0].finish_reason
                # loguru.logger.info(f"Round {round_cnt}, completion: {completion}, stop_reason: {stop_reason}")
                # parse response
                completion, search_res = parse_response(sample_id, completion, stop_reason, finish_reason)
                # loguru.logger.info(f"Round {round_cnt}, search_res: {search_res}")
                next_input = model_input + completion
                # loguru.logger.info(f"Next input: {next_input}")
                if search_res == "":
                    break
                else:
                    next_input += search_res
                    model_input = next_input
            answer = extract_answer(next_input)
            answers[idx] = answer
            rollout_sequences[idx] = next_input
            # loguru.logger.info(f"Final input: {next_input} answer: {answer}")
            return answer

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            future_list = [executor.submit(rollout_thread, idx) for idx in range(len(data))]
            with tqdm(total=len(future_list), desc="Rollout") as pbar:
                for i in range(len(future_list)):
                    try:
                        future = future_list[i]
                        future.result()  # wait for the result
                    except Exception as e:
                        loguru.logger.error(f"Error in rollout thread {i}: {e}")
                        import traceback
                        traceback.print_exc()
                    pbar.update(1)
            pbar.close()

        overall_qa_results = evaluate_rollout_sequence(gold_answers, answers)
        loguru.logger.info(f"gold_answers: {gold_answers}, \nanswers: {answers}")
        loguru.logger.info(f"Overall QA results: {overall_qa_results}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # save results
        loguru.logger.info(f"Saving results to {save_path}")
        with open(save_path, 'w') as f:
            for i, item in enumerate(data):
                item['answer'] = answers[i]
                item['rollout_sequence'] = rollout_sequences[i]
                item['local_agent_rollouts'] = local_agent_rollouts[sample_ids[i]]
                item['web_agent_rollouts'] = web_agent_rollouts[sample_ids[i]]
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

