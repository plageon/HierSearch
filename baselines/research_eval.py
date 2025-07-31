import argparse
from functools import partial
from typing import List

import openai
from transformers import AutoTokenizer
import sys
import concurrent.futures
import re
import json
import numpy as np
from tqdm import tqdm
import loguru
import os

sys.path.append("./")
from baselines.online_eval import batch_search, batch_web_search, graph_search, sample_response
from agentic_rag.eval_rollout_sequence import evaluate_rollout_sequence, extract_answer



def merged_search(remote_retriever_url, query: str, sources, top_n=5) -> List[str]:
    # conduct batch_search, batch_web_search, graph_search
    # concat results and return
    if sources == "original":
        batch_search_results = batch_search(remote_retriever_url, query, top_n)
        return batch_search_results
    elif sources == "all":
        batch_search_results = batch_search(remote_retriever_url, query, top_n)
        batch_web_search_results = batch_web_search(remote_web_retriever_url, query, "default", top_n)
        graph_search_results = graph_search(remote_retriever_url, query, top_n)
        merged_results = [f"Local Search Results: {i}\n\n{j}\n\nWeb Search Results: {k}"
                          for i, j, k in zip(batch_search_results, graph_search_results, batch_web_search_results)]
        return merged_results
    else:
        raise ValueError(f"Unsupported source: {sources}. Supported sources are 'original' and 'all'.")


def parse_response(completion, stop_reason, finish_reason):
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
                search_result = search_function(search_content)
                return completion, f" <result>\n{search_result[0]}\n</result>"
    return completion, ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running exp")
    parser.add_argument("--method_name", type=str, default="research")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--save_note", type=str, default='your-save-note-for-identification')
    parser.add_argument("--remote_llm_url", type=str, default="http://10.43.16.133:14004/v1")
    parser.add_argument("--model_path", type=str, default="models/ReSearch-Qwen-7B-Instruct/")
    parser.add_argument("--remote_web_retriever_url", type=str, default="http://10.10.15.46:15005")
    parser.add_argument("--serve_model_name", type=str, default="ReSearch-Qwen-7B-Instruct")
    parser.add_argument("--sys_template_name", type=str, default="default",)
    parser.add_argument("--max_turns", type=int, default=8, help="maximum number of turns for the conversation")
    parser.add_argument("--sources", type=str, default="original", help="sources to use for search, can be 'original' or 'all'. If 'all', it will use batch_search, batch_web_search and graph_search.")

    args = parser.parse_args()
    method_name = args.method_name
    data_dir = args.data_dir
    split = args.split
    save_note = args.save_note if args.save_note else args.method_name
    remote_llm_url = args.remote_llm_url
    remote_web_retriever_url = args.remote_web_retriever_url
    model_path = args.model_path
    serve_model_name = args.serve_model_name
    sys_template_name = args.sys_template_name
    sources = args.sources

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm_client = openai.OpenAI(
        api_key="EMPTY",
        base_url=remote_llm_url,
    )

    sys_template = """You are a helpful assistant that can solve the given question step by step with the help of the wikipedia search tool. \
Given a question, you need to first think about the reasoning process in the mind and then provide the answer. \
During thinking, you can invoke the wikipedia search tool to search for fact information about specific topics if needed. \
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively, \
and the search query and result are enclosed within <search> </search> and <result> </result> tags respectively. \
For example, <think> This is the reasoning process. </think> <search> search query here </search> <result> search result here </result> \
<think> This is the reasoning process. </think> <answer> The final answer is \\[ \\boxed{answer here} \\] </answer>. \
In the last part of the answer, the final exact answer is enclosed within \\boxed{} with latex format."""

    # test sample response
    # _response = sample_response("Hello, ", ["\n"])
    # loguru.logger.info(_response)

    all_start_patterns = ['<search>']
    all_end_patterns = ['</search>']



    # supported tools
    supported_tools = ['search']
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

    # dataset_names = ["musique", "omnieval", "bioasq", "nq", "hotpotqa", "pubmedqa"]
    dataset_names = ["musique"]
    for dataset_name in dataset_names:
        functions_dict = {
            '<search>': partial(merged_search, remote_retriever_urls[dataset_name], sources=sources, top_n=5),
        }
        if dataset_name == "omnieval":
            os.environ["EVAL_LANG"] = "zh"
        else:
            os.environ["EVAL_LANG"] = "en"
        loguru.logger.info(f"Setting EVAL_LANG to {os.environ['EVAL_LANG']}")
        # read data
        data_path = f'{data_dir}/{dataset_name}/{split}.jsonl'
        save_path = f'{data_dir}/{dataset_name}/rollout/{method_name}_{sources}_{split}.jsonl'
        with open(data_path, 'r') as f:
            data = [json.loads(line) for line in f]
        # data = data[:4]

        answers = ["" for _ in range(len(data))]
        rollout_sequences = ["" for _ in range(len(data))]

        def rollout_thread(idx):
            question = data[idx]['question']
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
            while round_cnt < args.max_turns:
                round_cnt += 1
                # loguru.logger.info(f"Round {round_cnt}")
                response = sample_response(llm_client, serve_model_name, model_input, stop_words)
                if isinstance(response, str) and "Error: maximum context length exceeded" in response:
                    loguru.logger.info(f"Round {round_cnt} exceeded max context length, stopping rollout for idx {idx}")
                    break
                # loguru.logger.info(f"Round {round_cnt}, response: {response}")
                completion = response.choices[0].text
                stop_reason = response.choices[0].stop_reason
                finish_reason = response.choices[0].finish_reason
                # loguru.logger.info(f"Round {round_cnt}, completion: {completion}, stop_reason: {stop_reason}")
                # parse response
                completion, search_res = parse_response(completion, stop_reason, finish_reason)
                # loguru.logger.info(f"Round {round_cnt}, search_res: {search_res}")
                next_input = model_input + completion
                # loguru.logger.info(f"Next input: {next_input}")
                if search_res == "":
                    break
                else:
                    next_input += search_res
                    model_input = next_input
            answer = extract_answer(next_input, remove_boxed_answer=True)
            rollout_sequences[idx] = next_input
            # loguru.logger.info(f"Final input: {next_input} answer: {answer}")
            answers[idx] = answer
            return answer


        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_list = [executor.submit(rollout_thread, idx) for idx in range(len(data))]
            with tqdm(total=len(future_list), desc="Rollout") as pbar:
                try:
                    for future in concurrent.futures.as_completed(future_list):
                        future.result()
                        pbar.update(1)
                except Exception as e:
                    loguru.logger.info(f"Error in rollout: {e}")
                    import traceback

                    traceback.print_exc()
                    pbar.update(1)

        # eval rollout sequence
        gold_answers = [item['golden_answers'] for item in data]
        overall_qa_results = evaluate_rollout_sequence(gold_answers, answers)
        loguru.logger.info(f"gold_answers: {gold_answers}, \nanswers: {answers}")
        loguru.logger.info(f"Overall QA results: {overall_qa_results}")

        loguru.logger.info(f"Saving results to {save_path}")
        # save results
        with open(save_path, 'w') as f:
            for i, item in enumerate(data):
                item['answer'] = answers[i]
                item['rollout_sequence'] = rollout_sequences[i]
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

