import argparse
import time
from functools import partial
from typing import List
from transformers import AutoTokenizer
import openai
import sys
import concurrent.futures
import re
import json
import numpy as np
from tqdm import tqdm
import loguru
import random
import os


sys.path.append("./")
from baselines.online_eval import batch_search, batch_web_search, graph_search, sample_response
from agentic_rag.eval_rollout_sequence import evaluate_rollout_sequence

def sample_chat_response(llm_client, serve_model_name, messages):
    _response = None
    while True:
        try:
            _response = llm_client.chat.completions.create(
                messages=messages,
                model=serve_model_name,
                temperature=0.3,
                max_tokens=2048,
            )
            break
        except Exception as e:
            loguru.logger.warning(f"Error in response: {e}")
            time.sleep(random.randint(1, 5))
            continue
    return _response

def extract_answer(solution_str: str):
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, solution_str, re.DOTALL)
    if not match:
        return ""

    answer_part = match.group(1)
    return answer_part.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running exp")
    parser.add_argument("--method_name", type=str, default="reasoning")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--save_note", type=str, default='your-save-note-for-identification')
    parser.add_argument("--remote_llm_url", type=str, default="")
    parser.add_argument("--web_retriever_url", type=str, default="http://127.0.0.1:15005")
    parser.add_argument("--serve_model_name", type=str, default="deepseek-r1")
    parser.add_argument("--sources", type=str, default="all")

    args = parser.parse_args()
    method_name = args.method_name
    data_dir = args.data_dir
    split = args.split
    save_note = args.save_note if args.save_note else args.method_name
    remote_llm_url = args.remote_llm_url
    web_retriever_url = args.web_retriever_url
    serve_model_name = args.serve_model_name
    sources = args.sources

    llm_client = openai.OpenAI(
        api_key="sk-",
        base_url=remote_llm_url,
    )

    sys_template = ("You are a helpful assistant that can solve the given question step by step with the help of the search tool results. "
        "Please answer the following question. You should provide the final answer in the format of <answer> FINAL ANSWER </answer>.\n"
        "Search Tool Results: {results}\n\nQuestion: {question}")

    remote_retriever_urls = {
        "musique": "http://127.0.0.1:18007",
        "omnieval": "http://127.0.0.1:18009",
        "bioasq": "http://127.0.0.1:18010",
        "nq": "http://127.0.0.1:18011",
        "hotpotqa": "http://127.0.0.1:18012",
        "pubmedqa": "http://127.0.0.1:18013",
    }

    dataset_names = ["musique", "omnieval", "bioasq", "nq", "hotpotqa", "pubmedqa"]
    # dataset_names = ["bioasq", "nq", "hotpotqa", "pubmedqa"]
    for dataset_name in dataset_names:
        # Prepare datasets and evaluation
        if dataset_name == "omnieval":
            os.environ["EVAL_LANG"] = "zh"
        else:
            os.environ["EVAL_LANG"] = "en"
        loguru.logger.info(f"Setting EVAL_LANG to {os.environ['EVAL_LANG']}")
        remote_retriever_url = remote_retriever_urls[dataset_name]
        loguru.logger.info(f"Using remote retriever url: {remote_retriever_url}")

        # read data
        data_path = f'{data_dir}/{dataset_name}/{split}.jsonl'
        save_path = f'{data_dir}/{dataset_name}/rollout/{method_name}_{sources}_{split}.jsonl'

        with open(data_path, 'r') as f:
            data = [json.loads(line) for line in f]
            loguru.logger.info(f"Loaded {len(data)} samples from {data_path}")
        # data = data[:4]

        answers = ["" for _ in range(len(data))]
        rollout_sequences = ["" for _ in range(len(data))]

        def rollout_thread(idx):
            question = data[idx]['question']
            # loguru.logger.info(f"Question: {question}")
            if "用户提问：" in question:
                query = question.split("用户提问：")[-1]
            else:
                query = question
            if sources == "web":
                results = batch_web_search(web_retriever_url, query, "default", 10)[0]
            elif sources == "local":
                results = batch_search(remote_retriever_url, query, 10)[0] + "\n\n"
                results += graph_search(remote_retriever_url, query, 10)[0]
            elif sources == "all":
                results = batch_web_search(web_retriever_url, query, "default", 10)[0] + "\n\n"
                results += (batch_search(remote_retriever_url, query, 10)[0] + "\n\n")
                results += graph_search(remote_retriever_url, query, 10)[0]
            else:
                raise ValueError(f"Invalid sources: {sources}")
            input_messages = [{
                "role": "user",
                "content": sys_template.format(
                    results=results,
                    question=question,
                )
            }]

            chat_res = sample_chat_response(llm_client, serve_model_name, input_messages)
            chat_content = chat_res.choices[0].message.content
            reasoning_content = chat_res.choices[0].message.reasoning_content
            # loguru.logger.info(f"Chat content: {chat_content}, Reasoning content: {reasoning_content}")

            answer = extract_answer(chat_content)
            answers[idx] = answer
            rollout_sequences[idx] = f"<think>{reasoning_content}</think>{chat_content}"
            # loguru.logger.info(f"Final input: {next_input} answer: {answer}")
            return answer


        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
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

        loguru.logger.info(f"Save results to {save_path}")
        # save results
        with open(save_path, 'w') as f:
            for i, item in enumerate(data):
                item['answer'] = answers[i]
                item['rollout_sequence'] = rollout_sequences[i]
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

