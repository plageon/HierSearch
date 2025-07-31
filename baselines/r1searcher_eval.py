import os
import argparse
import torch.distributed as dist
import json
from vllm import LLM, SamplingParams
from datasets import Dataset
from transformers import AutoTokenizer
import os
import copy
import re
import requests
import json
import loguru
from tqdm import tqdm
import openai
import concurrent.futures
import sys

sys.path.append("./")
from baselines.online_eval import batch_search, batch_web_search, graph_search, sample_response
from agentic_rag.eval_rollout_sequence import evaluate_rollout_sequence, extract_answer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default="")
    parser.add_argument("--start_sample", type=int, default=-1)
    parser.add_argument("--end_sample", type=int, default=100000)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--gpu_id", type=str, default="3")
    parser.add_argument("--model_path", type=str, default="models/R1-Searcher")
    parser.add_argument("--remote_web_retriever_url", type=str, default="http://10.10.15.46:15005")
    parser.add_argument("--remote_llm_url", type=str, default="http://10.43.16.133:12002/v1")
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--prompt_type", type=str, default="v0")
    parser.add_argument("--sources", type=str, default="all", choices=["original", "all"],)
    return parser.parse_args()

def process_text(examples,tokenizer,type=None):

    base_prompt_v0 = """The User asks a question, and the Assistant solves it.
The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer.
The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
During the thinking process, the Assistant can perform searching for uncertain knowledge if necessary with the format of "<|begin_of_query|> search query (only keywords) here <|end_of_query|>". **A query must involve only a single triple**.
Then, the system will provide the Assistant with helpful information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>".\n\nUser:{question}\nAssistant: <think>"""

    base_prompt_v1 = """The User asks a question, and the Assistant solves it.
The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer.
The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
During the reasoning process, the Assistant will break down the original question into sub-questions and address them step by step.
For each sub-question, **the Assistant can perform searching** for uncertain knowledge using the format: "<|begin_of_query|> keyword1\tkeyword2\t... <|end_of_query|>".
**The query must consist of straightforward and essential keywords separated by "\t"**. Furthermore, **the query must involve only a single triple to address a sub-question**.
Then, the search system will provide the Assistant with relevant information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>".

User:{question}
Assistant: <think>"""
    base_prompt_v2="""The User asks a question, and the Assistant solves it.
The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer.
The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
During the thinking process, **the Assistant can perform searching** for uncertain knowledge if necessary with the format of "<|begin_of_query|> search query (only list keywords separated by "\t" instead of the complete sentence , such as **"keyword_1 \t keyword_2 \t..."**)<|end_of_query|>". **A query must involve only a single triple**.
Then, the search system will provide the Assistant with the retrieval information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>".

User:{question}
Assistant: <think>"""

    base_prompt_v3 = """The User asks a **Judgment question**, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer. The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here (yes or no) </answer>". During the thinking process, the Assistant can perform searching for uncertain knowledge if necessary with the format of "<|begin_of_query|> search query (only keywords) here <|end_of_query|>". Then, the system will provide the Assistant with helpful information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>". The final answer **must be yes or no**.\n\nUser:{question}\nAssistant: <think>"""

    if type == "v0":
        question = examples["question"]
        prompt = base_prompt_v0.format(question=question)
        examples["chat_prompt"] = prompt
    elif type=="v1":
        question = examples["question"]
        prompt = base_prompt_v1.format(question=question)
        examples["chat_prompt"] = prompt
    elif type=="v2":
        question = examples["question"]
        prompt = base_prompt_v2.format(question=question)
        examples["chat_prompt"] = prompt
    elif type=="v3":
        question = examples["question"]
        prompt = base_prompt_v3.format(question=question)
        examples["chat_prompt"] = prompt
    else:
        raise ValueError("Invalid prompt type. Please choose from v0, v1, v2, or v3.")
    return examples

if __name__ == "__main__":
    args = parse_args()
    gpu_id = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    data_dir = args.data_dir
    split = "test"
    method_name = "r1searcher"
    serve_model_name = "R1-Searcher"
    temp=args.temp
    prompt_type=args.prompt_type
    model_path=args.model_path
    sources = args.sources
    remote_llm_url = args.remote_llm_url
    remote_web_retriever_url = args.remote_web_retriever_url

    remote_retriever_urls = {
        "musique": "http://127.0.0.1:18007",
        "omnieval": "http://127.0.0.1:18009",
        "bioasq": "http://127.0.0.1:18010",
        "nq": "http://127.0.0.1:18011",
        "hotpotqa": "http://127.0.0.1:18012",
        "pubmedqa": "http://127.0.0.1:18013",
    }

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # dataset_names = ["musique", "omnieval", "bioasq", "nq", "hotpotqa", "pubmedqa"]
    dataset_names = ["bioasq"]
    for dataset_name in dataset_names:
        remote_retriever_url = remote_retriever_urls[dataset_name]
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

        loguru.logger.info(f"Loaded {len(data)} samples from {data_path}")

        stop_tokens = ["<|im_end|>", "<|endoftext|>", "<|end_of_query|>", "</answer>"]
        sampling_params = SamplingParams(temperature=temp, top_p=0.95, max_tokens=512, stop=stop_tokens)

        llm_client = openai.OpenAI(
            api_key="EMPTY",
            base_url=remote_llm_url,
        )

        finished_all_list=[]

        continued_answer = copy.deepcopy(data)
        answers = ["" for _ in range(len(data))]
        rollout_sequences = ["" for _ in range(len(data))]

        def rollout_thread(idx):
            question = data[idx]['question']
            # loguru.logger.info(f"Question: {question}")

            original_messages = [
                {"role": "user", "content": process_text(data[idx], tokenizer, type=prompt_type)["chat_prompt"]},
            ]
            original_input = tokenizer.apply_chat_template(original_messages, tokenize=False, continue_final_message=True)
            model_input = original_input

            for k in range(10):
                response = sample_response(llm_client, serve_model_name, model_input, stop_tokens, 0.0)
                if isinstance(response, str) and "Error: maximum context length exceeded" in response:
                    loguru.logger.info(f"Round {k} exceeded max context length, stopping rollout for idx {idx}")
                    break
                completion = response.choices[0].text
                stop_reason = response.choices[0].stop_reason
                finish_reason = response.choices[0].finish_reason

                next_input = model_input + completion
                if k == 9: #检索次数太多了，直接停掉，就是未完成
                    break

                if "<answer>" in completion and stop_reason=="</answer>":
                    next_input = next_input + "</answer>"
                    break

                elif "<|begin_of_query|>" in completion and stop_reason=="<|end_of_query|>": #这里处理retrieve

                    query = completion.split("<|begin_of_query|>")[-1]
                    # loguru.logger.info(f"completion: {completion}, query: {query}")
                    query = query.replace('"',"").replace("'","").replace("\t"," ").replace("...","").strip()
                    if query:
                        topk = 5
                        if sources == "original":
                            batch_search_results = batch_search(remote_retriever_url, query, topk)
                            merged_results = batch_search_results
                        elif sources == "all":
                            batch_search_results = batch_search(remote_retriever_url, query, topk)
                            batch_web_search_results = batch_web_search("http://10.10.15.46:15005", query, "default", topk)
                            graph_search_results = graph_search(remote_retriever_url, query, topk)
                            merged_results = [f"Local Search Results: {i}\n\n{j}\n\nWeb Search Results: {k}"
                          for i, j, k in zip(batch_search_results, graph_search_results, batch_web_search_results)]
                        else:
                            raise ValueError("Invalid sources option. Please choose 'original' or 'all'.")
                        search_results_str = f"{merged_results[0]}"
                        next_input = next_input + "<|end_of_query|>\n\n"+ "<|begin_of_documents|>\n" + search_results_str + "<|end_of_documents|>\n\n"

                    else:
                        break
                else:
                    break
                model_input = next_input
            answer = extract_answer(next_input)
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
