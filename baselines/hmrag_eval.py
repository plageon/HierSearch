import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable
import sys
import argparse

from transformers import AutoTokenizer
import openai
import json
import loguru
import concurrent.futures
from tqdm import tqdm
import time
import random
from collections import Counter
import re

sys.path.append('./')
from agentic_rag.eval_rollout_sequence import evaluate_rollout_sequence, normalize_answer
from baselines.online_eval import batch_search, graph_search, batch_web_search

def sample_chat_response(llm_client, serve_model_name, messages, enable_thinking: bool = True):
    _response = None
    while True:
        try:
            _response = llm_client.chat.completions.create(
                messages=messages,
                model=serve_model_name,
                temperature=0.3,
                max_tokens=2048,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": enable_thinking,}
                }
            )
            break
        except Exception as e:
            loguru.logger.warning(f"Error in response: {e}")
            time.sleep(random.randint(1, 5))
            continue
    return _response


def extract_answer(solution_str: str) -> str:
    text = solution_str.strip()

    pattern = r"<answer>(.*?)</answer>"
    # remove FINAL ANSWER tags if they exist
    text = text.replace("FINAL ANSWER", "")
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return ""

    answer_part = match.group(1)
    return answer_part

class HMRAG:
    def __init__(self, config):
        self.client = openai.OpenAI(
            api_key="sk-",
            base_url=remote_llm_url,
        )

    def count_intents(self, query: str) -> int:
        """
        Determine the number of intents in the input query.
        Use LLM to analyze the number of intents contained in the input text.
        Args:
            query (str): The input query text.
        Returns:
            int: The number of intents.
        """
        # Clearly specify the prompt format
        prompt = "Please calculate how many independent intents are contained in the following query. Return only an integer:\n{query}\nNumber of intents: "
        max_attempts = 3
        for attempt in range(max_attempts):
            messages = [
                {"role": "user", "content": prompt.format(query=query)}
            ]
            response = sample_chat_response(self.client, serve_model_name, messages)
            response = response.choices[0].message.content
            loguru.logger.info(response)
            try:
                return int(response.strip())
            except ValueError:
                if attempt == max_attempts - 1:
                    return 1  # If parsing fails after multiple attempts, default to 1 intent
        return 1

    def decompose(self, query: str) -> List[str]:
        """
        Decompose the query. If the number of intents is greater than 1, perform intent decomposition.
        Args:
            query (str): The input query text.
        Returns:
            List[str]: A list of decomposed sub-queries.
        """
        intent_count = self.count_intents(query)
        intent_count = min(intent_count, 3)  # Limit the number of intents to a maximum of 5
        if intent_count > 1:
            return self.split_query(query)
        # return [query]
        return query

    def split_query(self, query: str) -> List[str]:
        """
        The method that actually performs query decomposition.
        Args:
            query (str): The input query text.
        Returns:
            List[str]: A list of decomposed sub-queries.
        """
        prompt = "Split the following query into multiple independent sub-queries, separated by '||', without additional explanations:\n{query}\nList of sub-queries: "
        messages = [
            {"role": "user", "content": prompt.format(query=query)}
        ]
        response = sample_chat_response(self.client, serve_model_name, messages, enable_thinking=False)
        response = response.choices[0].message.content
        # loguru.logger.info(response, reasoning)
        return [q.strip() for q in response.split("||") if q.strip()]

    def single_source_rag(self, question, query, source: str):
        if dataset_name == "pubmedqa":
            chat_template = ("You are a helpful assistant that can solve the given question step by step with the help of the search tool results. "
                "Please answer the following question. You should provide a \"yes\" or \"no\" answer wrapped in <answer> and </answer> tags. "
                "Example: <answer> yes </answer>. Do not add any additional explanations.\n"
                "Search Tool Results: {results}\n\nQuestion: {question}")
        else:
            chat_template = (
                "You are a helpful assistant that can solve the given question step by step with the help of the search tool results. "
                "Please answer the following question. You should provide the final answer wrapped in <answer> and </answer> tags. "
                "Example: <answer> FINAL ANSWER </answer>. Do not add any additional explanations.\n"
                "Search Tool Results: {results}\n\nQuestion: {question}")

        if len(query) == 0:
            query = [question]  # if query is empty, use the original question
        # for q in query:
        #     if len(q.split()) > 10:
        #         loguru.logger.warning(f"Query is too long: {q}. It may cause issues with the retriever.")
        #         q = " ".join(q.split()[:10])  # truncate the query to the first 10 words
        if source == "chunk":
            results = batch_search(remote_retriever_url, query, 5)
        elif source == "graph":
            results = graph_search(remote_retriever_url, query, 5)
        elif source == "web":
            results = batch_web_search(remote_web_retriever_url, query, ["default" for _ in query], 5)
        else:
            raise ValueError(f"Invalid source: {source}")
        # merge results
        results = "\n\n".join(results)
        messages = [
            {"role": "user", "content": chat_template.format(results=results, question=question)}
        ]
        response = sample_chat_response(self.client, serve_model_name, messages, enable_thinking=True)
        completion = response.choices[0].message.content
        # extract resoning content betwee <think> and </think>
        reasoning = completion
        answer = extract_answer(completion)

        # loguru.logger.info(f"Query: {query}, Source: {source}, Answer: {answer}")
        return answer, reasoning, results

    def get_most_common_answer(self, res):
        """
        Get the most common answer from the list of answers
        """
        # remove empty answers
        res = [r for r in res if r.strip() != ""]
        # loguru.logger.info(res)
        if len(res) == 0:
            return []
        counter = Counter(res)

        # 获取最高频率
        max_count = max(counter.values())

        # 收集所有频率等于 max_count 的值
        most_common_values = [item for item, count in counter.items() if count == max_count]
        return most_common_values

    def refine(self, output_text, output_graph, output_web, query):
        if dataset_name == "pubmedqa":
            # refine the output text if most common answer is not found
            chat_template = ("You are a helpful assistant that can solve the given question basing on the answers from the text chunk agent, graph agent and web agent. "
                "Please answer the following question. You should provide a \"yes\" or \"no\" answer wrapped in <answer> and </answer> tags. "
                "Example: <answer> yes </answer>. Do not add any additional explanations.\n"
                "Text Chunk Agent Answer: {output_text}\nGraph Agent Answer: {output_graph}\nWeb Agent Answer: {output_web}\n\nQuestion: {question}")
        else:
            # refine the output text if most common answer is not found
            chat_template = (
                "You are a helpful assistant that can solve the given the question basing on the answers from the text chunk agent, graph agent and web agent. "
                "Please answer the following question. You should provide the final answer wrapped in <answer> and </answer> tags. "
                "Example: <answer> FINAL ANSWER </answer>. Do not add any additional explanations.\n"
                "Text Chunk Agent Answer: {output_text}\nGraph Agent Answer: {output_graph}\nWeb Agent Answer: {output_web}\n\nQuestion: {question}")
        messages = [
            {"role": "user", "content": chat_template.format(
                output_text=output_text, output_graph=output_graph, output_web=output_web, question=query)}
        ]
        response = sample_chat_response(self.client, serve_model_name, messages)
        completion = response.choices[0].message.content
        reasoning = completion
        output = extract_answer(completion)
        # loguru.logger.info(f"Refined output: {output}")

        return output, reasoning



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running exp")
    parser.add_argument("--method_name", type=str, default="hmrag")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--save_note", type=str, default='your-save-note-for-identification')
    parser.add_argument("--remote_llm_url", type=str, default="")
    parser.add_argument("--model_path", type=str, default="/model_load/Qwen25-7B-Instruct-1M")
    parser.add_argument("--remote_web_retriever_url", type=str, default="http://10.10.15.46:15005")
    parser.add_argument("--serve_model_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--sys_template_name", type=str, default="default",)

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

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm_client = openai.OpenAI(
        api_key="EMPTY",
        base_url=remote_llm_url,
    )

    # hmrag agent
    hmrag = HMRAG(llm_client)

    remote_retriever_urls = {
        "musique": "http://127.0.0.1:18007",
        "omnieval": "http://127.0.0.1:18009",
        "bioasq": "http://127.0.0.1:18010",
        "nq": "http://127.0.0.1:18011",
        "hotpotqa": "http://127.0.0.1:18012",
        "pubmedqa": "http://127.0.0.1:18013",
    }
    # dataset_names = ["musique", "omnieval", "bioasq", "nq", "hotpotqa", "pubmedqa"]
    dataset_names = ["musique", "omnieval", "bioasq", "nq", "hotpotqa"]
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

        gold_answers = [item['golden_answers'] for item in data]

        remote_retriever_url = remote_retriever_urls[dataset_name]

        answers = ["" for _ in range(len(data))]
        decomposed_queries = [[] for _ in range(len(data))]
        multi_source_answers = [["", "", ""] for _ in range(len(data))]
        reasoning_contents = [[] for _ in range(len(data))]  # Store reasoning contents for each sample
        sample_ids = [item['id'] for item in data]


        def rollout_thread(idx):
            question = data[idx]['question']
            sample_id = data[idx]['id']

            d_queries = hmrag.split_query(question)
            d_queries = [q.strip() for q in d_queries if q.strip()]  # remove empty queries
            d_queries = d_queries[:3]  # limit to 3 queries
            decomposed_queries[idx] = d_queries
            # reasoning_contents[idx].append(decompose_reasoning)
            # loguru.logger.info(decomposed_queries)

            all_sources = {"chunk": 0, "graph": 1, "web": 2}
            # Randomly shuffle the sources to avoid bias
            sources = list(all_sources.keys())
            random.shuffle(sources)
            for source in sources:
                source_answer, reasoning, retrieved_contents = hmrag.single_source_rag(question, d_queries, source)
                multi_source_answers[idx][all_sources[source]] = source_answer
                reasoning_contents[idx].append(reasoning)
                # loguru.logger.info(f"Query: {question}, Source: {source}, Answer: {source_answer}")
            answer = hmrag.get_most_common_answer(multi_source_answers[idx])
            if len(answer) > 1 or len(answer) == 0:
                # loguru.logger.warning(f"Multiple answers found for sample {sample_id}: {answer}. Refining...")
                answer, reasoning_text = hmrag.refine(
                    output_text=multi_source_answers[idx][0],
                    output_graph=multi_source_answers[idx][1],
                    output_web=multi_source_answers[idx][2],
                    query=question
                )
                reasoning_contents[idx].append(reasoning_text)
            else:
                answer = answer[0]

            answers[idx] = answer
            # loguru.logger.info(f"Sample ID: {sample_id}, Final Answer: {answers[idx]}")


        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
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
                item['reasoning_contents'] = reasoning_contents[i]
                item['decomposed_queries'] = decomposed_queries[i]
                item['multi_source_answers'] = multi_source_answers[i]
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

