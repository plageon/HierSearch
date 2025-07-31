import argparse
from typing import Union, List, Tuple, Dict

import openai
import loguru
import time
import random
import json
import re
import os


from tqdm import tqdm
from transformers import AutoTokenizer
import concurrent.futures
import sys
from time import gmtime, strftime
sys.path.append('./')
from agentic_rag.eval_rollout_sequence import evaluate_rollout_sequence
from baselines.online_eval import sample_response, batch_web_search, webpage_browse, batch_search, graph_search

SYS_PROMPT = f"""## Background information 
* Today is {strftime("%Y-%m-%d", gmtime())}
* You are Deep AI Research Assistant

The question I give you is a complex question that requires a *deep research* to answer.

I will provide you with two tools to help you answer the question:
* A web search tool to help you perform google search. 
* A webpage browsing tool to help you get new page content.

You don't have to answer the question now, but you should first think about the research plan or what to search next.

Your output format should be one of the following two formats:

<think>
YOUR THINKING PROCESS
</think>
<answer>
YOUR ANSWER AFTER GETTING ENOUGH INFORMATION
</answer>

or

<think>
YOUR THINKING PROCESS
</think>
<tool_call>
YOUR TOOL CALL WITH CORRECT FORMAT
</tool_call>

You should always follow the above two formats strictly.
Only output the final answer (in words, numbers or phrase) inside the <answer></answer> tag, without any explanations or extra information. If this is a yes-or-no question, you should only answer yes or no.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for relevant information from google. You should use this tool if the historical page content is not enough to answer the question. Or last search result is not relevant to the question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "The query to search, which helps answer the question"
                        },
                        "description": "The queries to search"
                    }
                },
                "required": ["query"],
                "minItems": 1,
                "uniqueItems": True
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browse_webpage",
            "description": "Browse the webpage and return the content that not appeared in the conversation history. You should use this tool if the last action is search and the search result maybe relevant to the question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url_list": {"type": "array",
                            "items": {
                                "type": "string",
                                "description": "The chosen url from the search result, do not use url that not appeared in the search result"
                            },
                            "description": "The chosen urls from the search result."
                        },
                },
                "required": ["url_list"]
            }
        }
    }
]


def parse_response(completion, stop_reason, finish_reason, sources):

    def extract_search_content(start_tag, end_tag, text: str) -> Dict:
        try:
            tool_call = text.split(start_tag)[1].split(end_tag)[0]
            tool_call = json.loads(tool_call)
            assert "name" in tool_call, "no valid function name in tool_call"
            assert "arguments" in tool_call, "no valid arguments in tool_call"
            assert tool_call["name"] in ["web_search", "browse_webpage"], "invalid tool name"
            if tool_call["name"] == "web_search":
                assert "query" in tool_call["arguments"], "no valid query in tool_call"
                assert isinstance(tool_call["arguments"]["query"], list), "query should be a list"
            elif tool_call["name"] == "browse_webpage":
                assert "url_list" in tool_call["arguments"], "no valid url_list in tool_call"
                assert isinstance(tool_call["arguments"]["url_list"], list), "url_list should be a list"
                assert len(tool_call["arguments"]["url_list"]) >= 1, "url_list number must be greater than 0"
            return tool_call
        except Exception as e:
            print(f"model tool call format error: {e}")
            return None
    if "<answer>" in completion:
        return completion, "", ""

    if finish_reason == 'stop' and isinstance(stop_reason, str) and end_pattern in stop_reason:
        if start_pattern in completion:
            completion += end_pattern
            think = completion.split(start_pattern)[0]
            search_content = extract_search_content(start_pattern, end_pattern, completion)
            if search_content is not None:
                if search_content["name"] == "web_search":
                    if sources == "original":
                        search_result = batch_web_search("http://10.10.15.46:15005/", search_content["arguments"]["query"], top_n=10)
                        return think, search_content, f" <result>\n{search_result[0]}\n</result>"
                    elif sources == "all":
                        web_search_result = batch_web_search("http://10.10.15.46:15005/", search_content["arguments"]["query"], top_n=10)
                        batch_search_results = batch_search(remote_retriever_url, search_content["arguments"]["query"], top_n=5)
                        graph_search_results = graph_search(remote_retriever_url, search_content["arguments"]["query"], top_n=5)
                        return think, search_content, f" <result>\nLocal Search Results: {batch_search_results[0]}\n{graph_search_results[0]}\nWeb Search Results: {web_search_result[0]}\n</result>"
                elif search_content["name"] == "browse_webpage":
                    url_list = search_content["arguments"]["url_list"]
                    query = [think.replace("<think>", "").replace("</think>", "") for _ in url_list]
                    search_result = webpage_browse(remote_web_retriever_url, query, url_list)
                    search_result = "\n".join(search_result)
                    return think, search_content, f" <result>\n{search_result}\n</result>"
    return completion, "", ""


def extract_answer(solution_str: str):
    text = solution_str.strip()

    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return ""

    answer_part = match.group(1)
    return answer_part


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running exp")
    parser.add_argument("--method_name", type=str, default="deepresearcher")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--dataset_name", type=str, default="musique")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--save_note", type=str, default='your-save-note-for-identification')
    parser.add_argument("--remote_llm_url", type=str, default="http://127.0.0.1:13003/v1")
    parser.add_argument("--remote_web_retriever_url", type=str, default="http://10.10.15.46:15005")
    parser.add_argument("--model_path", type=str, default="models/DeepResearcher-7b/")
    parser.add_argument("--serve_model_name", type=str, default="DeepResearcher-7b")
    parser.add_argument("--sys_template_name", type=str, default="default")
    parser.add_argument("--max_turns", type=int, default=10, help="maximum number of turns for the conversation")
    parser.add_argument("--sources", type=str, default="all", help="sources to use for search, can be 'original' or 'all'. If 'all', it will use batch_search, batch_web_search and graph_search.")

    args = parser.parse_args()
    method_name = args.method_name
    data_dir = args.data_dir
    dataset_name = args.dataset_name
    split = args.split
    save_note = args.save_note if args.save_note else args.method_name
    remote_llm_url = args.remote_llm_url
    remote_web_retriever_url = args.remote_web_retriever_url
    model_path = args.model_path
    serve_model_name = args.serve_model_name
    sys_template_name = args.sys_template_name
    sources = args.sources

    remote_retriever_urls = {
        "musique": "http://127.0.0.1:18007",
        "omnieval": "http://127.0.0.1:18009",
        "bioasq": "http://127.0.0.1:18010",
        "nq": "http://127.0.0.1:18011",
        "hotpotqa": "http://127.0.0.1:18012",
        "pubmedqa": "http://127.0.0.1:18013",
    }

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm_client = openai.OpenAI(
        api_key="EMPTY",
        base_url=remote_llm_url,
    )

    # test sample response
    # _response = sample_response("Hello, ", ["\n"])
    # loguru.logger.info(_response)

    start_pattern = '<tool_call>'
    end_pattern = '</tool_call>'
    stop_words = [end_pattern]

    # dataset_names = ["musique", "omnieval", "bioasq", "nq", "hotpotqa", "pubmedqa"]
    dataset_names = ["musique", "nq"]
    for dataset_name in dataset_names:
        loguru.logger.info(f"Running {method_name} on {dataset_name} dataset, split: {split}, sources: {sources}")
        if dataset_name == "omnieval":
            os.environ["EVAL_LANG"] = "zh"
        else:
            os.environ["EVAL_LANG"] = "en"
        loguru.logger.info(f"Setting EVAL_LANG to {os.environ['EVAL_LANG']}")
        remote_retriever_url = remote_retriever_urls[dataset_name]
        loguru.logger.info(f"Using remote retriever URL: {remote_retriever_url}")
        # read data
        data_path = f'{data_dir}/{dataset_name}/{split}.jsonl'
        loguru.logger.info(f"Loading data from {data_path}")
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
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": question},
            ]
            # loguru.logger.info(f"Original messages: {original_messages}")

            # online eval
            round_cnt = 0
            curr_messages = original_messages.copy()
            while round_cnt < args.max_turns:
                round_cnt += 1
                # loguru.logger.info(f"Round {round_cnt}")
                model_input = tokenizer.apply_chat_template(curr_messages, tools=TOOLS, tokenize=False, add_generation_prompt=True)
                response = sample_response(llm_client, serve_model_name, model_input, stop_words)
                if isinstance(response, str) and "Error: maximum context length exceeded" in response:
                    loguru.logger.info(f"Round {round_cnt} exceeded max context length, stopping rollout for idx {idx}")
                    break
                # loguru.logger.info(f"Round {round_cnt}, response: {response}")
                completion = response.choices[0].text
                stop_reason = response.choices[0].stop_reason
                finish_reason = response.choices[0].finish_reason
                # loguru.logger.info(f"Round {round_cnt}, completion: {completion}, stop_reason: {stop_reason}")
                think, search_content, search_res = parse_response(completion, stop_reason, finish_reason, sources)
                if finish_reason == 'stop' and isinstance(stop_reason, str) and end_pattern in stop_reason:
                    completion += end_pattern
                curr_messages.append({"role": "assistant", "content": completion})
                # loguru.logger.info(f"Round {round_cnt}, think: {think}, search_content: {search_content}, search_result: {search_res}")
                if search_res == "":
                    break
                else:
                    curr_messages.append({"role": "tool", "content": search_res})
                    # loguru.logger.info(f"Round {round_cnt}, curr_messages: {curr_messages}")
            final_input = tokenizer.apply_chat_template(curr_messages, tokenize=False, add_generation_prompt=False)
            answer = extract_answer(curr_messages[-1]['content'])
            # loguru.logger.info(f"Final input: {final_input} answer: {answer}")
            answers[idx] = answer
            rollout_sequences[idx] = final_input
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

        # save results
        loguru.logger.info(f"Saving results to {save_path}")
        with open(save_path, 'w') as f:
            for i, item in enumerate(data):
                item['answer'] = answers[i]
                item['rollout_sequence'] = rollout_sequences[i]
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
