# run_web_thinker.py
import os
import json
import time
import re

import requests
from tqdm import tqdm
import numpy as np
import torch
import string
from typing import Optional, Tuple, List, Dict, Set, Union
import argparse
import random
import asyncio
import aiohttp
import loguru

from openai import AsyncOpenAI
import sys
sys.path.append("./")
from WebThinker.bing_search import (
    bing_web_search,
    extract_relevant_info,
    fetch_page_content,
    fetch_page_content_async,
    extract_snippet_with_context,
    bing_web_search_async,
    google_serper_search_async,
    extract_relevant_info_serper
)
from WebThinker.evaluate import (
    run_evaluation,
    extract_answer_fn
)
from WebThinker.prompts import (
    get_deep_web_explorer_instruction,
    get_web_page_reader_instruction,
    get_search_intent_instruction,
    get_click_intent_instruction,
    get_multiqa_search_o1_instruction,
    get_task_instruction_openqa,
    get_webpage_to_reasonchain_instruction,
)
from transformers import AutoTokenizer

from agentic_rag.eval_rollout_sequence import evaluate_rollout_sequence
from baselines.online_eval import batch_search, graph_search, batch_web_search

# tokenizer = AutoTokenizer.from_pretrained("/share/project/llm/QwQ-32B")
# # tokenizer = AutoTokenizer.from_pretrained("/share/project/llm/DeepSeek-R1-Distill-Qwen-32B")


# Define special tokens
BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"

BEGIN_CLICK_LINK = "<|begin_click_link|>"
END_CLICK_LINK = "<|end_click_link|>"
# BEGIN_CLICK_INTENT = "<|begin_click_intent|>"
# END_CLICK_INTENT = "<|end_click_intent|>"
BEGIN_CLICK_RESULT = "<|begin_click_result|>"
END_CLICK_RESULT = "<|end_click_result|>"

error_indicators = [
    'limit exceeded',
    'Error fetching',
    'Account balance not enough',
    'Invalid bearer token',
    'HTTP error occurred',
    'Error: Connection error occurred',
    'Error: Request timed out',
    'Unexpected error',
    'Please turn on Javascript',
    'Enable JavaScript',
    'port=443',
    'Please enable cookies',
]

invalid_search_queries = [
    "and end with",
    "search query",
    "query",
    "your query here",
    "your query",
    "your search query",
]



def parse_args():
    parser = argparse.ArgumentParser(description="Run Search-o1 for various datasets and models.")
    parser.add_argument('--single_question', type=str, default=None,
                        help="Single question to process instead of dataset")
    parser.add_argument('--dataset_name', type=str, required=False, default='custom',
                        help="Name of the dataset to use.")
    parser.add_argument('--split', type=str, required=False, default='test', help="Dataset split to use.")
    parser.add_argument('--subset_num', type=int, default=-1,
                        help="Number of examples to process. Defaults to all if not specified.")

    parser.add_argument('--temperature', type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument('--top_p', type=float, default=0.8, help="Top-p sampling parameter.")
    parser.add_argument('--min_p', type=float, default=0.05, help="Minimum p sampling parameter.")
    parser.add_argument('--top_k_sampling', type=int, default=20, help="Top-k sampling parameter.")
    parser.add_argument('--repetition_penalty', type=float, default=1.05,
                        help="Repetition penalty. If not set, defaults based on the model.")
    parser.add_argument('--max_tokens', type=int, default=81920,
                        help="Maximum number of tokens to generate. If not set, defaults based on the model and dataset.")

    parser.add_argument('--max_search_limit', type=int, default=20, help="Maximum number of searches per question.")
    parser.add_argument('--top_k', type=int, default=10, help="Maximum number of search documents to return.")
    parser.add_argument('--keep_links', action='store_true', default=False,
                        help="Whether to keep links in fetched web content")
    parser.add_argument('--use_jina', action='store_true', help="Whether to use Jina API for document fetching.")
    parser.add_argument('--jina_api_key', type=str, default='None', help="Your Jina API Key to Fetch URL Content.")
    parser.add_argument('--bing_subscription_key', type=str, default=None, help="Bing Search API subscription key.")
    parser.add_argument('--bing_endpoint', type=str, default="https://api.bing.microsoft.com/v7.0/search",
                        help="Bing Search API endpoint.")
    parser.add_argument('--serper_api_key', type=str, default=None, help="Google Serper API key.")
    parser.add_argument('--search_engine', type=str, default="bing", choices=["bing", "serper"],
                        help="Search engine to use (bing or serper). Default: bing")
    parser.add_argument('--eval', action='store_true', help="Whether to run evaluation after generation.")
    parser.add_argument('--seed', type=int, default=None,
                        help="Random seed for generation. If not set, will use current timestamp as seed.")
    parser.add_argument('--api_base_url', type=str, required=True, help="Base URL for the API endpoint")
    parser.add_argument('--api_key', type=str, default="empty", help="API key for the main model")
    parser.add_argument('--concurrent_limit', type=int, default=32, help="Maximum number of concurrent API calls")
    parser.add_argument('--model_name', type=str, default="QwQ-32B", help="Name of the model to use")
    parser.add_argument('--lora_name', type=str, default=None, help="Name of the LoRA adapter to load")
    parser.add_argument('--lora_path', type=str, default=None, help="Path to the LoRA weights")
    parser.add_argument('--tokenizer_path', type=str, default="/share/project/llm/QwQ-32B",
                        help="Path to the main tokenizer")
    parser.add_argument("--max_turn", type=int, default=10, help="Maximum number of turns in the conversation.")
    parser.add_argument("--max_doc_len", type=int, default=3000, help="Maximum length of documents.")
    parser.add_argument('--sources', type=str, default="original", help="Dataset to use for evaluation (default: custom)")
    parser.add_argument('--web_retrieval_url', type=str, help='URL of the web retrieval API')
    return parser.parse_args()


# Initialize tokenizers
args = parse_args()
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)


def wait_until_available(url):
    while True:
        available_retrievers = requests.get(f"{url}/health").json()["retrievers"]["available"]
        if available_retrievers > 0:
            # loguru.logger.info(f"Available retrievers: {available_retrievers}")
            return
        else:
            time.sleep(random.randint(5, 30))

def web_search_with_retry(web_retrieval_url, search_query, top_n=5):
    PATIENCE = 3
    patience = PATIENCE
    while patience > 0:
        data = {"query": search_query, "sample_id": "default", "top_n": top_n}
        try:
            wait_until_available(web_retrieval_url)
            relevant_info = requests.post(f"{web_retrieval_url}/web_search_base", json=data, timeout=120).json()
            return relevant_info
        except requests.exceptions.RequestException as e:
            loguru.logger.info(f"Error: {e}, Retrying...")
            patience -= 1
            time.sleep(1)
    loguru.logger.info(f"Failed to get relevant info after {PATIENCE} retries. Request data: {data}")
    return []


def extract_between(text, start_marker, end_marker):
    """Extracts text between two markers in a string."""
    try:
        pattern = re.escape(end_marker[::-1]) + r"(.*?)" + re.escape(start_marker[::-1])
        # Run pattern matching with timeout
        matches = re.findall(pattern, text[::-1], flags=re.DOTALL)
        if matches:
            return matches[0][::-1].strip()
        return None
    except Exception as e:
        loguru.logger.info(f"---Error:---\n{str(e)}")
        loguru.logger.info(f"-------------------")
        return None


def format_search_results(relevant_info: List[Dict]) -> str:
    """Format search results into a readable string"""
    formatted_documents = ""
    for i, doc_info in enumerate(relevant_info):
        doc_info['title'] = doc_info['title'].replace('<b>', '').replace('</b>', '')
        doc_info['snippet'] = doc_info['snippet'].replace('<b>', '').replace('</b>', '')
        formatted_documents += f"***Web Page {i + 1}:***\n"
        formatted_documents += json.dumps(doc_info, ensure_ascii=False, indent=2) + "\n"
        # formatted_documents += f"Title: {doc_info['title']}\n"
        # formatted_documents += f"URL: {doc_info['url']}\n"
        # formatted_documents += f"Snippet: {doc_info['snippet']}\n\n"
        # if 'page_info' in doc_info:
        #     formatted_documents += f"Web Page Information: {doc_info['page_info']}\n\n\n\n"
    return formatted_documents


async def generate_response(
    client: AsyncOpenAI,
    prompt: str,
    semaphore: asyncio.Semaphore,
    temperature: float,
    top_p: float,
    max_tokens: int,
    repetition_penalty: float,
    top_k: int,
    min_p: float,
    model_name: str,
    retry_limit: int = 3,
) -> str:
    """Generate a single response with retry logic"""
    for attempt in range(retry_limit):
        try:
            async with semaphore:
                messages = [{"role": "user", "content": prompt}]
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=min(max_tokens, 32768),  # Reserve 1000 tokens for prompt
                    stop=[END_SEARCH_QUERY],
                    extra_body={
                        'top_k': top_k,
                        'include_stop_str_in_output': True,
                        'repetition_penalty': repetition_penalty,
                        # 'min_p': min_p
                    },
                    timeout=1500,
                )
                # print('---\n', response.choices[0].message.content)
                return response.choices[0].message.content
        except Exception as e:
            print(f"Generate Response Error occurred: {e}, Starting retry attempt {attempt + 1}")
            if attempt == retry_limit - 1:
                print(f"Failed after {retry_limit} attempts: {e}")
                return ""
            await asyncio.sleep(1 * (attempt + 1))
    return ""


async def generate_webpage_to_reasonchain(
        client: AsyncOpenAI,
        original_question: str,
        prev_reasoning: str,
        search_query: str,
        document: str,
        dataset_name: str,
        batch_output_records: List[Dict],
        max_tokens: int = 32768,
        temperature: float = 0.7,
        top_p: float = 0.8,
        repetition_penalty: float = 1.05,
        top_k: int = 20,
        min_p: float = 0.05,
        model_name: str = "QwQ-32B",
        semaphore: asyncio.Semaphore = None,
) -> str:
    user_prompt = get_webpage_to_reasonchain_instruction(prev_reasoning, search_query, document)

    raw_output = await generate_response(
        client=client,
        prompt=user_prompt,
        semaphore=semaphore,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
        top_k=top_k,
        min_p=min_p,
        model_name=model_name,
    )

    extracted_info = extract_answer_fn(raw_output, mode='infogen')

    batch_output_records.append({
        'prompt': user_prompt,
        'raw_output': raw_output,
        'extracted_info': extracted_info
    })

    return extracted_info

async def process_single_sequence(
        seq: Dict,
        client: AsyncOpenAI,
        semaphore: asyncio.Semaphore,
        args: argparse.Namespace,
        search_cache: Dict,
        url_cache: Dict,
        batch_output_records: List[Dict],
        turn: int = 0,
        remote_retriever_url: str = None,
) -> Dict:
    """Process a single sequence through its entire reasoning chain"""
    seq["local_search_results"] = []
    seq["web_search_results"] = []

    while not seq['finished'] and turn < args.max_turn:
        # Generate next step in reasoning
        text = await generate_response(
            client=client,
            prompt=seq['prompt'],
            semaphore=semaphore,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            repetition_penalty=args.repetition_penalty,
            top_k=args.top_k_sampling,
            min_p=args.min_p,
            model_name=args.model_name,
        )

        seq['history'].append(text)
        seq['prompt'] += text
        seq['output'] += text


        # Extract search query
        search_query = extract_between(text, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)

        if search_query and seq['output'].rstrip().endswith(END_SEARCH_QUERY):
            # Remove the </think> tag from the prompt and output
            seq['prompt'] = seq['prompt'].replace('</think>\n', '')
            seq['output'] = seq['output'].replace('</think>\n', '')
            if seq['search_count'] < args.max_search_limit and search_query not in seq['executed_search_queries']:
                # Execute search
                # results = {}
                # if search_query in search_cache:
                #     results = search_cache[search_query]
                # else:
                #     try:
                #         if args.search_engine == "bing":
                #             results = bing_web_search(search_query, args.bing_subscription_key, args.bing_endpoint)
                #         search_cache[search_query] = results
                #     except Exception as e:
                #         print(f"Error during search query '{search_query}' using {args.search_engine}: {e}")
                #         search_cache[search_query] = {}
                #         results = {}
                #
                # if args.search_engine == "bing":
                #     relevant_info = extract_relevant_info(results)[:args.top_k]
                # elif args.search_engine == "serper":
                #     relevant_info = extract_relevant_info_serper(results)[:args.top_k]
                # else:  # Should not happen
                #     relevant_info = []
                results = await asyncio.to_thread(web_search_with_retry, args.web_retrieval_url, search_query,
                                                  top_n=args.top_k)
                relevant_info = results
                seq['relevant_info'] = relevant_info

                # Process documents
                formatted_documents = ""
                urls_to_fetch = []
                for doc_info in relevant_info:
                    url = doc_info['url']
                    if url not in url_cache:
                        urls_to_fetch.append(url)

                if urls_to_fetch:
                    try:
                        contents = fetch_page_content(urls_to_fetch, use_jina=args.use_jina,
                                                      jina_api_key=args.jina_api_key)
                        for url, content in contents.items():
                            url_cache[url] = content
                    except Exception as e:
                        print(f"Error fetching URLs: {e}")
                        for url in urls_to_fetch:
                            url_cache[url] = ""

                for i, doc_info in enumerate(relevant_info):
                    url = doc_info['url']
                    raw_context = url_cache[url]
                    doc_info['snippet'] = doc_info['snippet'].replace('<b>', '').replace('</b>', '')
                    success, filtered_context = extract_snippet_with_context(raw_context, doc_info['snippet'],
                                                                             context_chars=args.max_doc_len)
                    context = filtered_context if success else raw_context[:args.max_doc_len * 2]

                    doc_info['context'] = context
                    formatted_documents += f"**Web Page {i + 1}:**\n"
                    formatted_documents += json.dumps(doc_info, ensure_ascii=False, indent=2) + "\n"
                seq["web_search_results"].append(formatted_documents)
                # print(seq["web_search_results"])
                if args.sources == "all":
                    chunk_search_info = batch_search(remote_retriever_url, search_query, top_n=5)
                    graph_search_info = batch_search(remote_retriever_url, search_query, top_n=5)
                    local_search_info = f"\nLocal Search Results:\n{chunk_search_info}\nGraph Search Results:\n{graph_search_info}"

                    seq["local_search_results"].append(local_search_info)
                    # print(seq["local_search_results"])
                    formatted_documents += local_search_info

                # Process reasoning steps
                all_reasoning_steps = seq['output'].replace('\n\n', '\n').split("\n")
                truncated_prev_reasoning = ""
                for i, step in enumerate(all_reasoning_steps):
                    truncated_prev_reasoning += f"Step {i + 1}: {step}\n\n"

                prev_steps = truncated_prev_reasoning.split('\n\n')
                if len(prev_steps) > 5:
                    truncated_prev_reasoning = ''
                    for i, step in enumerate(prev_steps):
                        if i == 0 or i >= len(
                                prev_steps) - 4 or BEGIN_SEARCH_QUERY in step or BEGIN_SEARCH_RESULT in step:
                            truncated_prev_reasoning += step + '\n\n'
                        else:
                            if truncated_prev_reasoning[-len('\n\n...\n\n'):] != '\n\n...\n\n':
                                truncated_prev_reasoning += '...\n\n'
                truncated_prev_reasoning = truncated_prev_reasoning.strip('\n')

                # Generate webpage analysis
                analysis = await generate_webpage_to_reasonchain(
                    client=client,
                    original_question=seq['item']['question'],
                    prev_reasoning=truncated_prev_reasoning,
                    search_query=search_query,
                    document=formatted_documents,
                    dataset_name=args.dataset_name,
                    batch_output_records=batch_output_records,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    top_k=args.top_k_sampling,
                    min_p=args.min_p,
                    model_name=args.model_name,
                    semaphore=semaphore,
                )

                # Update sequence with analysis
                append_text = f"\n\n{BEGIN_SEARCH_RESULT}{analysis}{END_SEARCH_RESULT}\n\n"

                seq['prompt'] += append_text
                seq['output'] += append_text
                seq['history'].append(append_text)

                seq['search_count'] += 1
                seq['executed_search_queries'].add(search_query)

            elif seq['search_count'] >= args.max_search_limit:
                limit_message = f"\n{BEGIN_SEARCH_RESULT}\nThe maximum search limit is exceeded. You are not allowed to search.\n{END_SEARCH_RESULT}\n"
                seq['prompt'] += limit_message
                seq['output'] += limit_message
                seq['history'].append(limit_message)

            elif search_query in seq['executed_search_queries']:
                limit_message = f"\n{BEGIN_SEARCH_RESULT}\nYou have searched this query. Please refer to previous results.\n{END_SEARCH_RESULT}\n"
                seq['prompt'] += limit_message
                seq['output'] += limit_message
                seq['history'].append(limit_message)

        else:
            seq['finished'] = True

        turn += 1
    return seq


async def load_lora_adapter(api_base_url: str, lora_name: str, lora_path: str) -> bool:
    """Load a LoRA adapter with the specified name and path"""
    try:
        lora_load_url = f"{api_base_url}/load_lora_adapter"
        lora_payload = {
            "lora_name": lora_name,
            "lora_path": lora_path
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(lora_load_url, json=lora_payload) as response:
                return response.status == 200
    except Exception as e:
        loguru.logger.info(f"Error loading LoRA adapter: {e}")
        return False


async def unload_lora_adapter(api_base_url: str, lora_name: str) -> bool:
    """Unload a LoRA adapter with the specified name"""
    try:
        unload_url = f"{api_base_url}/unload_lora_adapter"
        unload_payload = {"lora_name": lora_name}
        async with aiohttp.ClientSession() as session:
            async with session.post(unload_url, json=unload_payload) as response:
                return response.status == 200
    except Exception as e:
        loguru.logger.info(f"Error unloading LoRA adapter: {e}")
        return False


async def main_async():
    # Set random seed
    if args.seed is None:
        args.seed = int(time.time())
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.jina_api_key == 'None':
        jina_api_key = None

    # Initialize the OpenAI client
    client = AsyncOpenAI(
        api_key=args.api_key,
        base_url=args.api_base_url,
    )

    remote_retriever_urls = {
        "musique": "http://127.0.0.1:18007",
        "omnieval": "http://127.0.0.1:18009",
        "bioasq": "http://127.0.0.1:18010",
        "nq": "http://127.0.0.1:18011",
        "hotpotqa": "http://127.0.0.1:18012",
        "pubmedqa": "http://127.0.0.1:18013",
    }

    split = args.split
    # Load and prepare data
    dataset_names = ["musique", "omnieval", "bioasq", "nq", "hotpotqa", "pubmedqa"]
    # dataset_names = ["omnieval", "bioasq", "hotpotqa"]
    for dataset_name in dataset_names:
        loguru.logger.info(f"Running prefrag on {dataset_name} dataset, split: {split}")
        if dataset_name == "omnieval":
            os.environ["EVAL_LANG"] = "zh"
        else:
            os.environ["EVAL_LANG"] = "en"

        loguru.logger.info(f"Setting EVAL_LANG to {os.environ['EVAL_LANG']}")
        remote_retriever_url = remote_retriever_urls[dataset_name]
        loguru.logger.info(f"Using remote retriever URL: {remote_retriever_url}")

        with open(f"data/{dataset_name}/{split}.jsonl", encoding="utf-8") as f:
            filtered_data = [json.loads(line) for line in f]
        # filtered_data = filtered_data[:4]

        data_dir = f"data/{dataset_name}"
        result_dir = f"{data_dir}/rollout"

        method_name = "searcho1"
        os.makedirs(result_dir, exist_ok=True)
        save_path = f'{result_dir}/{method_name}_{args.sources}_{split}.jsonl'

        # Prepare sequences
        active_sequences = []
        for item in filtered_data:
            question = item['question']
            instruction = get_multiqa_search_o1_instruction(args.max_search_limit)
            user_prompt = get_task_instruction_openqa(question)

            prompt = instruction + user_prompt
            item['prompt'] = prompt
            active_sequences.append({
                'item': item,
                'prompt': prompt,
                'output': '',
                'finished': False,
                'history': [],
                'search_count': 0,
                'executed_search_queries': set(),
            })

        # Initialize batch output records
        batch_output_records = []
        start_time = time.time()

        # Create semaphore for concurrent API calls
        semaphore = asyncio.Semaphore(args.concurrent_limit)

        # Load LoRA adapter if specified
        if args.lora_name and args.lora_path:
            loguru.logger.info(f"Loading LoRA adapter '{args.lora_name}' from {args.lora_path}")
            success = await load_lora_adapter(args.api_base_url, args.lora_name, args.lora_path)
            if not success:
                loguru.logger.info("Failed to load LoRA adapter")
                return
            else:
                loguru.logger.info("LoRA adapter loaded successfully")

        try:
            # Process all sequences concurrently
            tasks = [
                process_single_sequence(
                    seq=seq,
                    client=client,
                    semaphore=semaphore,
                    args=args,
                    search_cache={},
                    url_cache={},
                    batch_output_records=batch_output_records,
                    remote_retriever_url=remote_retriever_url
                )
                for seq in active_sequences
            ]

            # Run all sequences concurrently with progress bar
            with tqdm(total=len(tasks)) as pbar:
                async def track_progress(task):
                    result = await task
                    pbar.update(1)
                    return result

                tracked_tasks = [track_progress(task) for task in tasks]
                rollout_sequences = await asyncio.gather(*tracked_tasks)
        finally:
            # Unload LoRA adapter if it was loaded
            if args.lora_name:
                loguru.logger.info(f"Unloading LoRA adapter '{args.lora_name}'")
                await unload_lora_adapter(args.api_base_url, args.lora_name)
                loguru.logger.info("LoRA adapter unloaded successfully")

        total_time = time.time() - start_time
        loguru.logger.info(f"Total time taken: {total_time:.2f} seconds")

        # Save caches
        # save_caches()
        answers = [extract_answer_fn(seq['output'], mode='qa', extract_answer=True) for seq in rollout_sequences]
        gold_answers = [item['golden_answers'] for item in filtered_data]
        local_search_results = [seq['local_search_results'] for seq in rollout_sequences]
        web_search_results = [seq['web_search_results'] for seq in rollout_sequences]
        overall_qa_results = evaluate_rollout_sequence(gold_answers, answers)
        loguru.logger.info(f"gold_answers: {gold_answers}, \nanswers: {answers}")
        loguru.logger.info(f"Overall QA results: {overall_qa_results}")

        loguru.logger.info(f"Saving results to {save_path}")
        # save results
        with open(save_path, 'w') as f:
            for i, item in enumerate(filtered_data):
                item['answer'] = answers[i]
                item['rollout_sequence'] = rollout_sequences[i]["output"]
                item['local_search_results'] = local_search_results[i]
                item['web_search_results'] = web_search_results[i]
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        loguru.logger.info("Process completed.")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()