import random
from functools import wraps

from jinja2 import Template
import re
import time
import json
import os
from tqdm import tqdm
import logging
import yaml
import concurrent.futures
from vllm import LLM, SamplingParams
import loguru
import argparse
import sys
import openai
sys.path.append("../HippoRAG/src/")
sys.path.append("./")
from agentic_rag.eval_rollout_sequence import evaluate_rollout_sequence
from baselines.online_eval import sample_response, batch_web_search, webpage_browse, batch_search, graph_search

gpt_client = openai.OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"],
)

def retry(max: int = 10, sleep: int = 1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(max):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i == max - 1:
                        loguru.logger.error(f"func {func.__name__} with args {args} and kwargs {kwargs} failed after {max} times, error: {e}")
                        return None
                    elif sleep:
                        time.sleep(sleep + random.randint(0, 5))

        return wrapper

    return decorator

@retry(max=5, sleep=1)
def batch_web_search_with_retry(*args, **kwargs):
    return batch_web_search(*args, **kwargs)


def api_gen(model,messages,temperature=0.1,top_p=0.9,stop=None):
    try:
        # If your API supports unified OpenAI protocol calls (including GLM models), comment this out
        # ------------------------------------------------------------
        completion = gpt_client.chat.completions.create(
            model=model,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            max_tokens=4096,
            messages=messages
        )
        response=completion.choices[0].message.content

        return response

    except Exception as e:
        print(f"An error occurred: {e}")
        time.sleep(0.5)

    return None

@retry(max=5, sleep=1)
def call_api(prompt, stop=None):
    messages = [{"role": "user", "content": prompt}]
    res = api_gen(args.model, messages, temperature=args.temperature, top_p=args.top_p, stop=stop)
    assert res is not None
    return res


def call_vllm(prompt, stop=None):
    if "llama" in args.model:
        model_template = f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    elif "glm" in args.model:
        model_template = f"<|user|>\n{prompt}<|assistant|>\n"
    sampling_params = SamplingParams(max_tokens=4096, temperature=args.temperature, top_p=args.top_p, stop=stop,
                                     include_stop_str_in_output=True)
    response = llm.generate(model_template, sampling_params)[0].outputs[0].text
    return response


def save_log_to_file(logger, log_file="my_log", log_folder="logs"):
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    current_date = time.strftime("%Y%m%d-%H%M%S")
    log_file_name = f"{log_file}-{current_date}.log"
    file_handler = logging.FileHandler(os.path.join(log_folder, log_file_name))
    logger.addHandler(file_handler)


def base_web(question):
    refs = Search_Engine(args.dataset, args.retrieve_top_k).call(question)

    prompt = refs + base_format.format(question=question)
    answer = call_llm(prompt)
    return answer

def base_wo_retri(question):
    prompt = f'Answer this question :\n{question}\nGive me only the answer without including any other words.\n\nAnswer:'
    # if args.dataset=="bioasq":
    #     prompt=f'ANSWER "yes" OR "no" ONLY (You have to choose the most likely option).\nAnswer:'
    answer = call_llm(prompt)
    return answer


class Search_Engine:
    def __init__(self, dataset, topk):
        self.dataset = dataset
        self.topk = topk

    @staticmethod
    def introduction():
        return """{"name":"Search_Engine", "description":"This is a knowledge base general search engine that can be used to query external knowledge, learn facts, etc.", "input":"The phrase or question to be searched."}"""

    def call(self, query):
        # loguru.logger.info("====Call in Search Engine====")
        query = str(query).strip('"')
        result = batch_web_search_with_retry("http://10.10.15.46:15005/", [query], top_n=5)
        # loguru.logger.info(f"query: {query}\n result: {result}")
        return result



class Pref:
    def __init__(self, max_step=3, tools=[], language="EN"):

        self.tools = tools
        self.max_step = max_step
        self.language = language

    def call(self, question):

        return self.gen_answer(question)

    def gen_answer(self, question):
        key = "prefrag" if self.language == "EN" else "zh_prefrag"
        prompt_template = Template(config["prompt"][key])
        answer_try_search = False
        output_process, observations_logs, evaluation_process = [], [], []
        i = 0
        while i <= self.max_step:
            # loguru.logger.info(f"output_process:{output_process}")
            prompt = prompt_template.render(
                answer_format=answer_format, max_step=self.max_step,
                question=question, tools=self.tools, thought="\n".join(output_process)
            ).strip()
            output = call_llm(prompt, stop=["Observation:", "Observation:\n"])
            # loguru.logger.info(f"output:{output}")
            answer = self.extract_final_answer(output)
            action, action_input = self.extract_action_info(output)
            # loguru.logger.info(f"answer:{answer}")
            # loguru.logger.info(f"action:{action}")
            if answer:
                loguru.logger.info(f"output: {output}")
                self_evaluation_match = self.extract_self_evaluation(output)
                if self_evaluation_match and (
                        'PARTIALLY CORRECT' in self_evaluation_match or 'INCORRECT' in self_evaluation_match):
                    if not answer_try_search:
                        question = question.replace("\"", "")
                        evaluation_process.append(output)
                        i -= 1
                        search_info_local = f"Local Search Results: {batch_search(remote_retriever_url, [question], top_n=5)[0]}\n\n{graph_search(remote_retriever_url, [question], top_n=5)[0]}"

                        search_info_web = f"Web Search Results: {batch_web_search_with_retry(web_retrieval_url, [question], top_n=5)[0]}"

                        search_info = f"{search_info_local}\n\n{search_info_web}"

                        observations_logs.append({"info": search_info, "type": "web"})
                        output_process.extend([output + "\nObservation:", f"<result> {search_info} </result>"])
                        answer_try_search = True
                        continue
                output_process.append(output)
                return (answer.strip(), '\n'.join(output_process))
            if action and action_input:
                tool = next((t for t in self.tools if t.__name__.lower() in action.lower()), None)
                if tool:
                    observation = self.prefer_retrieval(question, action_input, observations_logs)
                    # loguru.logger.info(observation)
                    # loguru.logger.info(f"observation:{observation}")
                    observations_logs.append({"info": observation})
                    if "Observation" not in output:
                        output += "Observation:"
                    output_process.extend([output, observation])
                    # loguru.logger.info(output_process)
            elif "Observation" not in output or "Action" not in output:
                break
            i += 1
        thoughts_str = '\n'.join(output_process)
        prompt = thoughts_str + base_format.format(question=question)
        output = call_llm(prompt)
        answer = self.extract_final_answer("Final Answer:" + output)
        if not answer:
            answer = output.strip(":").strip()
        thoughts_str += f"Final Answer:{answer}"
        return (answer, thoughts_str)

    def prefer_retrieval(self, question, new_q, obser_logs):
        question = question.replace("\"", "")
        if not obser_logs:
            observation = f"<result> Local Search Results: {batch_search(remote_retriever_url, [question], top_n=5)[0]}\n\n{graph_search(remote_retriever_url, [question], top_n=5)[0]} </result>"
            return observation
        observation = f" <result> Local Search Results: {batch_search(remote_retriever_url, [question], top_n=5)[0]}\n\n{graph_search(remote_retriever_url, [question], top_n=5)[0]} </result>"
        existed_info = "\n".join([d["info"] for d in obser_logs])
        key = "prefer_retrieval" if self.language == "EN" else "zh_prefer_retrieval"
        template = config["prompt"][key]
        prompt = template.format(question=question, existed_info=existed_info, observation=observation).strip()

        response = call_llm(prompt)
        try:
            if 'json' in response:
                response = response[response.index('{'):response.rindex('}') + 1]

            result = json.loads(response)
            res = result["status"]

            if res.lower() == "true":
                return observation
        except:
            match = re.search("True|true", response)
            if match:
                return observation
        new_q = new_q.replace("\"", "")
        observation = f"{batch_web_search_with_retry(web_retrieval_url, [new_q], top_n=5)}"
        # loguru.logger.info("Prefer web retrieval")
        return observation

    @staticmethod
    def extract_self_evaluation(output):
        match = re.search(r"Self-Evaluation\s*:\s*(.+?)(?:\s|$)Explanation", output,
                          re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if match:
            return match.group(1).strip()

        return output

    @staticmethod
    def extract_final_answer(output):
        matches = re.findall(r"Final Answer\s*:\s*(.*?)(?:\n\s*Self-Evaluation|Explanation:|Observation:|Thought:|$)",
                             output, re.IGNORECASE | re.DOTALL)

        if matches:
            return matches[-1].strip(":").strip()
        elif "Self-Evaluation:" in output:
            match = re.search(r"(.*?)(?:\n\s*Self-Evaluation|$)", output, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip(":").strip()
        return None

    @staticmethod
    def extract_action_info(output):
        action_match = re.search(r"Action\s*:\s*(.*)", output, re.IGNORECASE)
        action_input_match = re.search(
            r"Action Input\s*:\s*(\".*?\"|.*?)\s*(?=Observation:|Thought:|Final Answer:|Self-Evaluation:|$)", output,
            re.IGNORECASE | re.DOTALL)

        if action_match and action_input_match:
            action = action_match.group(1).strip()
            action_input = action_input_match.group(1).strip()
            if "none" not in [action.lower(), action_input.lower()]:
                return action, action_input
        return None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        choices=["gpt-4o-mini", "glm-4-plus", "llama3.1-8b-instruct", "glm4-9b-chat",
                                 "glm4-9b-chat-dpo", "llama-3.1-70b-instruct"], default='glm4',
                        help="Specify the model to use for inference")
    parser.add_argument('--MaxClients', type=int, default=1, help="Maximum number of concurrent clients")
    parser.add_argument('--retrieve_method', type=str, default="es",
                        help="Retrieval method to use: elasticsearch (es) or embedding-based (emb)")
    parser.add_argument('--retrieve_top_k', type=int, default=5,
                        help="Number of top documents to retrieve for each query")
    parser.add_argument('--max_step', type=int, default=3, help="Maximum number of reasoning steps")
    parser.add_argument('--method_name', type=str, default="prefrag",
                        choices=["prefrag", "base_local", "base_web", "base_local_web", "base_wo_retri"],
                        help="Method for question answering")
    parser.add_argument('--resume_path', type=str, default="", help="Path to checkpoint file to resume generation from")
    parser.add_argument('--temperature', type=float, default=0.1, help="Sampling temperature for generation")
    parser.add_argument('--top_p', type=float, default=0.9, help="Top-p sampling parameter for generation")
    parser.add_argument('--device', type=str, default='cuda:6', help="Device for model inference (e.g., cuda:0, cpu)")
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.45, help="Fraction of GPU memory to use")
    parser.add_argument('--language', type=str, default="EN", help="Dataset language")
    parser.add_argument('--web_retrieval_url', type=str, help="Web retrieval url")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--split", type=str, default="test")

    remote_retriever_urls = {
        "musique": "http://127.0.0.1:18007",
        "omnieval": "http://127.0.0.1:18009",
        "bioasq": "http://127.0.0.1:18010",
        "nq": "http://127.0.0.1:18011",
        "hotpotqa": "http://127.0.0.1:18012",
        "pubmedqa": "http://127.0.0.1:18013",
    }

    with open('./PrefRAG/config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    from datetime import datetime

    system_prompt = f'''Current date: {datetime.now().strftime('%Y-%m-%d')}'''

    args = parser.parse_args()
    method_name = args.method_name
    data_dir = args.data_dir
    web_retrieval_url = args.web_retrieval_url
    split = args.split


    key = "answer_format" if args.language == "EN" else "zh_answer_format"
    answer_format = config["prompt"][key]
    key = "base_answer_format" if args.language == "EN" else "zh_base_answer_format"
    base_format = config["prompt"][key]

    if "gpt" in args.model or args.model in ["glm-4-plus"]:
        call_llm = call_api
    else:
        llm = LLM(model=config["model"][args.model], tensor_parallel_size=1, trust_remote_code=True, dtype='bfloat16',
                  gpu_memory_utilization=args.gpu_memory_utilization)

        call_llm = call_vllm

    dataset_names = ["musique", "omnieval", "bioasq", "nq", "hotpotqa", "pubmedqa"]
    # dataset_names = ["nq"]
    for dataset_name in dataset_names:
        loguru.logger.info(f"Running prefrag on {dataset_name} dataset, split: {split}")
        if dataset_name == "omnieval":
            os.environ["EVAL_LANG"] = "zh"
            prefrag = Pref(tools=[Search_Engine], max_step=args.max_step, language="ZH")
        else:
            os.environ["EVAL_LANG"] = "en"
            prefrag = Pref(tools=[Search_Engine], max_step=args.max_step, language="EN")

        loguru.logger.info(f"Setting EVAL_LANG to {os.environ['EVAL_LANG']}")
        remote_retriever_url = remote_retriever_urls[dataset_name]
        loguru.logger.info(f"Using remote retriever URL: {remote_retriever_url}")

        with open(f"data/{dataset_name}/{split}.jsonl", encoding="utf-8") as f:
            qa_data = [json.loads(line) for line in f]

        result_dir = f"data/{dataset_name}/rollout"
        # 检查目录是否存在
        if not os.path.exists(result_dir):
            # 创建目录，exist_ok=False时若目录已存在会抛出错误
            os.makedirs(result_dir, exist_ok=True)
            loguru.logger.info(f"已成功创建目录: {result_dir}")
        else:
            loguru.logger.info(f"目录已存在: {result_dir}")

        all_result = []

        # read data
        data_path = f'{data_dir}/{dataset_name}/{split}.jsonl'
        loguru.logger.info(f"Loading data from {data_path}")
        save_path = f'{data_dir}/{dataset_name}/rollout/{method_name}_{split}.jsonl'
        with open(data_path, 'r') as f:
            data = [json.loads(line) for line in f]
        # data = data[:4]

        answers = ["" for _ in range(len(data))]
        rollout_sequences = ["" for _ in range(len(data))]
        def rollout_thread(idx):
            question = data[idx]['question']
            result = prefrag.call(question)
            answers[idx] = result[0]
            rollout_sequences[idx] = result[1]


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
