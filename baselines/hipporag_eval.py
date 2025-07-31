from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
import json
import logging
import loguru
import numpy as np
import requests
from tqdm import tqdm
import vllm


from hipporag import HippoRAG
from hipporag.evaluation.qa_eval import QAExactMatch, QAF1Score
from hipporag.utils.config_utils import BaseConfig
from hipporag.utils.misc_utils import string_to_bool, QuerySolution, min_max_normalize, compute_mdhash_id
import argparse
from typing import List, Tuple, Union, Dict
from openai import OpenAI
import sys

sys.path.append("../HippoRAG/src/")
sys.path.append("./")
from agentic_rag.eval_rollout_sequence import evaluate_rollout_sequence
logger = logging.getLogger(__name__)


def get_gold_answers(samples):
    gold_answers = []
    for sample_idx in range(len(samples)):
        gold_ans = None
        sample = samples[sample_idx]

        if 'answer' in sample or 'gold_ans' in sample:
            gold_ans = sample['answer'] if 'answer' in sample else sample['gold_ans']
        elif 'reference' in sample:
            gold_ans = sample['reference']
        elif 'obj' in sample:
            gold_ans = set(
                [sample['obj']] + [sample['possible_answers']] + [sample['o_wiki_title']] + [sample['o_aliases']])
            gold_ans = list(gold_ans)
        elif 'golden_answers' in sample:
            gold_ans = sample['golden_answers']
        assert gold_ans is not None
        if isinstance(gold_ans, str):
            gold_ans = [gold_ans]
        assert isinstance(gold_ans, list)
        gold_ans = set(gold_ans)
        if 'answer_aliases' in sample:
            gold_ans.update(sample['answer_aliases'])

        gold_answers.append(gold_ans)

    return gold_answers


class WebHippoRAG(HippoRAG):
    def __init__(self, global_config: BaseConfig, **kwargs):
        super().__init__(global_config=global_config, **kwargs)
        openai_api_key = os.getenv("OPENAI_API_KEY", "sk-")
        # vllm local infer
        # self.chat_llm = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="sk-")
        self.chat_llm = OpenAI(base_url=global_config.llm_base_url, api_key=openai_api_key)
        self.web_retrieval_url = global_config.web_retrieval_url

    def batch_web_search(self, func_url, query: Union[str, List[str]], sample_id: Union[str, List[str]]=None, top_n=5) -> List[str]:
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

    def qa(self, queries: List[QuerySolution]) -> Tuple[List[QuerySolution], List[str], List[Dict]]:
        all_qa_messages = []

        for query_solution in tqdm(queries, desc="Collecting QA prompts"):

            # obtain the retrieved docs
            retrieved_passages = query_solution.docs[:self.global_config.qa_top_k]

            prompt_user = ''
            for passage in retrieved_passages:
                prompt_user += f'Wikipedia Title: {passage}\n\n'

            # provide web search results if available
            if self.global_config.web_retrieval:
                if "用户提问：" in query_solution.question:
                    # if the question is in a dialog format, we need to extract the actual question
                    query = query_solution.question.split("用户提问：")[-1].strip()
                else:
                    query = query_solution.question.strip()
                web_search_res = self.batch_web_search(self.web_retrieval_url,[query], ["default"], 5)
                # loguru.logger.info(f"web_search_res: {web_search_res}")
                prompt_user += f'Web Search Results: {web_search_res}\n\n'
            prompt_user += 'Question: ' + query_solution.question + '\nThought: '

            if self.prompt_template_manager.is_template_name_valid(name=f'rag_qa_{self.global_config.dataset}'):
                # find the corresponding prompt for this dataset
                prompt_dataset_name = self.global_config.dataset
            else:
                # the dataset does not have a customized prompt template yet
                logger.debug(
                    f"rag_qa_{self.global_config.dataset} does not have a customized prompt template. Using MUSIQUE's prompt template instead.")
                prompt_dataset_name = 'musique'
            all_qa_messages.append(
                self.prompt_template_manager.render(name=f'rag_qa_{prompt_dataset_name}', prompt_user=prompt_user))

        # all_qa_results = [self.llm_model.infer(qa_messages) for qa_messages in tqdm(all_qa_messages, desc="QA Reading")]
        all_qa_results = []
        # use vllm local infer
        loguru.logger.info(f"all_qa_messages[0]: {all_qa_messages[0]}")
        for qa_messages in tqdm(all_qa_messages, desc="QA Reading"):
            # loguru.logger.info(qa_messages)
            while True:
                try:
                    # loguru.logger.info(f"qa_messages: {qa_messages}")
                    # completion = self.chat_llm.chat.completions.create(model=self.global_config.chat_llm_name, messages=qa_messages, temperature=0.3, max_tokens=1024)
                    completion = self.chat_llm.chat.completions.create(
                        model=self.global_config.chat_llm_name,
                        messages=qa_messages,
                        temperature=0.3,
                        max_tokens=1024,
                    )
                    break
                except Exception as e:
                    loguru.logger.error(f"VLLM Runtime Error: {e}. Retrying...")
            all_qa_results.append(completion)
        loguru.logger.info(f"all_qa_results[0]: {all_qa_results[0]}")

        all_message, all_metadata = [], []
        for i, qa_result in enumerate(all_qa_results):
            response_message = qa_result.choices[0].message.content
            # append input messages
            qa_messages = all_qa_messages[i]
            qa_messages.append({'role': 'assistant', 'content': response_message})
            all_message.append(qa_messages)

        # Process responses and extract predicted answers.
        queries_solutions = []
        for query_solution_idx, query_solution in tqdm(enumerate(queries), desc="Extraction Answers from LLM Response"):
            response_content = all_message[query_solution_idx][-1]['content']
            try:
                pred_ans = response_content.split('Answer:')[1].strip()
            except Exception as e:
                logger.warning(f"Error in parsing the answer from the raw LLM QA inference response: {str(e)}!")
                pred_ans = response_content

            query_solution.answer = pred_ans
            queries_solutions.append(query_solution)

        return queries_solutions, all_message, all_metadata

    def rag_qa(self,
               queries: List[str | QuerySolution],
               gold_docs: List[List[str]] = None,
               gold_answers: List[List[str]] = None) -> Tuple[List[QuerySolution], List]:
        if gold_answers is not None:
            qa_em_evaluator = QAExactMatch(global_config=self.global_config)
            qa_f1_evaluator = QAF1Score(global_config=self.global_config)

        # loguru.logger.info(self.prompt_template_manager.render(name=f'rag_qa_pubmedqa', prompt_user="hello world"))
        # exit(0)

        # Retrieving (if necessary)
        overall_retrieval_result = None

        if not isinstance(queries[0], QuerySolution):
            if gold_docs is not None:
                queries, overall_retrieval_result = self.retrieve(queries=queries, gold_docs=gold_docs)
            else:
                queries = self.retrieve(queries=queries)

        # Performing QA

        queries_solutions, all_message, all_metadata = self.qa(queries)

        # Evaluating QA
        # if gold_answers is not None:
        #     overall_qa_em_result, example_qa_em_results = qa_em_evaluator.calculate_metric_scores(
        #         gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
        #         aggregation_fn=np.max)
        #     overall_qa_f1_result, example_qa_f1_results = qa_f1_evaluator.calculate_metric_scores(
        #         gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
        #         aggregation_fn=np.max)
        #
        #     # round off to 4 decimal places for QA results
        #     overall_qa_em_result.update(overall_qa_f1_result)
        #     overall_qa_results = overall_qa_em_result
        #     overall_qa_results = {k: round(float(v) * 100, 2) for k, v in overall_qa_results.items()}
        #     logger.info(f"Evaluation results for QA: {overall_qa_results}")
        #
        #     # Save retrieval and QA results
        #     for idx, q in enumerate(queries_solutions):
        #         q.gold_answers = list(gold_answers[idx])
        #         if gold_docs is not None:
        #             q.gold_docs = gold_docs[idx]
        #
        #     return queries_solutions, all_message, all_metadata, overall_retrieval_result, overall_qa_results
        # else:
        return queries_solutions, all_message


def main():
    parser = argparse.ArgumentParser(description="HippoRAG retrieval and QA")
    parser.add_argument('--llm_base_url', type=str, default='https://api.openai.com/v1', help='LLM base URL')
    parser.add_argument('--llm_name', type=str, default='gpt-4o-mini', help='LLM name')
    parser.add_argument("--chat_llm_name", type=str, default="Qwen3-8B", help="Chat LLM name")
    parser.add_argument('--embedding_name', type=str, default='nvidia/NV-Embed-v2', help='embedding model name')
    parser.add_argument('--force_index_from_scratch', type=str, default='false',
                        help='If set to True, will ignore all existing storage files and graph data and will rebuild from scratch.')
    parser.add_argument('--force_openie_from_scratch', type=str, default='false',
                        help='If set to False, will try to first reuse openie results for the corpus if they exist.')
    parser.add_argument('--openie_mode', choices=['online', 'offline'], default='online',
                        help="OpenIE mode, offline denotes using VLLM offline batch mode for indexing, while online denotes")
    parser.add_argument("--test_set", type=str, default="test", help="Test set name")
    parser.add_argument("--web_retrieval", action="store_true", help="Whether to use web retrieval")
    parser.add_argument("--web_retrieval_url", type=str, help="Web retrieval URL", )
    args = parser.parse_args()

    llm_base_url = args.llm_base_url
    llm_name = args.llm_name
    test_set = args.test_set


    force_index_from_scratch = string_to_bool(args.force_index_from_scratch)
    force_openie_from_scratch = string_to_bool(args.force_openie_from_scratch)

    # dataset_names = ["musique", "omnieval", "bioasq", "nq", "hotpotqa", "pubmedqa"]
    dataset_names = ["pubmedqa"]
    for dataset_name in dataset_names:
        # Prepare datasets and evaluation
        with open(f"data/{dataset_name}/{test_set}.jsonl", "r") as f:
            samples = [json.loads(line) for line in f]
        loguru.logger.info(f"Loaded {len(samples)} samples from {dataset_name} {test_set} set.")
        # samples = samples[:4]

        corpus_path = f"data/{dataset_name}/{dataset_name}_corpus.json"
        loguru.logger.info(corpus_path)
        with open(corpus_path, "r") as f:
            corpus = json.load(f)

        docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]

        all_queries = [s['question'] for s in samples]

        gold_answers = get_gold_answers(samples)
        gold_docs = None
        if dataset_name == "omnieval":
            dataset_language = "ZH"
            os.environ["OPENIE_LANG"] = "zh"
            os.environ["EVAL_LANG"] = "zh"
        else:
            dataset_language = "EN"
            os.environ["OPENIE_LANG"] = "en"
            os.environ["EVAL_LANG"] = "en"
        loguru.logger.info(f"dataset_language: {dataset_language}")

        if dataset_language == "EN":
            rerank_dspy_file_path = "./HippoRAG/src/hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json"
        else:
            rerank_dspy_file_path = "./HippoRAG/src/hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct_zh.json"
        graph_save_dir = f"data/{dataset_name}/graph"

        config = BaseConfig(
            save_dir=graph_save_dir,
            llm_base_url=llm_base_url,
            llm_name=llm_name,
            chat_llm_name=args.chat_llm_name,
            dataset=dataset_name,
            embedding_model_name=args.embedding_name,
            force_index_from_scratch=force_index_from_scratch,
            save_openie=False,
            # ignore previously stored index, set it to False if you want to use the previously stored index and embeddings
            force_openie_from_scratch=force_openie_from_scratch,
            rerank_dspy_file_path=rerank_dspy_file_path,
            retrieval_top_k=200,
            linking_top_k=5,
            web_retrieval=args.web_retrieval,
            web_retrieval_url=args.web_retrieval_url,
            max_qa_steps=3,
            qa_top_k=5,
            graph_type="facts_and_sim_passage_node_unidirectional",
            embedding_batch_size=8,
            max_new_tokens=None,
            corpus_len=len(corpus),
            openie_mode=args.openie_mode,
            language=dataset_language
        )

        logging.basicConfig(level=logging.INFO)

        web_hipporag = WebHippoRAG(global_config=config)

        web_hipporag.index(docs)
        queries_solutions, all_message = web_hipporag.rag_qa(queries=all_queries, gold_docs=gold_docs, gold_answers=gold_answers)
        # save reg_qa_res
        answers = [qs.answer for qs in queries_solutions]
        gold_answers = [item['golden_answers'] for item in samples]

        overall_qa_results = evaluate_rollout_sequence(gold_answers, answers)
        loguru.logger.info(f"gold_answers: {gold_answers}, \nanswers: {answers}")
        loguru.logger.info(f"Overall QA results: {overall_qa_results}")

        # save solutions
        result_dir = f"data/{dataset_name}/rollout"
        # 检查目录是否存在
        if not os.path.exists(result_dir):
            # 创建目录，exist_ok=False时若目录已存在会抛出错误
            os.makedirs(result_dir, exist_ok=True)
            loguru.logger.info(f"已成功创建目录: {result_dir}")
        else:
            loguru.logger.info(f"目录已存在: {result_dir}")

        if args.web_retrieval:
            result_file = f"{result_dir}/{args.chat_llm_name}_hipporag_all_{test_set}.jsonl"
        else:
            result_file = f"{result_dir}/{args.chat_llm_name}_hipporag_original_{test_set}.jsonl"
        with open(result_file, "w") as f:
            for i, sample in enumerate(samples):
                sample["answer"] = answers[i]
                sample['all_message'] = all_message[i]
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
