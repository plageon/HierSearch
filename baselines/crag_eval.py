import argparse
from tqdm import tqdm

import torch
import json
import os
import loguru
from vllm import LLM, SamplingParams
from transformers import T5Tokenizer, T5ForSequenceClassification
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
sys.path.append('./')

from baselines.online_eval import batch_search, batch_web_search, graph_search, sample_response
from agentic_rag.eval_rollout_sequence import evaluate_rollout_sequence

import time

# logger = logging.getLogger(__name__)


def get_evaluator_data(file):
    with_label = False
    # with_label = True
    content = []
    label = []
    with open(file, "r", encoding="utf-8") as f:
        if with_label:
            for line in f.readlines()[:]:
                c, l = line.split("\t")
                content.append(c)
                label.append((int(l.strip()) - 0.5) * 2)
            return content, label
        else:
            for line in f.readlines():
                content.append(line.strip())
            return content, None


def process_flag(scores, n_docs, threshold1, threshold2):
    flags = []
    for score in scores:
        if score >= threshold1:
            flags.append('2')
        elif score >= threshold2:
            flags.append('1')
        else:
            flags.append('0')

    tmp_flag = []
    identification_flag = []
    for i, f in enumerate(flags):
        tmp_flag.append(f)
        if i % n_docs == n_docs - 1:
            if '2' in tmp_flag:
                identification_flag.append(2)
            elif '1' in tmp_flag:
                identification_flag.append(1)
            else:
                identification_flag.append(0)
            tmp_flag = []
    return identification_flag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluator_path', default="models/crag-t5", type=str)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--download_dir', type=str, help="specify vllm model download dir", default=".cache")
    parser.add_argument("--ndocs", type=int, default=-1,
                        help="Number of documents to retrieve per questions")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of documents to retrieve per questions")
    parser.add_argument("--upper_threshold", type=float, default=0.592, help="Number of documents to retrieve per questions")
    parser.add_argument("--lower_threshold", type=float, default=0.995, help="Number of documents to retrieve per questions")
    parser.add_argument("--method_name", type=str, default="crag")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--save_note", type=str, default='your-save-note-for-identification')
    parser.add_argument("--remote_llm_url", type=str, default="")
    parser.add_argument("--web_retriever_url", type=str, default="http://127.0.0.1:15005")
    args = parser.parse_args()
    args.lower_threshold = -args.lower_threshold

    method_name = args.method_name
    data_dir = args.data_dir
    split = args.split
    save_note = args.save_note if args.save_note else args.method_name

    tokenizer = T5Tokenizer.from_pretrained(args.evaluator_path)
    model = T5ForSequenceClassification.from_pretrained(args.evaluator_path, num_labels=1)
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    remote_retriever_urls = {
        "musique": "http://127.0.0.1:18007",
        "omnieval": "http://127.0.0.1:18009",
        "bioasq": "http://127.0.0.1:18010",
        "nq": "http://127.0.0.1:18011",
        "hotpotqa": "http://127.0.0.1:18012",
        "pubmedqa": "http://127.0.0.1:18013",
    }

    dataset_names = ["musique", "omnieval", "bioasq", "nq", "hotpotqa", "pubmedqa"]
    # dataset_names = ["pubmedqa"]
    for dataset_name in dataset_names:
        if dataset_name == "omnieval":
            os.environ["EVAL_LANG"] = "zh"
        else:
            os.environ["EVAL_LANG"] = "en"
        loguru.logger.info(f"Setting EVAL_LANG to {os.environ['EVAL_LANG']}")
        remote_retriever_url = remote_retriever_urls[dataset_name]
        loguru.logger.info(f"Using remote retriever url: {remote_retriever_url}")

        internal_result_path = f"data/{dataset_name}/rollout/reasoning_local_test.jsonl"
        external_result_path = f"data/{dataset_name}/rollout/reasoning_web_test.jsonl"
        combined_result_path = f"data/{dataset_name}/rollout/reasoning_all_test.jsonl"

        save_path = f'{data_dir}/{dataset_name}/rollout/{method_name}_{split}.jsonl'

        internal_result_lines = [json.loads(line) for line in open(internal_result_path)]
        external_result_lines = [json.loads(line) for line in open(external_result_path)]
        combined_result_lines = [json.loads(line) for line in open(combined_result_path)]

        questions = [l["question"] for l in internal_result_lines]
        identification_flag = []
        for i in tqdm(range(len(questions))):
            question = questions[i]
            results = []
            scores = []
            results.extend(graph_search(remote_retriever_url, question, 5)[0].split("\n\n"))
            results.extend(batch_search(remote_retriever_url, question, 5)[0].split("\n\n"))
            all_inputs = [f"{question} [SEP] {result}" for result in results]
            # loguru.logger.info(f"Question: {question}")
            # loguru.logger.info("\n".join(results))

            for inputs in all_inputs:
                test = tokenizer(inputs, return_tensors="pt", padding="max_length", max_length=512)
                with torch.no_grad():
                    outputs = model(test["input_ids"].to(device), attention_mask=test["attention_mask"].to(device))
                scores.append(float(outputs["logits"].cpu()))
            # loguru.logger.info(f"Scores: {scores}")
            flags = []
            for score in scores:
                if score >= args.upper_threshold:
                    flags.append('2')
                elif score >= args.lower_threshold:
                    flags.append('1')
                else:
                    flags.append('0')

            if '2' in flags:
                identification_flag.append(2)
            elif '1' in flags:
                identification_flag.append(1)
            else:
                identification_flag.append(0)

            # loguru.logger.info(f"flags: {flags}, identification_flag: {identification_flag[-1]}")

        answers, rollout_sequences, retrieval_source = [], [], []
        for flag, i, e, c in zip(identification_flag, internal_result_lines, external_result_lines, combined_result_lines):
            if flag == 0:
                answers.append(e["answer"])
                rollout_sequences.append(e["rollout_sequence"])
                retrieval_source.append("web")
            elif flag == 1:
                answers.append(c["answer"])
                rollout_sequences.append(c["rollout_sequence"])
                retrieval_source.append("local")
            elif flag == 2:
                answers.append(i["answer"])
                rollout_sequences.append(i["rollout_sequence"])
                retrieval_source.append("both")

        # eval rollout sequence
        gold_answers = [item['golden_answers'] for item in internal_result_lines]
        overall_qa_results = evaluate_rollout_sequence(gold_answers, answers)
        loguru.logger.info(f"gold_answers: {gold_answers}, \nanswers: {answers}")
        loguru.logger.info(f"Overall QA results: {overall_qa_results}")

        loguru.logger.info(f"Save results to {save_path}")
        # save results
        with open(save_path, 'w') as f:
            for i, item in enumerate(internal_result_lines):
                item['answer'] = answers[i]
                item['rollout_sequence'] = rollout_sequences[i]
                item['retrieval_source'] = retrieval_source[i]
                f.write(json.dumps(item, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()