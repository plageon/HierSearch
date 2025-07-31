from __future__ import annotations

import time
from functools import partial

import logging

import torch
from fastapi import FastAPI, HTTPException
import argparse
from pydantic import BaseModel
from typing import List, Tuple, Union, Dict
import asyncio
from collections import deque
import sys
from transformers import AutoTokenizer

import openai
import loguru
from sentence_transformers import SentenceTransformer
sys.path.append("./")
from baselines.online_eval import batch_search, graph_search, get_adjacent_passages, batch_web_search, webpage_browse, sample_response
from agentic_rag.templates import prompt_template_dict
import concurrent.futures
from tqdm import tqdm

sys.path.append("./src/")
logger = logging.getLogger(__name__)
app = FastAPI()

agent_list = []
available_local_search_agents = deque()
available_web_search_agents = deque()
retriever_semaphore = None
from typing import List, Optional

import numpy as np

def min_max_normalize(x):
    # handle zero division
    if np.max(x) - np.min(x) == 0:
        return x
    return (x - np.min(x)) / (np.max(x) - np.min(x))


class SearchAgent:

    def __init__(
            self,
            search_domain: str,
            remote_llm_url,
            remote_retriever_url,
            remote_web_retriever_url,
            remote_web_browse_url,
            model_path,
            serve_model_name,
            supported_tools,
            max_turns,
            embedding_model_name,
            embedding_model_device,
    ):
        self.remote_llm_url = remote_llm_url
        loguru.logger.info(f"remote_retriever_url: {remote_retriever_url}")
        loguru.logger.info(f"remote_web_retriever_url: {remote_web_retriever_url}")
        if search_domain == "local":
            self.remote_web_browse_url = ""
            if remote_retriever_url.endswith('/'):
                remote_retriever_url = remote_retriever_url[:-1]
            self.remote_retriever_url = {
                "musique": f"{remote_retriever_url}:18007",
                "omnieval": f"{remote_retriever_url}:18009",
                "bioasq": f"{remote_retriever_url}:18010",
                "nq": f"{remote_retriever_url}:18011",
                "hotpotqa": f"{remote_retriever_url}:18012",
                "pubmedqa": f"{remote_retriever_url}:18013",
                "default": f"{remote_retriever_url}:18007",
            }
        else:
            self.remote_web_browse_url = remote_web_browse_url if remote_web_browse_url else remote_web_retriever_url
            if ":" in remote_web_retriever_url.split('//')[-1]:
                self.remote_retriever_url = {
                    "musique": f"{remote_web_retriever_url}",
                    "omnieval": f"{remote_web_retriever_url}",
                    "bioasq": f"{remote_web_retriever_url}",
                    "nq": f"{remote_web_retriever_url}",
                    "hotpotqa": f"{remote_web_retriever_url}",
                    "pubmedqa": f"{remote_web_retriever_url}",
                    "default": f"{remote_web_retriever_url}",
                }
            else:
                if remote_web_retriever_url.endswith('/'):
                    remote_web_retriever_url = remote_web_retriever_url[:-1]
                self.remote_retriever_url = {
                    "musique": f"{remote_web_retriever_url}:19007",
                    "omnieval": f"{remote_web_retriever_url}:19009",
                    "bioasq": f"{remote_web_retriever_url}:19010",
                    "nq": f"{remote_web_retriever_url}:19011",
                    "hotpotqa": f"{remote_web_retriever_url}:19012",
                    "pubmedqa": f"{remote_web_retriever_url}:19013",
                    "default": f"{remote_web_retriever_url}:19007",
                }
        self.model_path = model_path
        self.serve_model_name = serve_model_name
        self.max_turns = max_turns
        self.all_start_patterns = ['<chunk_search>', '<web_search>', '<graph_search>', '<browse_url>', '<web_search>']
        self.all_end_patterns = ['</chunk_search>', '</web_search>', '</graph_search>', '</browse_url>', '</web_search>']

        # supported tools
        self.start_patterns, self.end_patterns = [], []
        for start_pattern, end_pattern in zip(self.all_start_patterns, self.all_end_patterns):
            tool_name = start_pattern.split('<')[1].split('>')[0]
            if tool_name in supported_tools:
                self.start_patterns.append(start_pattern)
                self.end_patterns.append(end_pattern)
        self.stop_words = self.end_patterns

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.llm_client = openai.OpenAI(
            api_key="EMPTY",
            base_url=remote_llm_url,
        )
        if search_domain == "local":
            self.sys_template = {
                "omnieval": prompt_template_dict["zh_graph_search_agent_template_sys"],
                "default": prompt_template_dict["graph_search_agent_template_sys"],
            }
        else:
            self.sys_template = {
                "omnieval": prompt_template_dict["zh_web_only_search_agent_template_sys"],
                "default": prompt_template_dict["web_only_search_agent_template_sys"],
            }
        self.embedding_model = SentenceTransformer(
            embedding_model_name, trust_remote_code=True, device=embedding_model_device,
        )

    def parse_response(self, sample_id, data_source, completion, stop_reason, finish_reason):
        remote_retriever_url = self.remote_retriever_url.get(data_source, self.remote_retriever_url["default"])
        functions_dict = {
            '<chunk_search>': partial(batch_search, remote_retriever_url, top_n=10),
            '<web_search>': partial(batch_web_search, remote_retriever_url, sample_id=sample_id, top_n=10),
            '<graph_search>': partial(graph_search, remote_retriever_url, top_n=10),
            '<get_adjacent_passages>': partial(get_adjacent_passages, remote_retriever_url, top_n=10),
            '<browse_url>': partial(webpage_browse, self.remote_web_browse_url, sample_id=sample_id, top_n=10),
        }

        def extract_search_content(start_tag, end_tag, text: str) -> str:
            try:
                end_pos = text.rindex(end_tag)
                start_pos = text.rindex(start_tag, 0, end_pos)
                return text[start_pos + len(start_tag):end_pos].strip()
            except ValueError:
                return ""

        if finish_reason == 'stop' and isinstance(stop_reason, str) and any(
                [p in stop_reason for p in self.end_patterns]):
            ## process the search
            for start_pattern, end_pattern in zip(self.start_patterns, self.end_patterns):
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

    def parse_agent_rollout_sequence(self, sequence: str, filter_ratio_1: float=0.3) -> Union[Dict[str, list]]:
        """
        Parse the agent rollout sequence to extract the prompt and response.

        Args:
            sequence (str): The agent rollout sequence.

        Returns:
            Dict[str, str]: A dictionary containing the prompt and response.
        """
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

        evidences, hypotheses, conclusions = [], [], []
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
                hypotheses.append({
                    "contents": hypothesis,
                    "source": "Local Agent Hypothesis",
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
            num_retained_results = int(len(all_results) * filter_ratio_1)  # retain top 30% of results
            if num_retained_results == 0:
                num_retained_results = 1
            if not hypothesis:
                instant_scores = [0.0 for _ in all_results]  # if no hypothesis, set scores to 0
                # simply retain first num_retained_results results
                retained = [True if i < num_retained_results else False for i in range(len(all_results))]
            else:
                instant_scores = self.info_relevance(all_results, hypothesis)
                # retain num_retained_results results with the highest scores
                top_indices = np.argsort(instant_scores)[-num_retained_results:]
                retained = [True if i in top_indices else False for i in range(len(all_results))]
                # if instant_scores is 0 dimensional, convert it to 1 dimensional
                if isinstance(instant_scores, np.ndarray) and instant_scores.ndim == 0:
                    # loguru.logger.warning(f"Instant scores is 0 dimensional, converting to 1 dimensional: {instant_scores}, {all_results}")
                    instant_scores = np.array([instant_scores.item() for _ in all_results])
            for i, result in enumerate(all_results):
                evidences.append({
                    "contents": result,
                    "source": evidence_sources[start_pattern],
                    "instant_score": instant_scores[i],
                    "retained": retained[i] if 'retained' in locals() else True,
                    "start_pos": result_pos + len('<result>'),
                    "end_pos": result_end_pos
                })

        answer_start = text.find('<answer>')
        answer_end = text.find('</answer>')
        if answer_start != -1 and answer_end != -1:
            # extract conclusion
            conclusions.append({
                "contents": text[answer_start + len('<answer>'):answer_end].strip(),
                "source": "Local Agent Conclusion",
                "start_pos": answer_start + len('<answer>'),
                "end_pos": answer_end
            })
        return {
            "evidences": evidences,
            "hypotheses": hypotheses,
            "conclusions": conclusions
        }

    def agent_rollout(self, question, sample_id, data_source, return_rollout_sequence=False, filter_ratio_1=0.3):

        # loguru.logger.info(f"Question: {question}")
        if question == "不需要" or question.lower() == "Not necessary":
            loguru.logger.info(f"question: {question}, skip")
            return {
                "evidences": [],
                "hypotheses": [],
                "conclusions": [],
                "rollout_sequence": ""
            }
        sys_template = self.sys_template.get(data_source, self.sys_template["default"])
        original_messages = [
            {"role": "system", "content": sys_template},
            {"role": "user", "content": question},
        ]
        original_input = self.tokenizer.apply_chat_template(original_messages, tokenize=False, add_generation_prompt=True)
        # loguru.logger.info(f"Original input: {original_input}")

        # online eval
        round_cnt = 0
        model_input = original_input
        next_input = model_input
        while round_cnt < self.max_turns:
            round_cnt += 1
            # loguru.logger.info(f"Round {round_cnt}")
            response = sample_response(self.llm_client, self.serve_model_name, model_input, self.stop_words, 0.9)
            # loguru.logger.info(f"Round {round_cnt}, response: {response}")
            completion = response.choices[0].text
            stop_reason = response.choices[0].stop_reason
            finish_reason = response.choices[0].finish_reason
            # loguru.logger.info(f"Round {round_cnt}, completion: {completion}, stop_reason: {stop_reason}")
            # parse response
            completion, search_res = self.parse_response(sample_id, data_source, completion, stop_reason, finish_reason)
            # loguru.logger.info(f"Round {round_cnt}, search_res: {search_res}")
            next_input = model_input + completion
            # loguru.logger.info(f"Next input: {next_input}")
            if search_res == "":
                break
            else:
                next_input += search_res
                model_input = next_input
        try:
            agent_collected_info = self.parse_agent_rollout_sequence(next_input, filter_ratio_1=filter_ratio_1)
        except Exception as e:
            loguru.logger.error(f"Error in parsing agent rollout sequence: {e}")
            # loguru.logger.error(f"Next input: {next_input}")
            import traceback
            traceback.print_exc()
            agent_collected_info = {
                "evidences": [],
                "hypotheses": [],
                "conclusions": []
            }
        if return_rollout_sequence:
            agent_collected_info["rollout_sequence"] = next_input
        else:
            agent_collected_info["rollout_sequence"] = ""
        return agent_collected_info

    def batch_encode(self, texts: List[str], **kwargs) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(texts, str): texts = [texts]

        params = {
            "prompt": "",
            "normalize_embeddings": True,
            "show_progress_bar" : False,
        }
        if kwargs: params.update(kwargs)

        if "prompt" in kwargs:
            if kwargs["prompt"] != '':
                params["prompt"] = f"Instruct: {kwargs['prompt']}\nQuery: "
            # del params["prompt"]

        batch_size = params.get("batch_size", 16)

        logger.debug(f"Calling {self.__class__.__name__} with:\n{params}")
        if len(texts) <= batch_size:
            params["sentences"] = texts  # self._add_eos(texts=texts)
            results = self.embedding_model.encode(**params)
        else:
            # pbar = tqdm(total=len(texts), desc="Batch Encoding")
            results = []
            for i in range(0, len(texts), batch_size):
                params["sentences"] = texts[i:i + batch_size]
                results.append(self.embedding_model.encode(**params))
                # pbar.update(batch_size)
            # pbar.close()
            # results = toWe conducted ablation experiments on the key modules used in our method, as shown in Table 1. First, we ablated the local and web search agents. Due to the lack of local/network information, the performance decreased. Second, we ablated the cross-agent search result refiner, i.e., directly providing the planner with the content of agent rollouts. Since the rollouts contained irrelevant search results and hallucinations from low-level agents, the overall performance was affected.rch.cat(results, dim=0)
            results = np.concatenate(results, axis=0)

        if isinstance(results, torch.Tensor):
            results = results.cpu()
            results = results.numpy()
        if params.get("norm", False):
            results = (results.T / np.linalg.norm(results, axis=1)).T

        return results

    def info_relevance(self, agent_collected_info: List[str], query_info: str) -> Union[List[float], np.ndarray]:
        """
        Calculate the relevance of the agent collected information to the query.
        :param agent_collected_info: List of dictionaries containing 'contents' and 'source'.
        :param query_info: Dictionary containing 'query' and 'data_source'.
        :return: Relevance score.
        """
        if not agent_collected_info:
            loguru.logger.warning("Agent collected info is empty, returning empty relevance scores.")
            return []
        qinfo_embeddings = self.batch_encode([query_info], norm=True)

        cinfo_embedding = self.batch_encode(agent_collected_info, norm=True)
        qinfo_embeddings = np.stack(qinfo_embeddings, axis=0)
        query_entity_scores = np.dot(qinfo_embeddings, cinfo_embedding.T)  # shape: (#facts, )
        query_entity_scores = np.squeeze(query_entity_scores) if query_entity_scores.ndim == 2 else query_entity_scores
        query_entity_scores = min_max_normalize(query_entity_scores)
        return query_entity_scores

    def refine_evidences(self, agent_collected_info: Dict[str, Union[str, float]], cross_agent_conlusions: List[Dict[str, Union[str, float]]]=None, filter_ratio_2: float=0.3) -> List[Dict[str, Union[str, float]]]:
        """
        Refine the evidences by removing duplicates and sorting by relevance score.
        :param agent_collected_info: List of dictionaries containing 'contents', 'source', 'instant_score', and 'retained'.
        :return: Refined list of evidences.
        """
        if not agent_collected_info:
            return []
        # remove duplicates
        instantly_retained_evidences, remaining_evidences = {}, {}
        conclusion = agent_collected_info["conclusions"][0]["contents"] if agent_collected_info["conclusions"] else ""
        for evidence in agent_collected_info["evidences"]:
            contents = evidence["contents"]
            retained = evidence["retained"]
            instant_score = evidence["instant_score"]
            if contents in instantly_retained_evidences:
                legacy_score = instantly_retained_evidences[contents]["instant_score"]
                instantly_retained_evidences[contents]["instant_score"] = max(legacy_score, instant_score)
            elif contents in remaining_evidences:
                legacy_score = remaining_evidences[contents]["instant_score"]
                instant_score = max(legacy_score, instant_score)
                evidence["instant_score"] = instant_score
                if retained:
                    # move to instantly retained evidences
                    instantly_retained_evidences[contents] = evidence
                    remaining_evidences.pop(contents, None)
                else:
                    # keep in remaining evidences
                    remaining_evidences[contents] = evidence
            else: # in neither
                if retained:
                    instantly_retained_evidences[contents] = evidence
                else:
                    remaining_evidences[contents] = evidence
        # calculate relevance score for remaining evidences, and select top 30%
        num_global_retained = int(len(remaining_evidences) * filter_ratio_2)
        if num_global_retained == 0:
            num_global_retained = 1
        remaining_evidences = list(remaining_evidences.values())
        remaining_scores = self.info_relevance([e["contents"] for e in remaining_evidences], conclusion)
        if cross_agent_conlusions:
            cross_agent_conlusion = " ".join([c["contents"] for c in cross_agent_conlusions])
            cross_agent_scores = self.info_relevance([e["contents"] for e in remaining_evidences], cross_agent_conlusion)
            # merge the two scores
            remaining_scores = remaining_scores + cross_agent_scores
        top_indices = np.argsort(remaining_scores)[-num_global_retained:]
        global_retained_evidences = [remaining_evidences[i] for i in top_indices]
        # merge the two instantly retained evidences and globally retained evidences
        refined_evidences = list(instantly_retained_evidences.values()) + global_retained_evidences
        return refined_evidences

def init_retriever(args):
    global retriever_semaphore

    for i in range(args.num_agents):
        print(f"Initializing retriever {i+1}/{args.num_agents}")
        embedding_model_device = f"cuda:{i}"
        local_search_agent_config = {
            "search_domain": "local",
            "remote_llm_url": args.local_agent_llm_url,
            "remote_retriever_url": args.remote_retriever_url,
            "remote_web_retriever_url": args.remote_web_retriever_url,
            "remote_web_browse_url": args.remote_web_browse_url,
            "model_path": args.local_agent_llm_model_path,
            "serve_model_name": args.local_agent_serve_model_name,
            "supported_tools": ['chunk_search', 'graph_search', 'get_adjacent_passages'],
            "max_turns": args.max_turns,
            "embedding_model_name": args.embedding_model_name,
            "embedding_model_device": embedding_model_device,
        }
        local_search_agent = SearchAgent(**local_search_agent_config)
        embedding_model_device = f"cuda:{i}"
        web_search_agent_config = {
            "search_domain": "web",
            "remote_llm_url": args.web_agent_llm_url,
            "remote_retriever_url": args.remote_retriever_url,
            "remote_web_retriever_url": args.remote_web_retriever_url,
            "remote_web_browse_url": args.remote_web_browse_url,
            "model_path": args.web_agent_llm_model_path,
            "serve_model_name": args.web_agent_serve_model_name,
            "supported_tools": ["web_search", "browse_url"],
            "max_turns": args.max_turns,
            "embedding_model_name": args.embedding_model_name,
            "embedding_model_device": embedding_model_device,
        }
        web_search_agent = SearchAgent(**web_search_agent_config)
        agent_list.append(local_search_agent)
        agent_list.append(web_search_agent)
        available_local_search_agents.append(i*2)  # even indices for local search agents
        available_web_search_agents.append(i*2 + 1) # odd indices for web search agents
    # create a semaphore to limit the number of retrievers that can be used concurrently
    retriever_semaphore = asyncio.Semaphore(args.num_agents)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "retrievers": {
            "total": len(agent_list),
            "local_available": len(available_local_search_agents),
            "web_available": len(available_web_search_agents),
        }
    }

class QueryRequest(BaseModel):
    query: str
    sample_id: Optional[str] = None
    data_source: Optional[str] = None

class BatchQueryRequest(BaseModel):
    query: List[str]
    sample_id: Optional[List[str]] = None
    data_source: Optional[List[str]] = None
    return_rollouts: Optional[bool] = False
    filter_ratio: Optional[Union[float, List[float]]] = 0.3


class Information(BaseModel):
    source: str
    contents: str



@app.post("/local_search_agent", response_model=List[List[Information]])
async def local_search_agent(request: BatchQueryRequest):
    # perform local search and web search simultaneously
    query = request.query
    sample_id = request.sample_id
    data_source = request.data_source
    return_rollouts = request.return_rollouts
    if isinstance(request.filter_ratio, float):
        filter_ratio_1, filter_ratio_2 = request.filter_ratio, request.filter_ratio
    else:
        filter_ratio_1, filter_ratio_2 = request.filter_ratio

    async with retriever_semaphore:
        retriever_idx = available_local_search_agents.popleft()
        try:
            local_search_agent = agent_list[retriever_idx]
            result_list = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                future_list = [executor.submit(
                    local_search_agent.agent_rollout,
                    question=q,
                    sample_id=sample_id[i] if sample_id else None,
                    data_source=data_source[i] if data_source else None,
                    return_rollout_sequence=return_rollouts,
                    filter_ratio_1=filter_ratio_1,
                ) for i, q in enumerate(query) if q.strip() != ""
                ]
                for i, f in enumerate(future_list):
                    try:
                        result = await asyncio.wrap_future(f)
                        if result:
                            result_list.append((i, result))
                    except Exception as e:
                        loguru.logger.error(f"Error in processing query {query[i]}: {e}")
            # order the results by the original query order
            result_list.sort(key=lambda x: x[0])
            local_rollouts = [r["rollout_sequence"] for _, r in result_list] if return_rollouts else []
            result_list = [local_search_agent.refine_evidences(r, filter_ratio_2=filter_ratio_2) for _, r in result_list]
            agent_collected_info = []

            for i, info in enumerate(result_list):
                if not info:
                    agent_collected_info.append([Information(source="No results", contents="Web Agent Failed")])
                else:
                    agent_collected_info.append(
                        [Information(source=info_item['source'], contents=info_item['contents']) for info_item in info])
                if return_rollouts:
                    agent_collected_info[i].append(Information(source="Local Agent Rollout", contents=local_rollouts[i]))
            return agent_collected_info
        finally:
            available_local_search_agents.append(retriever_idx)

@app.post("/web_search_agent", response_model=List[List[Information]])
async def web_search_agent(request: BatchQueryRequest):
    # perform local search and web search simultaneously
    query = request.query
    sample_id = request.sample_id
    data_source = request.data_source
    return_rollouts = request.return_rollouts
    if isinstance(request.filter_ratio, float):
        filter_ratio_1, filter_ratio_2 = request.filter_ratio, request.filter_ratio
    else:
        filter_ratio_1, filter_ratio_2 = request.filter_ratio

    async with retriever_semaphore:
        retriever_idx = available_web_search_agents.popleft()
        try:
            web_search_agent = agent_list[retriever_idx]
            result_list = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                future_list = [executor.submit(
                    web_search_agent.agent_rollout,
                    question=q,
                    sample_id=sample_id[i] if sample_id else None,
                    data_source=data_source[i] if data_source else None,
                    return_rollout_sequence=return_rollouts,
                    filter_ratio_1=filter_ratio_1,
                ) for i, q in enumerate(query) if q.strip() != ""
                ]
                for i, f in enumerate(future_list):
                    try:
                        result = await asyncio.wrap_future(f)
                        if result:
                            result_list.append((i, result))
                    except Exception as e:
                        loguru.logger.error(f"Error in processing query {query[i]}: {e}")
            # order the results by the original query order
            result_list.sort(key=lambda x: x[0])
            web_rollouts = [r["rollout_sequence"] for _, r in result_list] if return_rollouts else []
            result_list = [web_search_agent.refine_evidences(r, filter_ratio_2=filter_ratio_2) for _, r in result_list]
            agent_collected_info = []

            for i, info in enumerate(result_list):
                if not info:
                    agent_collected_info.append([Information(source="No results", contents="Web Agent Failed")])
                else:
                    agent_collected_info.append([Information(source=info_item['source'], contents=info_item['contents']) for info_item in info])
                if return_rollouts:
                    agent_collected_info[i].append(Information(source="Web Agent Rollout", contents=web_rollouts[i]))
            return agent_collected_info
        finally:
            available_web_search_agents.append(retriever_idx)

@app.post("/all_search_agent", response_model=List[List[Information]])
async def all_search_agent(request: BatchQueryRequest):
    # perform local search and web search simultaneously
    query = request.query
    sample_id = request.sample_id
    data_source = request.data_source
    return_rollouts = request.return_rollouts
    if isinstance(request.filter_ratio, float):
        filter_ratio_1, filter_ratio_2 = request.filter_ratio, request.filter_ratio
    else:
        filter_ratio_1, filter_ratio_2 = request.filter_ratio

    async with retriever_semaphore:
        local_retriever_idx = available_local_search_agents.popleft()
        web_retriever_idx = available_web_search_agents.popleft()
        try:
            # perform search simultaneously
            local_search_agent: SearchAgent = agent_list[local_retriever_idx]
            web_search_agent: SearchAgent = agent_list[web_retriever_idx]
            local_result_list, web_result_list = [], []
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                future_list = []
                for i, q in enumerate(query):
                    if q.strip() != "":
                        future_list.append(executor.submit(
                            local_search_agent.agent_rollout,
                            question=q,
                            sample_id=sample_id[i] if sample_id else None,
                            data_source=data_source[i] if data_source else None,
                            return_rollout_sequence=return_rollouts,
                            filter_ratio_1=filter_ratio_1,
                        ))
                        future_list.append(executor.submit(
                            web_search_agent.agent_rollout,
                            question=q,
                            sample_id=sample_id[i] if sample_id else None,
                            data_source=data_source[i] if data_source else None,
                            return_rollout_sequence=return_rollouts,
                            filter_ratio_1=filter_ratio_1,
                        ))
                for i, f in enumerate(future_list):
                    try:
                        result = await asyncio.wrap_future(f)
                        if result:
                            if i % 2 == 0:
                                local_result_list.append((i, result))
                            else:
                                web_result_list.append((i, result))
                    except Exception as e:
                        loguru.logger.error(f"Error in processing query {query[i]}: {e}")
            # order the results by the original query order
            local_result_list.sort(key=lambda x: x[0])
            web_result_list.sort(key=lambda x: x[0])
            # result_list = local_result_list + web_result_list
            result_list = []
            local_rollouts, web_rollouts = [], []
            for i in range(len(query)):
                local_info = local_search_agent.refine_evidences(
                    local_result_list[i][1],
                    web_result_list[i][1]["conclusions"],
                    filter_ratio_2=filter_ratio_2,
                )
                web_info = web_search_agent.refine_evidences(
                    web_result_list[i][1],
                    local_result_list[i][1]["conclusions"],
                    filter_ratio_2=filter_ratio_2,
                )
                result_list.append(local_info + web_info)
                if return_rollouts:
                    local_rollouts.append(local_result_list[i][1]["rollout_sequence"])
                    web_rollouts.append(web_result_list[i][1]["rollout_sequence"])

            agent_collected_info = []
            for i, info in enumerate(result_list):
                if not info:
                    agent_collected_info.append([Information(source="No results", contents="Web Agent Failed")])
                else:
                    agent_collected_info.append(
                        [Information(source=info_item['source'], contents=info_item['contents']) for info_item in info])
                if return_rollouts:
                    agent_collected_info[i].append(Information(source="Local Agent Rollout", contents=local_rollouts[i]))
                    agent_collected_info[i].append(Information(source="Web Agent Rollout", contents=web_rollouts[i]))
            return agent_collected_info
        finally:
            available_local_search_agents.append(local_retriever_idx)
            available_web_search_agents.append(web_retriever_idx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", type=int, default=1)
    parser.add_argument("--port", type=int, default=16006)
    parser.add_argument("--save_note", type=str, default='your-save-note-for-identification')
    parser.add_argument("--local_agent_llm_url", type=str, default="http://172.24.88.76:18011/v1")
    parser.add_argument("--web_agent_llm_url", type=str, default="http://172.24.88.76:18011/v1")
    parser.add_argument("--remote_retriever_url", type=str, default="http://172.24.88.76")
    parser.add_argument("--remote_web_retriever_url", type=str, default="http://172.24.88.76")
    parser.add_argument("--remote_web_browse_url", type=str, default="")
    parser.add_argument("--local_agent_llm_model_path", type=str,
                        default="")
    parser.add_argument("--web_agent_llm_model_path", type=str,
                        default="")
    parser.add_argument("--local_agent_serve_model_name", type=str, default="")
    parser.add_argument("--web_agent_serve_model_name", type=str, default="")
    parser.add_argument("--max_turns", type=int, default=8, help="maximum number of turns for the conversation")
    parser.add_argument("--parallel_size", type=int, default=16,)
    parser.add_argument("--embedding_model_name", type=str, default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--filter_ratio_1", type=float, default=0.2)
    parser.add_argument("--filter_ratio_2", type=float, default=0.2)

    args = parser.parse_args()

    init_retriever(args)

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()
