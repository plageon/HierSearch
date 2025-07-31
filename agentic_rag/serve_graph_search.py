from __future__ import annotations

import time
from typing import List, Tuple, Dict
import json
import argparse
import logging

import loguru
import requests
import concurrent.futures

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig
from hipporag.utils.logging_utils import get_logger
from hipporag.utils.misc_utils import min_max_normalize, compute_mdhash_id

from fastapi import FastAPI, HTTPException
import argparse
from pydantic import BaseModel
from typing import List, Tuple, Union
import asyncio
from collections import deque
import sys

sys.path.append("./")
from search_utils.alibaba_search import AlibabaSearch
from search_utils.bing_search import BingSearch
from search_utils.html_splitter import HTMLHeaderTextSplitter

sys.path.append("./src/")
logger = logging.getLogger(__name__)
app = FastAPI()

retriever_list = []
available_retrievers = deque()
retriever_semaphore = None
from typing import List, Optional

import numpy as np

logger = get_logger(__name__)


class WebSearchMixin:
    def init_web_search(self):
        headers_to_split_on = [("h1", "Header 1"), ("h2", "Header 2"), ("h3", "Header 3"), ("h4", "Header 4")]
        self.en_html_splitter = HTMLHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            chunk_size=2048,
            chunk_overlap=0,
            separators=["\n\n", "\n", ".", ";", ",", " ", ""],
            keep_separator=True,
        )
        self.zh_html_splitter = HTMLHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            chunk_size=512,
            chunk_overlap=0,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
            keep_separator=True,
        )

    def web_search(self, query: Union[str, List[str]], top_k: int):
        if len(query) == 0:
            return 'invalid query'
        if isinstance(query, str):
            query = [query]

        result_list = []

        # send request to bing search in parallel
        # using concurrent.futures to send request in parallel
        def search(q, top_k):
            patience = 5
            res = []
            if any('\u4e00' <= char <= '\u9fff' for char in q):
                en_search = False
            else:
                en_search = True
            while patience > 0:
                try:
                    if en_search:
                        # res = BingSearch.delegate_search(q, top_k)["Search Results"]
                        res = BingSearch.search(q, top_k)
                    else:
                        res = AlibabaSearch.search(q, top_k)
                    # loguru.logger.info(res)
                    if res is None or len(res) == 0:
                        patience -= 1
                        time.sleep(3)
                        if patience == 0:
                            loguru.logger.info(f"Search results is None for {q}")
                        continue
                    break
                except Exception as e:
                    patience -= 1
                    time.sleep(3)
            return res

        search_results = [None for _ in range(len(query))]
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(search, q, top_k) for q in query]
            for i in range(len(query)):
                search_results[i] = futures[i].result()

        # process search results
        for i, q in enumerate(query):
            if q == '':
                return 'invalid query'
            try:
                assert len(search_results) == len(query)
                res = search_results[i]
                if res is None:
                    loguru.logger.info(f"Search results is None for {q}")
                    result_list.append([])
                    continue
                # update res to index
                new_docs = []
                snippets = []
                for i in range(len(res)):
                    title = res[i]['title']
                    url = res[i]['link']
                    snippet = res[i]['snippet']
                    contents = f"{title} | {url}\n{snippet}"
                    new_docs.append(contents)
                    snippet = {
                        "id": f"web-snippet-{i}",
                        "title": title,
                        "url": url,
                        "contents": contents,
                    }
                    # self.url2snippet_id[url] = len(self.web_snippets)
                    # self.web_snippets.append(snippet)
                    snippets.append(snippet)

                # self.index(new_docs)

            except Exception as e:
                snippets = "Error processing search results"
                # loguru.logger.error(f"Error: {e}")
            result_list.append(snippets)

        return result_list

    def webpage_browse(self, url: Union[str, List[str]], query: Union[str, List[str]], top_k: int):
        if len(query) == 0 or len(url) == 0:
            return 'invalid query'
        if isinstance(query, str):
            query = [query]
            url = [url]
        assert len(query) == len(url), f"len(query) {len(query)} != len(url) {len(url)}\nquery: {query}\nurl: {url}"

        request_results = ["" for _ in range(len(query))]
        proxies = {
            "http": "http://",
            "https": "http://",
        }
        # send request to bing search in parallel
        # using concurrent.futures to send request in parallel

        def get_web_content(url, use_proxy=False):
            patience = 2
            res = ""
            while patience > 0:
                try:
                    if use_proxy:
                        res = requests.get(url, proxies=proxies, timeout=60)
                    else:
                        requests.get(url, timeout=60)
                    try:
                        res = res.content.decode("utf-8")
                    except Exception as e:
                        res = res.text
                    # loguru.logger.info(res)
                    if res is None or "<html>" not in res:
                        patience -= 1
                        time.sleep(1)
                        continue
                    break
                except Exception as e:
                    patience -= 1
                    time.sleep(1)
            return res

        with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
            futures = []
            for i in range(len(query)):
                # check if query contains Chinese characters
                if any('\u4e00' <= char <= '\u9fff' for char in query[i]):
                    futures.append(executor.submit(get_web_content, url[i], use_proxy=False))
                else:
                    futures.append(executor.submit(get_web_content, url[i], use_proxy=True))
            for i in range(len(query)):
                request_results[i] = futures[i].result()
        browse_results = []
        for idx in range(len(query)):
            query[idx] = query[idx].lower()
            webcontent = request_results[idx]
            if webcontent is None:
                loguru.logger.info(f"Web content is None for {url[idx]}")
                browse_results.append(f"Error retrieving content from {url[idx]}")
                continue
            if any('\u4e00' <= char <= '\u9fff' for char in query[idx]):
                documents = self.zh_html_splitter.split_text(webcontent)
            else:
                documents = self.en_html_splitter.split_text(webcontent)
            snippet_id = self.url2snippet_id.get(url[idx], None)
            if snippet_id is None:
                # find title from webpage
                if "<title>" not in webcontent:
                    title = ""
                else:
                    title = webcontent.split("<title>")[1].split("</title>")[0]
            else:
                title = self.web_snippets[snippet_id]["title"]
            webpages = []
            new_docs = []
            # loguru.logger.info(documents)
            for i, doc in enumerate(documents):
                content = doc.page_content
                contents = f"{title} | {url[idx]}\n{content}"
                webpage = {
                    "id": f"webpage-{i}",
                    "title": title,
                    "url": url[idx],
                    "contents": contents,
                }
                # self.webpages.append(webpage)
                webpages.append(webpage)
                new_docs.append(contents)

            # rerank html splits
            if len(webpages) < top_k:
                reranked_webpages = webpages
            else:
                reranked_webpages = self.rerank_html_splits(query[idx], webpages)
                # update webpages to index
                # self.index(new_docs)
                reranked_webpages = reranked_webpages[:top_k]
            browse_results.append(reranked_webpages)

        return browse_results

    def rerank_html_splits(self, query: Union[str, List[str]], html_splits: List[Dict[str, str]]):
        # embed query and html splits
        query_embeddings = self.embedding_model.batch_encode([query],
                                                             instruction="",
                                                             norm=True)
        html_splits_embeddings = self.embedding_model.batch_encode([split['contents'] for split in html_splits],
                                                                   instruction="",
                                                                   norm=True)

        # compute cosine similarity
        scores = np.dot(query_embeddings, html_splits_embeddings.T)
        scores = scores[0]

        # sort by score
        sorted_indices = np.argsort(scores)[::-1]
        sorted_html_splits = [html_splits[i] for i in sorted_indices]
        return sorted_html_splits


class GraphRAGAgent(HippoRAG, WebSearchMixin):
    def __init__(self, global_config=None, save_dir=None, llm_model_name=None, llm_base_url=None,
                 embedding_model_name=None, embedding_base_url=None, azure_endpoint=None,
                 azure_embedding_endpoint=None):
        super().__init__(global_config, save_dir, llm_model_name, llm_base_url, embedding_model_name,
                         embedding_base_url, azure_endpoint, azure_embedding_endpoint)
        self.init_web_search()

    def dense_passage_retrieval_top_k(self, query: Union[str, List[str]], top_k: int):
        if isinstance(query, str):
            query = [query]
        encode_idx = []
        query_embeddings = [None for _ in range(len(query))]
        for idx in range(len(query)):
            query_embedding = self.query_to_embedding['passage'].get(query[idx], None)
            if query_embedding is None:
                encode_idx.append(idx)
            else:
                query_embeddings[idx] = query_embedding
        encode_batch = [query[i] for i in encode_idx]

        if len(encode_batch) > 0:
            query_embedding = self.embedding_model.batch_encode(encode_batch,
                                                                instruction="",
                                                                norm=True)
            for idx, encode_idx in enumerate(encode_idx):
                query_embeddings[encode_idx] = query_embedding[idx]
                self.query_to_embedding['passage'][query[encode_idx]] = query_embedding[idx]
        query_embeddings = np.stack(query_embeddings, axis=0)
        batch_query_doc_scores = np.dot(query_embeddings, self.passage_embeddings.T)
        top_k_docs, top_k_scores = [], []
        for idx in range(len(query)):
            top_k_docs.append([])
            query_doc_scores = min_max_normalize(batch_query_doc_scores[idx])
            sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
            sorted_doc_scores = query_doc_scores[sorted_doc_ids.tolist()]
            for idx in sorted_doc_ids[:top_k]:
                doc = self.chunk_embedding_store.get_row(self.passage_node_keys[idx])
                top_k_docs[-1].append({
                    "id": doc["hash_id"],
                    "contents": doc["content"],
                })
            top_k_scores.append(sorted_doc_scores[:top_k])
        return top_k_docs, top_k_scores

    def graph_search(self, query: Union[str, List[str]], top_k: int):
        if isinstance(query, str):
            query = [query]
        candidate_facts, candidate_fact_scores = [], []
        for idx in range(len(query)):
            query_fact_scores = self.get_fact_scores(query[idx])
            candidate_fact_indices = np.argsort(query_fact_scores)[-top_k:][::-1].tolist()  # list of ranked link_top_k
            real_candidate_fact_ids = [self.fact_node_keys[idx] for idx in candidate_fact_indices]  # list of ranked link_top_k fact keys
            candidate_fact_scores.append(query_fact_scores[candidate_fact_indices])  # list of ranked link_top_k fact scores
            fact_row_dict = self.fact_embedding_store.get_rows(real_candidate_fact_ids)
            candidate_facts.append([eval(fact_row_dict[id]['content']) for id in real_candidate_fact_ids])
        return candidate_facts, candidate_fact_scores

    def get_entity_scores(self, query: List[str]) -> np.ndarray:
        encode_idx = []
        query_embeddings = [None for _ in range(len(query))]

        for idx in range(len(query)):
            query_embedding = self.query_to_embedding['entity'].get(query[idx], None)
            if query_embedding is None:
                encode_idx.append(idx)
            else:
                query_embeddings[idx] = query_embedding

        encode_batch = [query[i] for i in encode_idx]
        if len(encode_batch) > 0:
            query_embedding = self.embedding_model.batch_encode(encode_batch, norm=True)
            for idx, encode_idx in enumerate(encode_idx):
                query_embeddings[encode_idx] = query_embedding[idx]
                self.query_to_embedding['entity'][query[encode_idx]] = query_embedding[idx]
        query_embeddings = np.stack(query_embeddings, axis=0)
        query_entity_scores = np.dot(query_embeddings, self.entity_embeddings.T)  # shape: (#facts, )
        # query_entity_scores = np.squeeze(query_entity_scores) if query_entity_scores.ndim == 2 else query_entity_scores
        query_entity_scores = min_max_normalize(query_entity_scores)

        return query_entity_scores

    def get_adjacent_passages(self, query: Union[str, List[str]]):
        if isinstance(query, str):
            query = [query]
        phrase_ids = ["" for _ in range(len(query))]
        entity_search_idx = []
        for idx in range(len(query)):
            query[idx] = query[idx].lower()
            # get adjacent passages of the query
            node_key = compute_mdhash_id(content=query[idx], prefix=("entity-"))
            phrase_id = self.node_name_to_vertex_idx.get(node_key, None)
            if phrase_id is None:
                entity_search_idx.append(idx)
            else:
                phrase_ids[idx] = phrase_id

        # retrieve the most similar phrase
        try:
            phrases = []
            if len(entity_search_idx) > 0:
                entity_scores = self.get_entity_scores([query[i] for i in entity_search_idx])
                # loguru.logger.info(entity_scores.shape)
                top_phrase_index = np.argmax(entity_scores, axis=-1)
                # loguru.logger.info(top_phrase_index)
                for i, idx in enumerate(entity_search_idx):
                    phrase = self.entity_embedding_store.get_row(self.entity_node_keys[top_phrase_index[i]])["content"]
                    phrases.append(phrase)
                    node_key = compute_mdhash_id(content=phrase, prefix=("entity-"))
                    phrase_id = self.node_name_to_vertex_idx.get(node_key, None)

                    assert phrase_id is not None, f"Phrase {query} not found in the graph."
                    phrase_ids[idx] = phrase_id
        except:
            loguru.logger.info(f"entity_scores.shape {entity_scores.shape}")
            loguru.logger.info(f"top_phrase_index {top_phrase_index}")
            loguru.logger.info(f"phrase {phrase}")
            raise Exception(f"Phrase {query} not found in the graph.")
        # get the adjacent passages
        adjacent_passages = []
        for idx in range(len(query)):
            adjacent_nodes = self.graph.neighbors(phrase_ids[idx])
            adjacent_passages_indices = [i for i in range(len(self.passage_node_idxs)) if self.passage_node_idxs[i] in adjacent_nodes]
            adjacent_passages_keys = [self.passage_node_keys[i] for i in adjacent_passages_indices]
            adjacent_passages.append([self.chunk_embedding_store.get_row(adjacent_passages_key) for adjacent_passages_key in adjacent_passages_keys])
        return adjacent_passages


    def save_local_corpus(self, corpus_path):
        if len(self.web_snippets) > 0:
            with open(f"{corpus_path}/web_snippets.jsonl", "w") as f:
                for snippet in self.web_snippets:
                    f.write(json.dumps(snippet, ensure_ascii=False) + "\n")
        if len(self.webpages) > 0:
            with open(f"{corpus_path}/webpages.jsonl", "w") as f:
                for webpage in self.webpages:
                    f.write(json.dumps(webpage, ensure_ascii=False) + "\n")

def init_retriever(args):
    global retriever_semaphore

    for i in range(args.num_retriever):
        print(f"Initializing retriever {i+1}/{args.num_retriever}")
        corpus_path = args.corpus_path
        with open(corpus_path, "r") as f:
            corpus = json.load(f)

        docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]
        cuda_device = f"cuda:{i}"
        global_config = BaseConfig(save_openie=False, embedding_model_device=cuda_device,)

        rag_server = GraphRAGAgent(
            global_config=global_config,
            save_dir=args.save_dir,
           llm_model_name=args.llm_model_name,
           embedding_model_name=args.embedding_model_name,
           llm_base_url=args.llm_base_url,
           )
        rag_server.index(docs)
        rag_server.prepare_retrieval_objects()
        rag_server.ready_to_retrieve = True
        retriever_list.append(rag_server)
        available_retrievers.append(i)
    # create a semaphore to limit the number of retrievers that can be used concurrently
    retriever_semaphore = asyncio.Semaphore(args.num_retriever)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "retrievers": {
            "total": len(retriever_list),
            "available": len(available_retrievers)
        }
    }

class QueryRequest(BaseModel):
    query: str
    top_n: Optional[int] = 10
    return_score: Optional[bool] = False
    url: Optional[str] = None
    sample_id: Optional[str] = None

class BatchQueryRequest(BaseModel):
    query: List[str]
    top_n: Optional[int] = 10
    return_score: Optional[bool] = False
    url: Optional[List[str]] = None
    sample_id: Optional[List[str]] = None

class BatchBrowseRequest(BaseModel):
    query: List[str]
    url: List[str]
    top_n: int = 10

class Document(BaseModel):
    id: str
    contents: str

class FactTriplet(BaseModel):
    subject_phrase: str
    predicate_phrase: str
    object_phrase: str

@app.post("/search", response_model=Union[Tuple[List[Document], List[float]], List[Document]])
async def search(request: QueryRequest):
    query = request.query
    top_n = request.top_n
    return_score = request.return_score

    if not query or not query.strip():
        print(f"Query content cannot be empty: {query}")
        raise HTTPException(
            status_code=400,
            detail="Query content cannot be empty"
        )

    async with retriever_semaphore:
        retriever_idx = available_retrievers.popleft()
        try:
            if return_score:
                results, scores = retriever_list[retriever_idx].dense_passage_retrieval_top_k(query, top_n)
                return [Document(id=result['id'], contents=result['contents']) for result in results[0]], scores[0]
            else:
                results = retriever_list[retriever_idx].dense_passage_retrieval_top_k(query, top_n)
                return [Document(id=result['id'], contents=result['contents']) for result in results[0]]
        finally:
            available_retrievers.append(retriever_idx)

@app.post("/batch_search", response_model=Union[List[List[Document]], Tuple[List[List[Document]], List[List[float]]]])
async def batch_search(request: BatchQueryRequest):
    # perform local search and web search simultaneously
    query = request.query
    top_n = request.top_n
    return_score = request.return_score

    async with retriever_semaphore:
        retriever_idx = available_retrievers.popleft()
        try:
            if return_score:
                results, scores = retriever_list[retriever_idx].dense_passage_retrieval_top_k(query, top_n)
                return [[Document(id=result['id'], contents=result['contents']) for result in results[i]] for i in range(len(results))], scores
            else:
                results, scores = retriever_list[retriever_idx].dense_passage_retrieval_top_k(query, top_n)
                return [[Document(id=result['id'], contents=result['contents']) for result in results[i]] for i in range(len(results))]
        finally:
            available_retrievers.append(retriever_idx)


@app.post("/batch_web_search", response_model=Union[List[List[Document]], Tuple[List[List[Document]], List[List[float]]]])
async def batch_web_search(request: BatchQueryRequest):
    # perform web search
    query = request.query
    top_n = request.top_n

    async with retriever_semaphore:
        retriever_idx = available_retrievers.popleft()
        try:
            results = retriever_list[retriever_idx].web_search(query, top_n)
            assert len(results) == len(query), f"len(results) {len(results)} != len(query) {len(query)}\nresults: {results}"
            return [[Document(id=result['id'], contents=result['contents']) for result in results[i]] for i in range(len(results))]
        finally:
            available_retrievers.append(retriever_idx)

@app.post("/batch_webpage_browse", response_model=Union[List[List[Document]], Tuple[List[List[Document]], List[List[float]]]])
async def batch_browse_search(request: BatchBrowseRequest):
    # perform webpage browse
    query = request.query
    top_n = request.top_n
    url = request.url

    async with retriever_semaphore:
        retriever_idx = available_retrievers.popleft()
        try:
            results = retriever_list[retriever_idx].webpage_browse(url, query, top_n)
            return [[Document(id=result['id'], contents=result['contents']) for result in results[i]] for i in range(len(results))]
        finally:
            available_retrievers.append(retriever_idx)

@app.post("/web_search", response_model=List[Document])
async def web_search(request: QueryRequest):
    # perform web search
    query = request.query
    top_n = request.top_n

    async with retriever_semaphore:
        retriever_idx = available_retrievers.popleft()
        try:
            results = retriever_list[retriever_idx].web_search(query, top_n)
            if not results[0] or isinstance(results[0], str):
                return [Document(id="Error", contents=f"No results found for query: {query}")]
            else:
                return [Document(id=doc['id'], contents=doc['contents']) for doc in results[0]]
        finally:
            available_retrievers.append(retriever_idx)

@app.post("/webpage_browse", response_model=List[Document])
async def webpage_browse(request: QueryRequest):
    # perform webpage browse
    query = request.query
    top_n = request.top_n
    url = request.url

    async with retriever_semaphore:
        retriever_idx = available_retrievers.popleft()
        try:
            results = retriever_list[retriever_idx].webpage_browse(url, query, top_n)
            if not results[0] or isinstance(results[0], str):
                return [Document(id="Error", contents=f"No results found for query: {query} and url: {url}")]
            else:
                return [Document(id=doc['id'], contents=doc['contents']) for doc in results[0]]
        finally:
            available_retrievers.append(retriever_idx)


@app.post("/graph_search", response_model=Union[List[List[FactTriplet]], Tuple[List[List[FactTriplet]], List[List[float]]]])
async def graph_search(request: BatchQueryRequest):
    query = request.query
    top_n = request.top_n
    return_score = request.return_score
    if not query:
        print(f"Query content cannot be empty: {query}")
        raise HTTPException(
            status_code=400,
            detail="Query content cannot be empty"
        )

    async with retriever_semaphore:
        retriever_idx = available_retrievers.popleft()
        # loguru.logger.info(f"params: {query}, {top_n}, {return_score}")
        try:
            if return_score:
                top_k_facts, top_k_fact_scores = retriever_list[retriever_idx].graph_search(query, top_n)
                # loguru.logger.info(f"top_k_facts: {top_k_facts}, top_k_fact_scores: {top_k_fact_scores}")
                return [[FactTriplet(subject_phrase=f[0], predicate_phrase=f[1], object_phrase=f[2]) for f in top_k_facts[i]] for i in range(len(top_k_facts))], top_k_fact_scores
            else:
                top_k_facts, top_k_fact_scores = retriever_list[retriever_idx].graph_search(query, top_n)
                return [[FactTriplet(subject_phrase=f[0], predicate_phrase=f[1], object_phrase=f[2]) for f in top_k_facts[i]] for i in range(len(top_k_facts))]
        except Exception as e:
            import traceback
            loguru.logger.info(f"Error in graph_search: {e}")
            traceback.print_exc()
        finally:
            available_retrievers.append(retriever_idx)

@app.post("/get_adjacent_passages", response_model=List[List[Document]])
async def get_adjacent_passages(request: BatchQueryRequest):
    query = request.query
    top_n = request.top_n
    # return_score = request.return_score

    if not query:
        print(f"Query content cannot be empty: {query}")
        raise HTTPException(
            status_code=400,
            detail="Query content cannot be empty"
        )

    async with retriever_semaphore:
        retriever_idx = available_retrievers.popleft()
        try:
            adjacent_passages = retriever_list[retriever_idx].get_adjacent_passages(query)
            for i in range(len(adjacent_passages)):
                if len(adjacent_passages[i]) > top_n:
                    adjacent_passages[i] = adjacent_passages[i][:top_n]
            return [[Document(id=passage["hash_id"], contents=passage["content"]) for passage in adjacent_passages[i]] for i in range(len(adjacent_passages))]
        finally:
            available_retrievers.append(retriever_idx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_retriever", type=int, default=1)
    parser.add_argument("--port", type=int, default=18009)

    # hippoRAG specific arguments
    parser.add_argument("--dataset_name", type=str, default="musique")
    parser.add_argument("--save_dir", type=str, default="./data/graph")
    parser.add_argument("--llm_model_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--embedding_model_name", type=str, default="model/bge-m3")
    parser.add_argument("--llm_base_url", type=str, default="")
    parser.add_argument("--corpus_path", type=str, default="data/nq/nq_corpus_train.json")

    args = parser.parse_args()

    init_retriever(args)

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()
