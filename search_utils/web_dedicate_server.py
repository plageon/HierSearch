from __future__ import annotations

import time
from typing import List, Tuple, Dict
import json
import argparse
import logging
from chardet.universaldetector import UniversalDetector
import loguru
import requests
import concurrent.futures
import torch
from fastapi import FastAPI, HTTPException
import argparse
from pydantic import BaseModel
from typing import List, Tuple, Union
import asyncio
from collections import deque
import sys
import asyncio
sys.path.append("./")
from search_utils.alibaba_search import AlibabaSearch
from search_utils.bing_search import BingSearch
from search_utils.html_splitter import HTMLHeaderTextSplitter

sys.path.append("./src/")
logger = logging.getLogger(__name__)
app = FastAPI()

retriever_list = []
task_list = []
available_retrievers = deque()
retriever_semaphore = None
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import numpy as np



class WebSearchServer:
    def __init__(self,
                 embedding_model_name,
                 embedding_model_device,
                 ):
        self.embedding_model = SentenceTransformer(
            embedding_model_name, trust_remote_code=True, device=embedding_model_device,
        )
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
        self.web_snippets = []
        self.url2snippet_id = {}
        self.webpages = []
        self.url_black_list = set()
        # self.selenium_search = SeleniumBingSearch()

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
            results = np.concatenate(results, axis=0)

        if isinstance(results, torch.Tensor):
            results = results.cpu()
            results = results.numpy()
        if params.get("norm", False):
            results = (results.T / np.linalg.norm(results, axis=1)).T

        return results

    async def web_search(self, query: Union[str, List[str]], top_k: int):
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
                    # res = BingSearch.search(q, top_k)
                    # res = self.selenium_search.search(q, top_k)
                    if en_search:
                        res = BingSearch.delegate_search(q, top_k)["Search Results"]
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
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(search, q, top_k) for q in query]
            for i in range(len(query)):
                search_results[i] = await asyncio.wrap_future(futures[i])

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

    async def webpage_browse(self, url: Union[str, List[str]], query: Union[str, List[str]], top_k: int):
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

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Connection": "keep-alive",
        }
        # send request to bing search in parallel
        # using concurrent.futures to send request in parallel

        def get_decoded_content(response):
            # 1. 尝试使用HTTP头中的编码
            if response.encoding != 'ISO-8859-1':
                return response.text

            # 2. 使用chardet检测编码
            detector = UniversalDetector()
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    detector.feed(chunk)
                    if detector.done:
                        break
            detector.close()

            # 应用检测到的编码
            if detector.result['confidence'] > 0.7:  # 设置置信度阈值
                response.encoding = detector.result['encoding']
                return response.text

            # 3. 备选：从HTML元标签中提取编码（简化版）
            try:
                # 检查前1024字节的HTML内容
                html_prefix = response.content[:1024].decode('latin-1')
                # 简单正则匹配<meta>标签中的charset
                import re
                match = re.search(r'<meta.*?charset=["\']?([^"\'>]*)', html_prefix, re.IGNORECASE)
                if match:
                    detected_encoding = match.group(1)
                    response.encoding = detected_encoding
                    return response.text
            except:
                pass

            # 4. 作为最后的手段，使用默认解码
            return response.text  # 回退到ISO-8859-1或chardet的最佳猜测

        def get_web_content(url, use_proxy=False):
            patience = 2
            page_html = ""
            if url in self.url_black_list:
                loguru.logger.info(f"URL {url} is in black list, skipping")
                return ""
            while patience > 0:
                try:
                    if use_proxy:
                        res = requests.get(url, headers=headers, proxies=proxies, timeout=30, allow_redirects=True)
                    else:
                        res = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
                    try:
                        page_html = get_decoded_content(res)
                    except Exception as e:
                        loguru.logger.error(f"Error decoding content from {url}: {e}")
                        page_html = res.content
                    # loguru.logger.info(res)
                    if page_html is None or "<html>" not in page_html:
                        patience -= 1
                        time.sleep(1)
                        continue
                    break
                except Exception as e:
                    loguru.logger.error(f"Error fetching content from {url}")
                    self.url_black_list.add(url)
                    patience -= 1
                    time.sleep(1)
            return page_html

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for i in range(len(query)):
                # check if query contains Chinese characters
                if any('\u4e00' <= char <= '\u9fff' for char in query[i]):
                    futures.append(executor.submit(get_web_content, url[i], use_proxy=False))
                else:
                    futures.append(executor.submit(get_web_content, url[i], use_proxy=True))
            for i in range(len(query)):
                request_results[i] = await asyncio.wrap_future(futures[i])
        browse_results = []
        for idx in range(len(query)):
            query[idx] = query[idx].lower()
            webcontent = request_results[idx]
            snippet_id = self.url2snippet_id.get(url[idx], None)
            if snippet_id is None:
                # find title from webpage
                if "<title>" not in webcontent:
                    title = ""
                else:
                    title = webcontent.split("<title>")[1].split("</title>")[0]
            else:
                title = self.web_snippets[snippet_id]["title"]
            if webcontent is None or webcontent == "":
                loguru.logger.info(f"Web content is None for {url[idx]}")
                browse_results.append([{
                    "id": f"webpage-0",
                    "title": title,
                    "url": url[idx],
                    "contents": f"Web content is None for {url[idx]}",
                }])
                continue
            if any('\u4e00' <= char <= '\u9fff' for char in query[idx]):
                documents = self.zh_html_splitter.split_text(webcontent)
            else:
                documents = self.en_html_splitter.split_text(webcontent)

            webpages = []
            new_docs = []
            # loguru.logger.info(f"Processing {len(documents)} documents for query: {query[idx]} and url: {url[idx]}")
            for i, doc in enumerate(documents):
                content = doc.page_content
                content.replace("\n", " ")
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
            # loguru.logger.info(f"Reranking {len(webpages)} webpages for query: {query[idx]} and url: {url[idx]}")
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
        query_embeddings = self.batch_encode([query], instruction="", norm=True)
        html_splits_embeddings = self.batch_encode([split['contents'] for split in html_splits],
                                                                   instruction="",
                                                                   norm=True)

        # compute cosine similarity
        scores = np.dot(query_embeddings, html_splits_embeddings.T)
        scores = scores[0]

        # sort by score
        sorted_indices = np.argsort(scores)[::-1]
        sorted_html_splits = [html_splits[i] for i in sorted_indices]
        return sorted_html_splits

def init_retriever(args):
    global retriever_semaphore

    for i in range(args.num_retriever):
        print(f"Initializing retriever {i+1}/{args.num_retriever}")
        retriever = WebSearchServer(
            embedding_model_name=args.embedding_model_name,
            embedding_model_device=f"cuda:{i}",
        )
        retriever_list.append(retriever)
        available_retrievers.append(i)
        task_list.append(())
    # create a semaphore to limit the number of retrievers that can be used concurrently
    retriever_semaphore = asyncio.Semaphore(args.num_retriever)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "retrievers": {
            "total": len(retriever_list),
            "available": len(available_retrievers)
        },
        "tasks": task_list
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


@app.post("/batch_web_search", response_model=Union[List[List[Document]], Tuple[List[List[Document]], List[List[float]]]])
async def batch_web_search(request: BatchQueryRequest):
    # perform web search
    query = request.query
    top_n = request.top_n

    async with retriever_semaphore:
        retriever_idx = available_retrievers.popleft()
        task_list[retriever_idx] = ("batch_web_search", query, top_n)
        try:
            results = await retriever_list[retriever_idx].web_search(query, top_n)
            assert len(results) == len(query), f"len(results) {len(results)} != len(query) {len(query)}\nresults: {results}"
            return [[Document(id=result['id'], contents=result['contents']) for result in results[i]] for i in range(len(results))]
        finally:
            available_retrievers.append(retriever_idx)
            task_list[retriever_idx] = ()

@app.post("/batch_webpage_browse", response_model=Union[List[List[Document]], Tuple[List[List[Document]], List[List[float]]]])
async def batch_browse_search(request: BatchBrowseRequest):
    # perform webpage browse
    query = request.query
    top_n = request.top_n
    url = request.url

    async with retriever_semaphore:
        retriever_idx = available_retrievers.popleft()
        task_list[retriever_idx] = ("batch_webpage_browse", query, top_n, url)
        try:
            results = await retriever_list[retriever_idx].webpage_browse(url, query, top_n)
            return [[Document(id=result['id'], contents=result['contents']) for result in results[i]] for i in range(len(results))]
        finally:
            available_retrievers.append(retriever_idx)
            task_list[retriever_idx] = ()

@app.post("/web_search", response_model=List[Document])
async def web_search(request: QueryRequest):
    # perform web search
    query = request.query
    top_n = request.top_n

    async with retriever_semaphore:
        retriever_idx = available_retrievers.popleft()
        task_list[retriever_idx] = ("web_search", query, top_n)
        try:
            results = await retriever_list[retriever_idx].web_search(query, top_n)
            if not results[0] or isinstance(results[0], str):
                return [Document(id="Error", contents=f"No results found for query: {query}")]
            else:
                return [Document(id=doc['id'], contents=doc['contents']) for doc in results[0]]
        finally:
            available_retrievers.append(retriever_idx)
            task_list[retriever_idx] = ()

@app.post("/webpage_browse", response_model=List[Document])
async def webpage_browse(request: QueryRequest):
    # perform webpage browse
    query = request.query
    top_n = request.top_n
    url = request.url

    async with retriever_semaphore:
        retriever_idx = available_retrievers.popleft()
        task_list[retriever_idx] = ("webpage_browse", query, top_n, url)
        try:
            results = await retriever_list[retriever_idx].webpage_browse(url, query, top_n)
            if not results[0] or isinstance(results[0], str):
                return [Document(id="Error", contents=f"No results found for query: {query} and url: {url}")]
            else:
                return [Document(id=doc['id'], contents=doc['contents']) for doc in results[0]]
        finally:
            available_retrievers.append(retriever_idx)
            task_list[retriever_idx] = ()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_retriever", type=int, default=1)
    parser.add_argument("--port", type=int, default=15005)

    # hippoRAG specific arguments
    parser.add_argument("--embedding_model_name", type=str, default="model/bge-m3")

    args = parser.parse_args()

    init_retriever(args)

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()
