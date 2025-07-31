# -*- coding: utf-8 -*-
import sys
import json
import logging
import requests
from urllib.parse import urlencode

from bs4 import BeautifulSoup


#Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36
class BingSearch(object):
    
    BASE_PATH = "https://api.bing.microsoft.com/v7.0/search?{}"
    SECRET_KEY = ""

    headers = {
        "User-Agent": "Mozilla/5.0 (Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Mobile Safari/537.36 EdgA/123.0.0.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
    }
    
    @classmethod
    def _query(cls, query, count, mkt="zh-CN"):
        params = {
            "q": query,
            "textDecorations":False,
            "mkt":"en-US",
            "count": count,
            "SafeSearch": "Strict"
        }
        
        url = BingSearch.BASE_PATH.format(urlencode(params))
        # print(url)
        rsp = requests.get(url, headers={'Ocp-Apim-Subscription-Key': BingSearch.SECRET_KEY})
        if rsp is None or rsp.status_code != 200:
            print(f"request bing search api failed, rsp:{rsp.text}")
            return False, "http request failed"
        
        return True, rsp.json()
    
    @classmethod
    def search(cls, query: str, count=10, db_handler=None):
        ret, jdata = cls._query(query, count)
        if ret == False or 'webPages' not in jdata or 'value' not in jdata['webPages']:
            # logging.error("[search]-[error] query failed, query:{}, msg:{}".format(query, jdata))
            return []
        
        # print(json.dumps(jdata, ensure_ascii=False))
        return jdata['webPages']['value']


class ZhipuSearch(object):

    @classmethod
    def search(self, query: str, count=10, db_handler=None):
        from zhipuai import ZhipuAI
        client = ZhipuAI(api_key="")
        response = client.web_search.web_search(
            search_engine="search-std",
            search_query=query,
        )
        return response

    
if __name__ == '__main__':
    query = 'first president of the Democratic Republic of Congo after independence'
    if len(sys.argv) >= 2: query=sys.argv[1]
    res = BingSearch.delegate_search(query)
    # res = BingSearch.search(query)
    # res = ZhipuSearch.search(query)
    for item in res['Search Results']:
        print(f"Title: {item['title']}, Link: {item['link']}")
