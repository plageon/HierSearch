import os
from typing import List, Dict

from Tea.exceptions import TeaException
from alibabacloud_iqs20241111 import models
from alibabacloud_iqs20241111.client import Client
from alibabacloud_tea_openapi import models as open_api_models
class AlibabaSearch:
    def __init__(self):
        pass
    @staticmethod
    def create_client() -> Client:
        config = open_api_models.Config(
            # TODO: 使用您的AK/SK进行替换(建议通过环境变量加载)
            access_key_id='',
            access_key_secret=''
        )
        config.endpoint = f'iqs.cn-zhangjiakou.aliyuncs.com'
        return Client(config)

    @classmethod
    def search(cls, query, count=10) -> List[Dict]:
        client = cls.create_client()
        run_instances_request = models.UnifiedSearchRequest(
            body=models.UnifiedSearchInput(
                query=query,
                time_range='NoLimit',
                contents=models.RequestContents(
                    summary=True,
                    main_text=True,
                )
            )
        )
        res = []
        try:
            response = client.unified_search(run_instances_request)
            print(
                f"api success, request_id:{response.body.request_id}, size :{len(response.body.page_items)}, server_cost:{response.body.search_information.search_time}")
            if len(response.body.scene_items) > 0:
                print(f"scene_items:{response.body.scene_items[0]}")

            for index, item in enumerate(response.body.page_items):
                res.append({
                    "title": item.title,
                    "snippet": item.snippet,
                    "summary": item.summary,
                    "published_time": item.published_time,
                    "link": item.link,
                    "rerank_score": item.rerank_score
                })
            return res
        except TeaException as e:
            code = e.code
            request_id = e.data.get("requestId")
            message = e.data.get("message")
            print(f"api exception, requestId:{request_id}, code:{code}, message:{message}")
            return []
if __name__ == '__main__':
    query = '中国银行财报'
    alibaba_search = AlibabaSearch()
    results = alibaba_search.search(query)
    for result in results:
        print(result)