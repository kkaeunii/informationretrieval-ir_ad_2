"""Elasticsearch 연결 및 인덱스 관리 유틸."""

from __future__ import annotations

import json

from typing import Dict, Iterable, List

from elasticsearch import Elasticsearch, helpers

from config.settings import ElasticsearchConfig


# create_client는 설정을 바탕으로 Elasticsearch 클라이언트를 생성한다.
def create_client(es_config: ElasticsearchConfig) -> Elasticsearch:
    client_kwargs = {
        "hosts": [es_config.host],
        "request_timeout": 30,
    }
    if es_config.username and es_config.password:
        client_kwargs["basic_auth"] = (es_config.username, es_config.password)
    if es_config.ca_cert:
        client_kwargs["ca_certs"] = es_config.ca_cert
    return Elasticsearch(**client_kwargs)


# recreate_index는 동일 이름의 인덱스를 삭제 후 재생성한다.
def recreate_index(
    client: Elasticsearch,
    index_name: str,
    settings: Dict,
    mappings: Dict,
):
    if client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)
    client.indices.create(index=index_name, settings=settings, mappings=mappings)


# bulk_index_documents는 문서 리스트를 지정한 인덱스에 일괄 색인한다.
def bulk_index_documents(client: Elasticsearch, index_name: str, docs: Iterable[Dict]):
    actions = [{"_index": index_name, "_source": doc} for doc in docs]
    helpers.bulk(client, actions)


# fetch_all_documents는 JSONL 데이터를 읽어 파싱한 뒤 반환한다.
def fetch_all_documents(jsonl_path: str) -> List[Dict]:
    docs: List[Dict] = []
    with open(jsonl_path, encoding="utf-8") as file:
        for line in file:
            docs.append(json.loads(line))
    return docs
