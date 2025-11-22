"""LangGraph 콜백 구성 모듈."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from elasticsearch import Elasticsearch

from config.settings import Settings
from llm.embedding import EmbeddingService
from llm.generators import (
    build_standalone_query,
    create_llm_client,
    generate_final_answer,
)
from pipelines.langgraph_pipeline import RagGraphCallbacks
from retrieval.non_science import is_science_query
from retrieval.retriever import hybrid_retrieve


# RagDependencies는 LangGraph 콜백이 의존하는 자원을 묶어 둔 데이터 클래스다.
@dataclass
class RagDependencies:
    settings: Settings
    es_client: Elasticsearch
    embedder: EmbeddingService


# build_callbacks는 RagGraphCallbacks 인스턴스를 생성해 반환한다.
def build_callbacks(deps: RagDependencies) -> RagGraphCallbacks:
    llm_client = create_llm_client(deps.settings.llm)

    # classify_query는 규칙 기반 비과학 판별기를 호출한다.
    def classify_query(messages: List[Dict]):
        return is_science_query(messages)

    # build_query는 standalone query 생성을 담당한다.
    def build_query(messages: List[Dict]):
        return build_standalone_query(messages, llm_client, deps.settings.llm)

    # retrieve는 하이브리드 검색을 호출한다.
    def retrieve(query_text: str, kwargs: Dict):
        size = kwargs.get("size", 3)
        alpha = kwargs.get("alpha", 0.5)
        return hybrid_retrieve(
            client=deps.es_client,
            index_name=deps.settings.es.index_name,
            query_text=query_text,
            size=size,
            embedder=deps.embedder,
            alpha=alpha,
        )

    # generate_answer는 검색 결과를 바탕으로 최종 답변을 생성한다.
    def generate_answer(messages: List[Dict], docs: List[Dict]):
        return generate_final_answer(messages, docs, llm_client, deps.settings.llm)

    return RagGraphCallbacks(
        classify_query=classify_query,
        build_query=build_query,
        retrieve=retrieve,
        generate_answer=generate_answer,
    )
