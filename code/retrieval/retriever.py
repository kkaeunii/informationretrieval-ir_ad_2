"""검색 함수 모듈."""

from __future__ import annotations

from typing import Any, Dict

from elasticsearch import Elasticsearch

from llm.embedding import EmbeddingService


# sparse_retrieve는 BM25 검색 결과를 반환한다.
def sparse_retrieve(
    client: Elasticsearch,
    index_name: str,
    query_text: str,
    size: int,
) -> Dict[str, Any]:
    query = {"match": {"content": {"query": query_text}}}
    return client.search(index=index_name, query=query, size=size, sort="_score")


# dense_retrieve는 KNN 검색을 수행한다.
def dense_retrieve(
    client: Elasticsearch,
    index_name: str,
    query_text: str,
    size: int,
    embedder: EmbeddingService,
) -> Dict[str, Any]:
    query_embedding = embedder.encode([query_text])[0]
    knn_query = {
        "field": "embeddings",
        "query_vector": query_embedding.tolist(),
        "k": size,
        "num_candidates": 100,
    }
    return client.search(index=index_name, knn=knn_query)


# hybrid_retrieve는 sparse/dense 결과를 정규화 후 병합한다.
def hybrid_retrieve(
    client: Elasticsearch,
    index_name: str,
    query_text: str,
    size: int,
    embedder: EmbeddingService,
    alpha: float = 0.5,
):
    sparse = sparse_retrieve(client, index_name, query_text, size)
    dense = dense_retrieve(client, index_name, query_text, size, embedder)

    combined: Dict[str, Dict[str, Any]] = {}

    def normalized_and_add(results: Dict[str, Any], weight: float):
        hits = results.get("hits", {}).get("hits", [])
        if not hits:
            return
        max_score = max(hit.get("_score", 0.0) for hit in hits) or 1.0
        for hit in hits:
            source = hit.get("_source", {})
            docid = source.get("docid")
            if docid is None:
                continue
            norm_score = (hit.get("_score", 0.0) / max_score) * weight
            combined.setdefault(docid, {"_source": source, "_score": 0.0})
            combined[docid]["_score"] += norm_score

    normalized_and_add(sparse, alpha)
    normalized_and_add(dense, 1 - alpha)

    merged_hits = sorted(
        [
            {"_source": payload["_source"], "_score": payload["_score"]}
            for payload in combined.values()
        ],
        key=lambda entry: entry["_score"],
        reverse=True,
    )[:size]

    return {"hits": {"hits": merged_hits}}
