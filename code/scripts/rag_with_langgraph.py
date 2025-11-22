"""LangGraph 기반 RAG 실행 스크립트."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

from config.settings import load_settings
from llm.embedding import EmbeddingService
from pipelines.langgraph_pipeline import build_rag_graph
from pipelines.rag_callbacks import RagDependencies, build_callbacks
from retrieval.elasticsearch_utils import (
    bulk_index_documents,
    create_client,
    fetch_all_documents,
    recreate_index,
)


# build_arg_parser는 CLI 인자를 정의한다.
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LangGraph 기반 RAG 실행")
    parser.add_argument("--eval", dest="eval_path", default="code/data/eval.jsonl")
    parser.add_argument("--output", dest="output_path", default="code/sample_submission_hybrid2.csv")
    parser.add_argument("--documents", dest="documents_path", default="code/data/documents.jsonl")
    parser.add_argument("--topk", dest="topk", type=int, default=3)
    parser.add_argument("--alpha", dest="alpha", type=float, default=0.5)
    parser.add_argument("--skip-index", dest="skip_index", action="store_true")
    return parser


# prepare_index는 문서를 읽어 임베딩을 계산하고 ES 인덱스를 재구성한다.
def prepare_index(
    embedder: EmbeddingService,
    documents_path: str,
    es_client,
    index_name: str,
):
    docs = fetch_all_documents(documents_path)
    embeddings = embedder.encode_documents(docs)
    for doc, embedding in zip(docs, embeddings):
        doc["embeddings"] = embedding.tolist()

    settings = {
        "analysis": {
            "analyzer": {
                "nori": {
                    "type": "custom",
                    "tokenizer": "nori_tokenizer",
                    "decompound_mode": "mixed",
                    "filter": ["nori_posfilter"],
                }
            },
            "filter": {
                "nori_posfilter": {
                    "type": "nori_part_of_speech",
                    "stoptags": ["E", "J", "SC", "SE", "SF", "VCN", "VCP", "VX"],
                }
            },
        }
    }
    mappings = {
        "properties": {
            "content": {"type": "text", "analyzer": "nori"},
            "embeddings": {
                "type": "dense_vector",
                "dims": 768,
                "index": True,
                "similarity": "l2_norm",
            },
        }
    }

    recreate_index(es_client, index_name=index_name, settings=settings, mappings=mappings)
    bulk_index_documents(es_client, index_name=index_name, docs=docs)


# run_graph_on_eval은 LangGraph 파이프라인을 eval.jsonl에 적용한다.
def run_graph_on_eval(graph, eval_path: str, output_path: str, retrieval_kwargs: Dict):
    stats = {"science": 0, "non_science": 0}
    with open(eval_path, encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            record = json.loads(line)
            messages = record["msg"]
            state = graph.invoke(
                {
                    "messages": messages,
                    "retrieval_kwargs": retrieval_kwargs,
                }
            )
            topk_docs = state.get("topk_doc_ids", [])
            output = {
                "eval_id": record["eval_id"],
                "standalone_query": state.get("standalone_query", ""),
                "topk": topk_docs,
                "answer": state.get("answer", ""),
                "references": state.get("references", []),
            }
            outfile.write(json.dumps(output, ensure_ascii=False) + "\n")
            if state.get("is_science_query", True):
                stats["science"] += 1
            else:
                stats["non_science"] += 1
    return stats


# main은 전체 LangGraph 실행 흐름을 담당한다.
def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    settings = load_settings()
    es_client = create_client(settings.es)
    embedder = EmbeddingService()

    if not args.skip_index:
        prepare_index(embedder, args.documents_path, es_client, settings.es.index_name)

    deps = RagDependencies(settings=settings, es_client=es_client, embedder=embedder)
    callbacks = build_callbacks(deps)
    graph = build_rag_graph(callbacks)

    retrieval_kwargs = {"size": args.topk, "alpha": args.alpha}
    stats = run_graph_on_eval(
        graph,
        eval_path=args.eval_path,
        output_path=args.output_path,
        retrieval_kwargs=retrieval_kwargs,
    )
    print(
        f"[INFO] 처리 결과 - 과학 질의: {stats['science']}개, 비과학 질의: {stats['non_science']}개"
    )


if __name__ == "__main__":
    main()
