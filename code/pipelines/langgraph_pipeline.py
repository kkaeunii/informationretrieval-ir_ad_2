"""LangGraph 기반 RAG 파이프라인 스켈레톤."""

from __future__ import annotations

import json

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, START, END


# RagPipelineState는 LangGraph 노드 간 공유되는 상태 구조를 정의한다.
class RagPipelineState(TypedDict, total=False):
    messages: List[Dict[str, Any]]
    standalone_query: str
    retrieval_kwargs: Dict[str, Any]
    retrieved_context: List[str]
    topk_doc_ids: List[str]
    references: List[Dict[str, Any]]
    answer: str
    evaluation: Dict[str, Any]
    metadata: Dict[str, Any]
    is_science_query: bool


# RagGraphCallbacks는 LangGraph 각 노드에 필요한 외부 의존성을 모듈화한다.
@dataclass
class RagGraphCallbacks:
    classify_query: Optional[Callable[[List[Dict[str, Any]]], bool]] = None
    build_query: Optional[Callable[[List[Dict[str, Any]]], str]] = None
    retrieve: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None
    generate_answer: Optional[
        Callable[[List[Dict[str, Any]], List[str]], Dict[str, Any]]
    ] = None
    evaluate: Optional[Callable[[RagPipelineState], Dict[str, Any]]] = None


# _classify_node는 비과학 질의 여부를 판단하여 상태에 기록한다.
def _classify_node(state: RagPipelineState, callbacks: RagGraphCallbacks) -> RagPipelineState:
    messages = state.get("messages", [])
    if callbacks.classify_query is None:
        state["is_science_query"] = True
        return state
    state["is_science_query"] = callbacks.classify_query(messages)
    return state


# _build_query_node는 standalone query를 생성하고 상태에 저장한다.
def _build_query_node(state: RagPipelineState, callbacks: RagGraphCallbacks) -> RagPipelineState:
    if not state.get("is_science_query", True):
        state["standalone_query"] = ""
        return state
    if callbacks.build_query is None:
        raise RuntimeError("build_query callback must be provided")
    state["standalone_query"] = callbacks.build_query(state.get("messages", []))
    return state


# _retrieve_node는 검색 노드를 나타내며 LangGraph 상태에 검색 결과를 기록한다.
def _retrieve_node(state: RagPipelineState, callbacks: RagGraphCallbacks) -> RagPipelineState:
    if not state.get("is_science_query", True):
        state["retrieved_context"] = []
        state["topk_doc_ids"] = []
        state["references"] = []
        return state
    if callbacks.retrieve is None:
        raise RuntimeError("retrieve callback must be provided")
    retrieval_kwargs = state.get("retrieval_kwargs", {"size": 3})
    search_result = callbacks.retrieve(
        state.get("standalone_query", ""),
        retrieval_kwargs,
    )
    hits = search_result.get("hits", {}).get("hits", [])
    retrieved_context = []
    topk_doc_ids = []
    references = []
    for hit in hits:
        source = hit.get("_source", {})
        content = source.get("content")
        docid = source.get("docid")
        if content:
            retrieved_context.append(content)
        if docid:
            topk_doc_ids.append(docid)
        references.append({
            "score": hit.get("_score", 0.0),
            "content": content,
        })
    state["retrieved_context"] = retrieved_context
    state["topk_doc_ids"] = topk_doc_ids
    state["references"] = references
    return state


# _generate_node는 생성 모델 호출을 수행하고 답변 및 메타데이터를 채운다.
def _generate_node(state: RagPipelineState, callbacks: RagGraphCallbacks) -> RagPipelineState:
    if not state.get("is_science_query", True):
        state["answer"] = ""
        state["references"] = []
        return state
    if callbacks.generate_answer is None:
        raise RuntimeError("generate_answer callback must be provided")
    generation = callbacks.generate_answer(
        state.get("messages", []),
        state.get("retrieved_context", []),
    )
    state["answer"] = generation.get("answer", "")
    references = generation.get("references")
    if references is not None:
        state["references"] = references
    topk_override = generation.get("topk")
    if topk_override is not None:
        state["topk_doc_ids"] = topk_override
    return state


# _evaluate_node는 선택적으로 평가 정보를 채운다.
def _evaluate_node(state: RagPipelineState, callbacks: RagGraphCallbacks) -> RagPipelineState:
    if callbacks.evaluate is None:
        state.setdefault("evaluation", {"status": "skipped"})
        return state
    state["evaluation"] = callbacks.evaluate(state)
    return state


# build_rag_graph는 LangGraph StateGraph를 생성해 노드 연결을 완성한다.
def build_rag_graph(callbacks: RagGraphCallbacks):
    graph = StateGraph(RagPipelineState)

    graph.add_node("classify_query", lambda state: _classify_node(state, callbacks))
    graph.add_node("build_query", lambda state: _build_query_node(state, callbacks))
    graph.add_node("retrieve", lambda state: _retrieve_node(state, callbacks))
    graph.add_node("generate", lambda state: _generate_node(state, callbacks))
    graph.add_node("evaluate", lambda state: _evaluate_node(state, callbacks))

    graph.add_edge(START, "classify_query")
    graph.add_edge("classify_query", "build_query")
    graph.add_edge("build_query", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "evaluate")
    graph.add_edge("evaluate", END)

    return graph.compile()


# sample_usage는 팀원이 LangGraph 스켈레톤을 빠르게 실험하도록 돕는 헬퍼다.
def sample_usage():  # pragma: no cover
    dummy_callbacks = RagGraphCallbacks(
        classify_query=lambda messages: "과학" in messages[-1]["content"],
        build_query=lambda messages: messages[-1]["content"],
        retrieve=lambda query, _: {
            "hits": {
                "hits": [
                    {
                        "_source": {"docid": "doc-1", "content": f"doc for {query}"},
                        "_score": 1.0,
                    }
                ]
            }
        },
        generate_answer=lambda messages, docs: {
            "answer": f"답변: {docs[0]}" if docs else "",
        },
    )

    graph = build_rag_graph(dummy_callbacks)
    initial_state: RagPipelineState = {
        "messages": [{"role": "user", "content": "과학 질문"}],
        "retrieval_kwargs": {"size": 3},
    }
    result = graph.invoke(initial_state)
    return result


if __name__ == "__main__":  # pragma: no cover
    final_state = sample_usage()
    print(json.dumps(final_state, ensure_ascii=False, indent=2))
