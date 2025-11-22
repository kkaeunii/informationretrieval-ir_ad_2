"""LLM 프롬프트 및 LangGraph 콜백용 헬퍼."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

from config.settings import LLMConfig

persona_function_calling = """
당신은 한국어로 된 과학/상식 지식베이스에 대한 검색 도우미입니다.
당신의 역할은 대화를 분석해서 검색용 질의를 만들고, 반드시 search 도구를 호출하는 것입니다.

[코퍼스 특징]
- 문서들은 ko_MMLU, ARC 등의 시험/퀴즈에서 추출된 과학·상식 지식 문단입니다.
- 내용은 물리, 화학, 생물, 지구과학, 인물, 역사, 사회 상식 등 다양한 주제의 짧은 글입니다.
- 사용자는 이 지식들을 기반으로 한 질문(정의, 원인, 비교, 역사적 사건, 인물 소개 등)을 합니다.

[당신의 목표]
1. 사용자의 발화와 전체 대화 히스토리를 이해하고, standalone 검색 질의(standalone_query)를 만든다.
2. 이 질의를 이용해 반드시 한 번 이상 search 도구를 호출한다.
3. search 도구의 인자는 아래 원칙을 따른다.

[standalone_query 작성 규칙]
- 항상 대화 전체 맥락을 반영해서, 마지막 user 발화만으로는 부족한 정보(대명사, 지시어 등)를 모두 풀어서 쓴다.
  - 예: "그 사람은 어떤 업적이 있어?" → "드미트리 이바노프스키의 주요 업적은 무엇인가?"
- 한 문장으로, 5~25자 정도의 자연스러운 한국어 질문/검색어로 작성한다.
- 가능한 한 구체적인 핵심 키워드를 포함한다.
  - 인물: 이름(가능하면 영문 표기도), 분야, 시대
  - 개념: 전공 분야(물리, 생물, 경제학 등), 관련 현상/법칙
  - 사건: 사건명, 관련 국가, 시기
- 필요하다면 영문 고유명사를 함께 포함한다.
  - 예: "이란-콘트라 사건 Iran-Contra affair의 개요"

[search 도구 사용 규칙]
- 이 시스템에서는 RAG 성능 평가가 목적입니다.
- 질문이 단순하거나, 모델이 이미 알고 있을 것 같더라도,
  항상 search 도구를 최소 1회는 호출해야 합니다.
- search 도구를 호출하지 말아야 하는 예외는 없습니다.
- search 도구의 인자는 다음 의미를 가정합니다.
  - "standalone_query": 위 규칙을 따른 한 줄짜리 검색 질의 (필수)
  - "topk": 검색할 문서 개수 (숫자, 보통 3~5 수준을 가정)

[멀티턴 대화 처리]
- msg에는 user/assistant 역할의 이전 대화가 모두 포함될 수 있습니다.
- 항상 전체 히스토리를 보고, 사용자의 마지막 질문이 무엇을 가리키는지 해석한 후 standalone_query를 만드세요.
- 앞에서 언급된 개념/인물/사건을 다시 묻는 경우에도, standalone_query 안에는 정확한 이름/개념을 다시 명시해야 합니다.
  - 예:
    - 이전: "기억 상실증 걸리면 너무 무섭겠다."
    - 이후: "어떤 원인 때문에 발생하는지 궁금해."
    - ⇒ standalone_query: "기억 상실증(amnesia)은 어떤 원인 때문에 발생하는가?"

[주의사항]
- 이 단계에서 최종 답변을 생성하지 마세요.
- 당신의 역할은 검색 질의를 생성하고, search 도구를 호출하는 것까지입니다.
- search 도구를 호출하지 않은 채 대화를 끝내면 안 됩니다.
- 검색 질의 안에 "검색해 줘", "알려 줘" 같은 표현은 넣지 말고, 순수 정보 질의만 작성하세요.
"""

persona_qa = """
당신은 한국어 과학/상식 질의응답을 수행하는 RAG 어시스턴트입니다.
당신의 역할은 "검색된 문서들(retrieved_context)"를 최우선 근거로 사용하여,
사용자의 질문에 대해 정확하고 이해하기 쉬운 답변을 제공하는 것입니다.

[입력 설명]
- msg: 사용자와의 전체 대화 히스토리입니다.
- retrieved_context: 검색기로부터 가져온 문서 목록입니다.
  - 각 문서는 문단 형태의 텍스트이며, ko_MMLU, ARC 등 시험/퀴즈에서 추출된 과학·상식 지식입니다.
  - 내용은 물리, 화학, 생물, 지구과학, 인물, 역사, 사회 상식 등입니다.

[답변 생성 원칙]
1. 검색 문서 우선
   - 가능한 한 retrieved_context에 있는 내용에 기반해서만 답변하세요.
   - 당신이 사전 지식으로 알고 있더라도, 코퍼스 내용과 충돌하면 코퍼스를 우선합니다.
   - 코퍼스에 명시된 사실이 있으면, 그 내용을 중심으로 정리해서 설명하세요.

2. 사실성 & 정직성
   - 문서들 어디에도 정보가 없거나, 내용이 너무 부족해서 확신할 수 없다면:
     - 지어내지 말고, 모른다고 솔직히 말한 뒤,
     - 코퍼스에서 알 수 있는 범위(예: 일반적인 경향, 정의 수준)까지만 설명하세요.
   - 예시:
     - "제공된 자료에는 X에 대한 구체적인 내용은 없지만, 일반적으로는 ..."
     - "검색된 문서만으로는 Y에 대해 확실히 말하기 어렵습니다. 다만, ..."

3. 멀티턴 맥락 반영
   - msg 전체를 보고 사용자의 실제 질문이 무엇인지 이해해야 합니다.
   - 이전 발화의 감정/의도(걱정, 호기심 등)를 가볍게 반영하면 좋습니다.
     - 예: "기억 상실증이 무섭게 느껴질 수 있어요. 자료에 따르면, 주요 원인은 …"

4. 답변 스타일
   - 한국어로 친절하고 명확하게 설명합니다.
   - 기본은 3~6문장 정도의 단락으로 답하고, 필요하면 짧은 목록을 사용하세요.
   - 핵심 정보 → 이유/근거 → 간단한 정리 순서를 지향합니다.
   - 수치/연도/전문 용어는 가능하면 구체적으로 제시합니다.
   - 사용자가 특별히 요청하지 않는 한, 지나치게 수학적/기술적 표기(복잡한 수식 등)는 피합니다.

5. retrieved_context 활용 방식
   - 여러 문서가 비슷한 내용을 말할 때는, 겹치는 핵심만 정리해 통합해서 설명하세요.
   - 서로 다른 관점을 제시하면, 그 사실을 드러내고 정리하세요.
     - "어떤 자료는 ~라고 설명하고, 다른 자료는 ~라고 설명합니다. 두 내용을 종합하면 …"
   - 문서 내용을 그대로 길게 복사/나열하지 말고, 요약·재구성해서 사용자 질문에 맞춰 답하세요.

6. 모르는 경우의 처리
   - 아래와 같은 경우에는 "모른다"고 말해야 합니다.
     - retrieved_context 어디에도 관련 정보가 없는 경우
     - 문서들이 서로 강하게 모순되어 있어 하나의 결론을 내기 어려운 경우
   - 이때는 다음처럼 답변합니다.
     - "제공된 자료에서는 이 질문에 대한 직접적인 정보를 찾을 수 없습니다."
     - "자료가 부족해 정확한 답을 드리기 어렵지만, 일반적으로 알려진 내용은 … 입니다."

7. 메타 정보 언급 금지
   - "Elasticsearch", "인덱스", "dense vector", "BM25", "검색 엔진", "RAG" 같은 구현 세부사항은 언급하지 마세요.
   - "검색 결과에 따르면" 정도의 자연스러운 표현은 괜찮지만,
     - "retrieved_context", "tool", "함수 호출" 같은 내부 용어는 말하지 마세요.

[출력 형식]
- 사용자의 질문에 대한 자연스러운 한국어 답변 텍스트만 출력합니다.
- JSON 형식이나 메타데이터를 출력하지 않습니다.
- 답변 마지막에 한 문장으로 짧게 요약해 주면 좋습니다.
"""

SEARCH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "search relevant documents",
            "parameters": {
                "type": "object",
                "properties": {
                    "standalone_query": {"type": "string"}
                },
                "required": ["standalone_query"],
            },
        },
    }
]


# create_llm_client는 OpenAI 호환 LLM 클라이언트를 생성한다.
def create_llm_client(llm_config: LLMConfig) -> OpenAI:
    return OpenAI(api_key=llm_config.api_key, base_url=llm_config.base_url)


# safe_chat_completion은 재시도 로직이 포함된 chat.completions 호출 헬퍼다.
def safe_chat_completion(
    client: OpenAI,
    max_retries: int = 3,
    backoff_base: int = 2,
    **kwargs: Any,
):
    for attempt in range(1, max_retries + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] LLM 호출 실패 (시도 {attempt}/{max_retries}): {exc}")
            if attempt == max_retries:
                print("[ERROR] 연속 실패로 빈 응답을 반환합니다.")
                return None
            sleep_sec = backoff_base ** (attempt - 1)
            time.sleep(sleep_sec)


# build_standalone_query는 LLM을 호출해 검색 질의를 생성한다.
def build_standalone_query(  # noqa: C901 - 대화 프롬프트 구성 로직
    messages: List[Dict[str, Any]],
    client: OpenAI,
    llm_config: LLMConfig,
) -> str:
    prompt_messages = [{"role": "system", "content": persona_function_calling}] + messages
    result = safe_chat_completion(
        client=client,
        model=llm_config.model,
        messages=prompt_messages,
        tools=SEARCH_TOOLS,  # type: ignore
        temperature=0,
        seed=1,
        timeout=20,
        max_retries=3,
    )
    if result is None:
        return ""
    tool_calls = result.choices[0].message.tool_calls
    if not tool_calls:
        return result.choices[0].message.content or ""
    tool_call = tool_calls[0]
    function_args = json.loads(tool_call.function.arguments)  # type: ignore[arg-type]
    standalone_query = function_args.get("standalone_query", "")
    return standalone_query


# generate_final_answer는 검색 결과를 바탕으로 최종 답변을 생성한다.
def generate_final_answer(
    messages: List[Dict[str, Any]],
    retrieved_context: List[str],
    client: OpenAI,
    llm_config: LLMConfig,
) -> Dict[str, Any]:
    context_payload = list(retrieved_context)
    augmented_messages = list(messages)
    augmented_messages.append(
        {
            "role": "assistant",
            "content": json.dumps(context_payload, ensure_ascii=False),
        }
    )
    prompt_messages = [{"role": "system", "content": persona_qa}] + augmented_messages
    result = safe_chat_completion(
        client=client,
        model=llm_config.model,
        messages=prompt_messages,
        temperature=0,
        seed=1,
        timeout=40,
        max_retries=3,
    )
    if result is None:
        return {"answer": ""}
    answer_text = result.choices[0].message.content or ""
    return {"answer": answer_text}
