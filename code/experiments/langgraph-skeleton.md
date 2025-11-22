# LangGraph 스켈레톤 공유

`code/pipelines/langgraph_pipeline.py`는 기존 `rag_with_elasticsearch.py` 흐름(standalone query → 검색 → 생성 → 평가)을 LangGraph(StateGraph) 기반으로 재구성하기 위한 최소 스켈레톤이다. 이 문서는 각 노드의 입·출력 정의와 확장 가이드를 간결하게 정리한다.

## 상태 정의

| 필드 | 설명 |
| --- | --- |
| `messages` | system/user/assistant 히스토리. 기존 `eval.jsonl`의 `msg`를 그대로 전달 |
| `standalone_query` | 프롬프트/에이전트가 생성한 검색 질의 |
| `retrieval_kwargs` | `size`, `alpha` 등 검색 튜닝 값 |
| `retrieved_context` | 검색 노드가 반환한 문서 배열(`content`, `docid`, `score`) |
| `topk_doc_ids` | 제출용 docid 리스트 |
| `references` | 제출 CSV에 포함할 근거(문단 + 스코어) |
| `answer` | 생성 모델 최종 답변 |
| `evaluation` | 선택적 사후 평가 정보(MAP 계산용 메타데이터 등) |
| `metadata` | 실행 파라미터(실험 이름, 시드 등) |
| `is_science_query` | 비과학 질의 처리 노드에서 판단한 bool 값 |

## 노드 시그니처 요약

| 노드 | 입력 | 출력 |
| --- | --- | --- |
| `classify_query` | `messages` | `is_science_query` 설정(미구현 시 True) |
| `build_query` | `messages`, `is_science_query` | `standalone_query` (비과학 시 빈 문자열) |
| `retrieve` | `standalone_query`, `retrieval_kwargs` | `retrieved_context`, `topk_doc_ids`, `references` |
| `generate` | `messages`, `retrieved_context` | `answer` 및 필요 시 `references`, `topk_doc_ids` 덮어쓰기 |
| `evaluate` | 전체 상태 | `evaluation` 사전(없으면 `{"status": "skipped"}`) |

```
START → classify_query → build_query → retrieve → generate → evaluate → END
```

## 확장 가이드

1. **새 노드 추가**: `code/pipelines/langgraph_pipeline.py`에 함수 + 주석을 추가하고 `build_rag_graph`에서 `graph.add_node` / `add_edge` 호출로 연결한다.
2. **의존성 주입**: 실 구현은 `RagGraphCallbacks`에 콜백을 주입해 관리한다. 예: standalone query 생성을 기존 `safe_chat_completion` 기반 함수로 래핑 후 `build_query`에 전달.
3. **비과학 질의 처리 레버리지**: `classify_query` 콜백에서 False 반환 시 `retrieve`/`generate` 노드가 자동으로 빈 결과를 유지하므로 다음 Task(2.x) 구현 시 상태 분기만 채우면 된다.
4. **팀 공유**: `graph.invoke(...)` 호출 시 사용한 파라미터/콜백 설명을 실험 로그(`docs/experiments/experiment-log.md`)에 남기면 재현성이 확보된다.

## 실행 예시

```
uv run python - <<'PY'
from pipelines.langgraph_pipeline import sample_usage
from pprint import pprint

result = sample_usage()
pprint(result)
PY
```

위 예시는 더미 콜백으로 그래프를 실제 실행해보는 방법이다. 팀원은 각 콜백을 실 구현으로 교체해 LangGraph 기반 파이프라인을 점진적으로 이식하면 된다.

## LangGraph 기반 실행 스크립트

- 경로: `code/scripts/rag_with_langgraph.py`
- 실행 예시: `uv run python code/scripts/rag_with_langgraph.py --skip-index --alpha 0.5 --topk 3`
- 주요 옵션:
  - `--documents`: 색인에 사용할 JSONL (기본 `code/data/documents.jsonl`)
  - `--eval`: 평가 JSONL 경로 (기본 `code/data/eval.jsonl`)
  - `--output`: 제출용 JSONL 저장 경로 (기본 `code/sample_submission_hybrid2.csv`)
  - `--skip-index`: 이미 Elasticsearch 인덱스가 존재하면 색인 단계를 생략

실제 실행 흐름은 다음과 같다.

1. `.env`에서 ES/LLM 설정 로딩 (`config/settings.py`).
2. 필요 시 문서 임베딩 → ES 인덱스 재생성 (`llm/embedding.py`, `retrieval/elasticsearch_utils.py`).
3. `pipelines/rag_callbacks.py`에서 LangGraph 콜백 생성 후 `pipelines/langgraph_pipeline.py`에 주입.
4. `graph.invoke` 결과를 반복적으로 호출해 `code/sample_submission_hybrid2.csv`를 생성.

## 비과학 질의 필터 개요

- 규칙 위치: `code/retrieval/non_science.py`
- 주요 패턴: 인사/감정 표현(안녕/반가, 기분이 좋아, 힘들다 등), 모델 신원/능력 질문(“너는 누구야?”, “너 뭘 잘해?”), 모델 상태·농담/심심 요청, 취향 추천(영화/노래/게임), 실시간 날씨, 감정 상담(사랑/연애/친구 고민) 등
- LangGraph 연동: `pipelines/rag_callbacks.py`의 `classify_query` 콜백에서 `is_science_query`를 호출 → False 시 `retrieve`/`generate` 노드가 빈 `topk`/`answer`를 유지
- 검증: 스크립트 실행 로그에 `[INFO] 처리 결과 - 과학 X개 / 비과학 Y개`가 출력되므로, 제출 전에 숫자와 `code/sample_submission_hybrid2.csv`의 빈 `topk` ID를 교차 확인

## 리팩토링 이후 사용 가이드

1. **환경 확인**: `.env`에 `ES_USERNAME/PASSWORD/CA_CERT`, `SOLAR_API_KEY`를 채우고 `uv sync`를 실행한다.
2. **인덱스 재사용 여부 결정**: 기존 인덱스를 그대로 쓰면 `--skip-index`, 새로 구축하려면 기본 옵션 그대로 실행한다.
3. **명령 실행**:
   ```bash
   uv run python code/scripts/rag_with_langgraph.py --skip-index --alpha 0.5 --topk 3
   ```
   - 로그에 비과학 건수가 17개(2025-11-21 기준)인지 확인
4. **CSV 검증**: `python - <<'PY' ...` 형태로 `topk`가 빈 eval ID가 로그와 일치하는지 확인하고, 제출용 CSV를 별도 파일명으로 백업한다.
5. **실험 로그 업데이트**: 점수/명령/비과학 건수 변화를 `docs/experiments/experiment-log.md`에 추가해 팀과 공유한다.

필요 시 이 문서를 계속 확장해 LangGraph 노드 추가 방법이나 팀별 실험 절차(예: sparse-only 비교, ICT/QG 파이프라인 연동)를 이어서 기록하면 된다.
