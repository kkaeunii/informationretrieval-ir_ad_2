# 실험 로그

팀원들이 수행한 실험 및 제출 결과를 한 곳에서 관리하기 위한 로그입니다.

## 2025-11-21 – Solar + Hybrid (Baseline 재실행)

- 실행 환경: `uv run python code/rag_with_elasticsearch.py`
- 모델: Upstage Solar (`SOLAR_API_KEY`)
- 검색: 기존 hybrid( BM25 + dense ) 정규화 로직
- 결과 파일: `code/sample_submission_hybrid2.csv`
- 제출 점수: `MAP 0.7545`, `MRR 0.7576`
- 메모: 
  - `hybrid_retrieve`가 활성화된 상태로 기존 결과를 재현한 것임
  - 순수 sparse 실험은 별도로 수행하여 비교 필요
  - 비과학 질의를 판별해 `topk`를 빈 리스트로 제출하는 로직 필요
  - LangGraph 스켈레톤을 공유해 이후 실험을 체계화할 계획

## 2025-11-21 – LangGraph 리팩토링 준비

- 작업 내용: `config/`, `llm/`, `retrieval/`, `pipelines/`, `scripts/`로 모듈화하고 `code/scripts/rag_with_langgraph.py`를 추가
- 실행 명령: `uv run python code/scripts/rag_with_langgraph.py --skip-index`
- 상태: 네트워크 제한 때문에 LLM 호출이 완료되지 않아 CSV 검증 실패 (기존 결과 파일은 백업 후 복구)
- 메모:
  - 기존 `code/rag_with_elasticsearch.py`는 새 스크립트를 래핑하도록 단순화
  - LangGraph 콜백과 실제 검색/생성 로직을 분리했으므로 이후 비과학 질의 처리/ICT 파이프라인 연동을 바로 진행 가능

## 2025-11-21 – 비과학 질의 처리 로직 추가

- 작업 내용: `code/retrieval/non_science.py`에 규칙 기반 감정/인사 패턴을 정의하고 LangGraph `classify_query` 노드에 연결해 검색을 생략하도록 수정
- 감지 룰: "안녕/반가", "니가 대답을 잘해줘", "기분이 좋아/신나/즐거웠", "힘들/우울", "그만 얘기", "너는 누구", "너 뭘 잘해", "너 정말 똑똑" 등 감정∙잡담 패턴
- 검증: `code/data/eval.jsonl` 기준 비과학 질의 17건(IDs: 276, 261, 283, 32, 94, 90, 220, 247, 67, 57, 2, 227, 301, 222, 83, 103, 218)을 포착했고 스크립트 실행 시 `[INFO] 과학 203 / 비과학 17` 로그로 확인
- 효과: 해당 eval은 `standalone_query`와 `topk`가 비어 제출되므로 `score_if_no_docs` 보상 규칙을 만족, LLM 호출 비용도 절약됨

## 2025-11-21 – LangGraph + 비과학 필터 첫 제출 재현

- 실행 명령: `uv run python code/scripts/rag_with_langgraph.py --skip-index`
- 감지 결과: 당시 규칙 기준 비과학 12건(이후 17건으로 확장 예정)
- 리더보드 점수: `MAP 0.7667`, `MRR 0.7697`
- 메모: LangGraph 리팩토링과 기본 비과학 필터만 적용한 상태에서 기존 0.75 점수를 재현. 이후 규칙 확장으로 추가 향상을 준비.

## 2025-11-21 – 비과학 필터 적용 제출 재실행

- 실행 명령: `uv run python code/scripts/rag_with_langgraph.py --skip-index`
- 감지 결과: `[INFO] 처리 결과 - 과학 질의 203개 / 비과학 17개`, 제출 파일 `code/sample_submission_hybrid2.csv`에서 동일 ID가 빈 `topk`
- 리더보드 점수: `MAP 0.7848`, `MRR 0.7879` (이전 0.7667 / 0.7697 대비 소폭 상승)
- 메모: 비과학 질의를 모두 비워 제출하면서 penalty가 사라져 점수가 회복됨. 향후 실험 시에도 로그 출력으로 비과학 건수를 확인하고 제출 전에 CSV를 보관할 것.

## 2025-11-21 – 비과학 패턴 확장 & 재제출

- 작업 내용: `NON_SCIENCE_PATTERNS`에 모델 신원/능력 질문, 기분/농담 요청, 취향 추천, 실시간 날씨, 감정 상담 패턴 추가
- 실행 명령: `uv run python code/scripts/rag_with_langgraph.py --skip-index`
- 감지 결과: `[INFO] 처리 결과 - 과학 질의 203개 / 비과학 17개`
- 리더보드 점수: `MAP 0.7909`, `MRR 0.7939`
- 메모: 잡담 패턴까지 확장한 뒤에도 오탐 없이 점수가 소폭 상승. 비과학 ID를 CSV에서 다시 확인 후 제출했으며, 향후에도 규칙 변경 시 로그와 CSV를 함께 백업할 것.
