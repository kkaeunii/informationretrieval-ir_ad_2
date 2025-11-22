## LangGraph 기반 RAG 실행 가이드

### 1. 사전 준비
- `uv sync`로 Python 의존성을 설치합니다. (`pyproject.toml`, `code/requirements.txt` 기준)
- `code/.env`에 다음 항목을 채웁니다.
  - `ES_HOST`, `ES_USERNAME`, `ES_PASSWORD`, `ES_CA_CERT`
  - `SOLAR_API_KEY` 혹은 `OPENAI_API_KEY`
- 필요하면 `code/.env.example`을 참고하세요.

### 2. Elasticsearch 설치/관리
1. `bash code/install_elasticsearch.sh` – ES 바이너리 설치
2. `bash code/run_elasticsearch.sh` – 데몬 실행
3. `code/elasticsearch-8.8.0/config/elasticsearch.yml`에서 아래가 `false`인지 확인
   - `xpack.security.enabled`
   - `xpack.security.http.ssl.enabled`
   - `xpack.security.transport.ssl.enabled`
4. 종료 시 `bash code/stop_elasticsearch.sh`

### 3. LangGraph 파이프라인 실행
1. 새 인덱스를 만들고 싶으면 기본 옵션 그대로, 기존 인덱스를 재사용하려면 `--skip-index`
2. 실행 명령 예시:
   ```bash
   uv run python code/scripts/rag_with_langgraph.py --skip-index --alpha 0.5 --topk 3
   ```
3. 실행 로그에 `[INFO] 처리 결과 - 과학 질의 XXX개 / 비과학 YYY개`가 출력됩니다. 현재 규칙(2025-11-21) 기준 YYY는 17입니다.

### 4. 제출 파일 검증
1. 결과 파일: `code/sample_submission_hybrid2.csv`
2. `python` 스니펫으로 `topk`가 빈 eval ID를 확인하여 로그와 일치하는지 검증합니다.
3. 제출 전에 `cp code/sample_submission_hybrid2.csv code/sample_submission_hybrid2_YYYYMMDD.csv`처럼 백업합니다.
4. 리더보드 제출 후 점수/실행 옵션/비과학 건수를 `docs/experiments/experiment-log.md`에 기록합니다.

### 5. 참고 문서
- LangGraph/비과학 필터 요약: `experiments/langgraph-skeleton.md`
- 실험 히스토리 및 제출 점수: `experiments/experiment-log.md`
