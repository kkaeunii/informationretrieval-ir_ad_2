#!/bin/bash
set -e

ELASTIC_VERSION="8.8.0"
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ES_DIR="${BASE_DIR}/elasticsearch-${ELASTIC_VERSION}"
ES_USER="elastic"

echo "=== Elasticsearch 실행 (보안/SSL 비활성화, http://localhost:9200) ==="
echo "[DEBUG] 현재 쉘 사용자: $(whoami)"

if [ ! -d "$ES_DIR" ]; then
    echo "!! ES_DIR(${ES_DIR}) 디렉터리가 없습니다. 경로를 확인해 주세요."
    exit 1
fi

echo "[권한] ES 디렉터리 소유자 정리..."
sudo chown -R "$ES_USER:$ES_USER" "$ES_DIR"

echo "[ES] Elasticsearch 백그라운드 실행..."
# sudo -u "${ES_DIR}/bin/elasticsearch" -d > "${ES_DIR}/logs/run_stdout.log" 2>&1 &
# "${ES_DIR}/bin/elasticsearch" -d > "${ES_DIR}/logs/run_stdout.log" 2>&1 &
sudo -u "$ES_USER" bash -c "echo \"[DEBUG] elastic 내부 사용자: \$(whoami)\"; \"$ES_DIR/bin/elasticsearch\" -d" \
  > "${ES_DIR}/logs/run_stdout.log" 2>&1 &
  
echo "[ES] 부팅 대기 (30초)..."
sleep 30

echo "[ES] 상태 체크: curl http://localhost:9200"
curl -s "http://localhost:9200" || echo "curl 접속 실패: ${ES_DIR}/logs/elasticsearch.log 를 확인하세요."
