#!/bin/bash
set -e

ELASTIC_VERSION="8.8.0"
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ES_DIR="${BASE_DIR}/elasticsearch-${ELASTIC_VERSION}"
ES_TAR="elasticsearch-${ELASTIC_VERSION}-linux-x86_64.tar.gz"
ES_USER="elastic"

cd "$BASE_DIR"

echo "=== Elasticsearch 설치 시작 ==="

# 1) elastic 사용자 생성
if id "$ES_USER" &>/dev/null; then
    echo "사용자 '$ES_USER' 이미 존재함."
else
    echo "사용자 '$ES_USER' 생성 중..."
    sudo useradd -m -s /bin/bash "$ES_USER"
fi

# 2) ES 패키지 다운로드
if [ ! -f "${ES_TAR}" ]; then
    echo "Elasticsearch 패키지 다운로드 중..."
    wget https://artifacts.elastic.co/downloads/elasticsearch/${ES_TAR}
else
    echo "이미 다운로드된 파일이 있습니다: ${ES_TAR}"
fi

# 3) 압축 해제
if [ ! -d "${ES_DIR}" ]; then
    echo "압축 해제 중..."
    tar -xzvf "${ES_TAR}"
else
    echo "이미 압축 해제된 디렉터리가 존재합니다: ${ES_DIR}"
fi

# 4) 소유권 변경
echo "디렉토리 소유권을 ${ES_USER}:${ES_USER}로 변경..."
sudo chown -R "${ES_USER}:${ES_USER}" "${ES_DIR}/"

# 5) Nori 플러그인 설치
echo "Nori 플러그인 설치..."
sudo -u "${ES_USER}" "${ES_DIR}/bin/elasticsearch-plugin" install analysis-nori

# 6) elasticsearch.yml 생성 (없을 때만)
ES_CONFIG="${ES_DIR}/config/elasticsearch.yml"
if [ ! -f "$ES_CONFIG" ]; then
    echo "기본 elasticsearch.yml 생성 (보안 OFF, http://localhost:9200)..."
    cat > "$ES_CONFIG" << 'EOF'
cluster.name: skqa-cluster
node.name: skqa-node-1

path.data: data
path.logs: logs

network.host: 127.0.0.1
http.port: 9200

# 개발용: 보안/인증 비활성화
xpack.security.enabled: false
EOF
else
    echo "기존 elasticsearch.yml이 있어서 덮어쓰지 않았습니다: $ES_CONFIG"
    echo "보안 OFF로 쓰려면 xpack.security.enabled: false 인지 확인하세요."
fi

echo "=== Elasticsearch 설치 완료 ==="
echo "설치 경로: ${ES_DIR}"
echo "다음 명령으로 실행하세요:"
echo "  bash run_elasticsearch.sh"
