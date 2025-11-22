#!/bin/bash

echo "=== Elasticsearch 종료 시작 ==="

# 프로세스 찾기
PIDS=$(pgrep -f "org.elasticsearch.bootstrap.Elasticsearch")
LOG_FILE=$(find ./ -type f -path "*logs/elasticsearch.log" 2>/dev/null | head -n 1)

if [ -z "$PIDS" ]; then
    echo "Elasticsearch 프로세스가 실행 중이 아닙니다."
else
    echo "찾은 Elasticsearch 프로세스 ID: $PIDS"
    echo "정상 종료(SIGTERM) 시도..."
    pkill -f "org.elasticsearch.bootstrap.Elasticsearch"

    # 대기
    sleep 10

    # 잔여 프로세스 확인
    PIDS_STILL=$(pgrep -f "org.elasticsearch.bootstrap.Elasticsearch")

    if [ -z "$PIDS_STILL" ]; then
        echo "정상 종료 완료!"
    else
        echo "정상 종료되지 않아 강제 종료(SIGKILL) 수행..."
        pkill -9 -f "org.elasticsearch.bootstrap.Elasticsearch"
        echo "강제 종료 완료."
    fi
fi

echo
echo "=== 최근 로그 출력 ==="

# 로그가 있는 경우라면 출력
if [ -n "$LOG_FILE" ]; then
    echo "로그 파일: $LOG_FILE"
    echo "------------------------------"
    tail -n 50 "$LOG_FILE"
    echo "------------------------------"
else
    echo "로그 파일을 찾지 못했습니다."
fi
