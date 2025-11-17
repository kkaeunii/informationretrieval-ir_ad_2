## Baseline code를 실행하기 위해 2가지 작업이 필요합니다.

1. 실행방법
- install_elasticsearch.sh 실행
- elasticsearch-.8.8.0/config/elasticsearch.yml에서

    xpack.security.enabled: false
  
    xpack.security.http.ssl.enabled: false
  
    xpack.security.transport.ssl.enabled: false 확인
- run_elasticsearch.sh 실행
- rag_with_elasticsearch.py 실행
- stop_elasticsearch.sh를 통해 종료

2. OpenAI API를 사용하기 위해서는 API 키가 필요
- .env.example 확인

