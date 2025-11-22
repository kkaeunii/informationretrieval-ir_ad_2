"""환경 설정 로더 모듈."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


# Elasticsearch 접속 정보를 표현하는 데이터 클래스.
@dataclass
class ElasticsearchConfig:
    host: str
    username: Optional[str]
    password: Optional[str]
    ca_cert: Optional[str]
    index_name: str


# LLM 호출에 필요한 설정을 표현하는 데이터 클래스.
@dataclass
class LLMConfig:
    api_key: str
    model: str
    base_url: str


# 전체 실행에 필요한 설정을 하나로 모은 데이터 클래스.
@dataclass
class Settings:
    es: ElasticsearchConfig
    llm: LLMConfig


# load_settings는 .env에서 환경 변수를 읽고 Settings 객체를 반환한다.
def load_settings(default_model: str = "solar-pro2") -> Settings:
    load_dotenv()

    es_host = os.getenv("ES_HOST", "http://localhost:9200")
    es_username = os.getenv("ES_USERNAME")
    es_password = os.getenv("ES_PASSWORD")
    es_ca_cert = os.getenv("ES_CA_CERT")
    es_index_name = os.getenv("ES_INDEX", "test")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    solar_api_key = os.getenv("SOLAR_API_KEY")
    llm_api_key = openai_api_key or solar_api_key
    if llm_api_key is None:
        raise RuntimeError("OPENAI_API_KEY 또는 SOLAR_API_KEY 중 하나는 반드시 설정해야 합니다.")

    llm_base_url = os.getenv("LLM_BASE_URL", "https://api.upstage.ai/v1")
    llm_model = os.getenv("LLM_MODEL", default_model)

    es_config = ElasticsearchConfig(
        host=es_host,
        username=es_username,
        password=es_password,
        ca_cert=es_ca_cert,
        index_name=es_index_name,
    )
    llm_config = LLMConfig(api_key=llm_api_key, model=llm_model, base_url=llm_base_url)
    return Settings(es=es_config, llm=llm_config)
