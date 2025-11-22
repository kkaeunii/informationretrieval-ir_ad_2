"""SentenceTransformer 기반 임베딩 유틸."""

from __future__ import annotations

from typing import Dict, List

from sentence_transformers import SentenceTransformer


# EmbeddingService는 SentenceTransformer 모델을 로드하고 배치 인코딩을 제공한다.
class EmbeddingService:
    def __init__(self, model_name: str = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"):
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    # encode는 입력 문장 배열을 임베딩 목록으로 변환한다.
    def encode(self, sentences: List[str]):
        return self._model.encode(sentences)

    # encode_documents는 documents 리스트를 받아 content 필드를 임베딩한다.
    def encode_documents(self, docs: List[Dict[str, str]], batch_size: int = 100):
        batch_embeddings: List[List[float]] = []
        for idx in range(0, len(docs), batch_size):
            batch = docs[idx : idx + batch_size]
            contents = [doc["content"] for doc in batch]
            embeddings = self.encode(contents)
            batch_embeddings.extend(embeddings)
        return batch_embeddings
