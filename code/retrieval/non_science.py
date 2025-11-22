"""비과학(non-science) 질의 판별 규칙."""

from __future__ import annotations

import re
from typing import Dict, List


# 감정/잡담 표현을 담은 정규식 목록. eval.jsonl에서 추출한 사례를 기반으로 구성했다.
NON_SCIENCE_PATTERNS = [
    re.compile(pattern)
    for pattern in [
        r"안녕|반가",  # 인사/잡담
        r"니가\s*대답을\s*잘해줘",  # 모델 칭찬/대화 언급
        r"(기분이\s*좋아|신나|즐거웠)",  # 감정 표현
        r"힘들|힘든|힘드|우울",  # 고민/상담 요청
        r"그만\s*얘기",  # 대화 종료 표현
        r"너는\s*(누구|어디|무슨)\s*(야|니|니\?|출신|소속)",  # 챗봇 신원 질문 확장
        r"너\s*(뭘|무슨\s*일을)\s*잘해",  # 모델 능력 질문
        r"너\s*정말\s*똑똑",  # 모델 칭찬/감탄
        r"(오늘|지금)\s*(기분|상태)\s*(어때|어떻|궁금)",  # 모델 상태 질문
        r"(농담|재밌는\s*이야기|웃긴\s*얘기).*해줘",  # 농담 요청
        r"(심심|지루).*해서",  # 심심함 표현
        r"(영화|드라마|노래|게임).*추천해줘",  # 취향 추천
        r"(오늘|내일).*(날씨|기온).*알려줘",  # 실시간 날씨 질의
        r"(사랑|연애|이별|친구).*상담",  # 감정 상담 요청
    ]
]


# is_science_query는 user 메시지를 합쳐 비과학 패턴에 해당하는지 판별한다.
def is_science_query(messages: List[Dict[str, str]]) -> bool:
    user_text = " ".join(msg.get("content", "") for msg in messages if msg.get("role") == "user")
    normalized = user_text.strip()
    if not normalized:
        return True
    for pattern in NON_SCIENCE_PATTERNS:
        if pattern.search(normalized):
            return False
    return True
