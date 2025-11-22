"""기존 경로 호환을 위한 LangGraph 실행 래퍼."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.rag_with_langgraph import main


# main은 code/scripts/rag_with_langgraph.py의 main을 그대로 재사용한다.
if __name__ == "__main__":
    main()
