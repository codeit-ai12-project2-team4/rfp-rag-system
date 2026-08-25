"""
API 키, 파이프라인 공용 파라미터 등 '환경'에 관한 값만 모아두는 곳.

API 키는 절대 코드에 하드코딩 하지 말아주세요..

"""

import os

from dotenv import load_dotenv

load_dotenv()  # 프로젝트 루트의 .env 파일을 읽어 환경변수로 등록

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY가 설정되지 않았습니다. .env 파일에 OPENAI_API_KEY=... 를 추가하세요."
    )

# 청킹/임베딩 후, 수정사항 생기면 여기만 수정하면 됩니당
MAX_CONTEXT_CHARS = 6000
DEFAULT_TOP_K = 5
