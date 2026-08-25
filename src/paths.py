"""경로 모음. 코드 어디에도 경로를 직접 쓰지 않는다.

.env 파일이 있으면 읽어 환경변수로 올린다 (OPENAI_API_KEY 등).
"""

import os
from pathlib import Path

# 이 파일이 src/bidmate/paths.py 이므로 두 단계 위가 프로젝트 폴더
ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "files"  # 원본 hwp / pdf
META_CSV = DATA / "data_list.csv"  # 공고 메타데이터
INTERIM = DATA / "interim"  # 추출한 본문
PROCESSED = DATA / "processed"  # 청크, 카드
INDEX = DATA / "index"  # FAISS 인덱스
EVALSETS = ROOT / "evalsets"  # 평가용 질문 세트
NOTEBOOKS = ROOT / "notebooks"

DOCUMENTS_JSONL = INTERIM / "documents.jsonl"
EXTRACTION_REPORT = INTERIM / "extraction_report.csv"


def make_dirs():
    for folder in (INTERIM, PROCESSED, INDEX, EVALSETS):
        folder.mkdir(parents=True, exist_ok=True)


def load_env():
    """.env 파일을 읽어 환경변수로 올린다. python-dotenv 가 없어도 동작한다."""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


load_env()
