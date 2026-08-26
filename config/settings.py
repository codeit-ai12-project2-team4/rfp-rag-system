"""
API 키, 파이프라인 공용 파라미터 등 '환경'에 관한 값만 모아두는 곳.

API 키는 절대 코드에 하드코딩 하지 말아주세요..

"""

import os
from pathlib import Path

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY가 설정되지 않았습니다. .env 파일에 OPENAI_API_KEY=... 를 추가하세요."
    )

# 청킹/임베딩 후, 수정사항 생기면 여기만 수정하면 됩니당
MAX_CONTEXT_CHARS = 6000
DEFAULT_TOP_K = 5


# 이 파일이 config/settings.py 이므로 두 단계 위가 프로젝트 폴더
ROOT = Path(__file__).resolve().parents[1]

# ── 입력 ────────────────────────────────────────────────────────────────
DATA = ROOT / "data"
RAW = DATA / "raw"  # 원본 hwp / pdf
METADATA = DATA / "metadata"
META_CSV = METADATA / "data_list.csv"  # 공고 메타데이터
PROCESSED = DATA / "processed"
DOCUMENTS_JSONL = PROCESSED / "documents.jsonl"  # 전처리 완료본
EVAL_QA = DATA / "eval_qa.json"  # 평가용 정답 데이터셋

# ── 산출물 ──────────────────────────────────────────────────────────────
OUTPUTS = ROOT / "outputs"
VECTORSTORE = OUTPUTS / "vectorstore"  # FAISS 인덱스
CHUNKS = OUTPUTS / "chunks"  # 청크 jsonl
REPORTS = OUTPUTS / "reports"  # 전처리 경고/이슈 로그
EVAL_RESULTS = OUTPUTS / "eval_results"  # 평가 지표 결과

PREPROCESSING_REPORT = REPORTS / "preprocessing_report.json"


def make_dirs():
    for folder in (METADATA, PROCESSED, VECTORSTORE, CHUNKS, REPORTS, EVAL_RESULTS):
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
