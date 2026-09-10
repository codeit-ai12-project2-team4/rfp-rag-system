"""
API 키, 파이프라인 공용 파라미터 등 '환경'에 관한 값만 모아두는 곳.

API 키는 절대 코드에 하드코딩 하지 말아주세요..

"""

import os
import warnings
from enum import StrEnum
from pathlib import Path

# 이 파일이 config/settings.py 이므로 두 단계 위가 프로젝트 폴더
ROOT = Path(__file__).resolve().parents[1]


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
        value = value.strip()
        # 따옴표로 감싼 값은 안쪽을 그대로 쓴다. 임베딩 접두어처럼 **뒤 공백이
        # 의미를 갖는** 값이 있어서, 무조건 strip 하면 조용히 틀린 값이 된다.
        #     EMBED_QUERY_PREFIX="query: "   ← 따옴표가 있어야 공백이 산다
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


load_env()  # 키를 읽기 전에 .env 부터 올린다

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    # 여기서 raise 하면 OpenAI 를 안 쓰는 작업(BM25·TEI·전처리)까지 import 가 막힌다.
    # 실제로 부르는 쪽(models/llm.py)에서 없으면 그때 터진다.
    warnings.warn(
        "OPENAI_API_KEY 가 없습니다. .env 에 OPENAI_API_KEY=... 를 넣으세요. "
        "OpenAI 를 안 쓰는 작업은 그대로 돌아갑니다.",
        stacklevel=2,
    )

# 청킹/임베딩 후, 수정사항 생기면 여기만 수정하면 됩니당
MAX_CONTEXT_CHARS = 6000
DEFAULT_TOP_K = 5

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
LANCEDB = OUTPUTS / "lancedb"  # LanceDB 테이블 (FAISS 와 A/B 비교용)
CHUNKS = OUTPUTS / "chunks"  # 청크 jsonl
REPORTS = OUTPUTS / "reports"  # 전처리 경고/이슈 로그
EVAL_RESULTS = OUTPUTS / "eval_results"  # 평가 지표 결과

PREPROCESSING_REPORT = REPORTS / "preprocessing_report.json"


class Provider(StrEnum):
    """모델을 어디서 부르는지. `ModelConfig.provider` 와 같은 축이다.

    `StrEnum` 이라 문자열과 그대로 비교된다. 그래서 `generation.py` 의
    `_PROVIDER_RUNNERS.get(cfg.provider)` 같은 기존 코드를 한 줄도 안 고쳐도 된다.

        Provider.OPENAI == "openai"        # True
        Provider("  OpenAI ")              # Provider.OPENAI  (대소문자·공백 관용)

    **여기 없는 건 안 쓴다.** vLLM 은 OpenAI 규격과 호환이라 `OPENAI` 에
    `base_url` 만 바꿔 붙인다 (`models/llm.py` 의 `OpenAILLM` 이 그렇게 한다).
    Anthropic·Google·Groq 등은 키도 계획도 없다 — 필요해지면 한 줄 늘린다.

    Attributes:
        OPENAI: OpenAI API. gpt-5-mini / gpt-5-nano. 팀 한도 $20.
        HF: transformers 로 VM 안에 직접 올린다 (시나리오 A).
    """

    OPENAI = "openai"
    HF = "huggingface"

    @classmethod
    def _missing_(cls, value):
        """대소문자와 앞뒤 공백을 봐준다. .env 나 CSV 에서 오는 값이 지저분하다.

        Args:
            value: 매칭에 실패한 원래 값.

        Returns:
            맞는 멤버, 없으면 None (그러면 파이썬이 ValueError 를 낸다).
        """
        if not isinstance(value, str):
            return None
        cleaned = value.strip().lower()
        for member in cls:
            if cleaned in (member.value, member.name.lower()):
                return member
        return None

    @classmethod
    def list_values(cls):
        """에러 메시지에 "선택 가능한 값" 을 찍을 때 쓴다.

        Returns:
            문자열 값 리스트.
        """
        return [member.value for member in cls]


def make_dirs():
    for folder in (
        METADATA,
        PROCESSED,
        VECTORSTORE,
        LANCEDB,
        CHUNKS,
        REPORTS,
        EVAL_RESULTS,
    ):
        folder.mkdir(parents=True, exist_ok=True)
