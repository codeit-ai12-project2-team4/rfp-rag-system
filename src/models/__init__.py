"""모델 붙이기 — 임베딩 / 리랭커 / LLM.

어떤 걸 쓰든 부품 쪽 코드는 그대로다. 여기서만 바꾸면 된다.

    from models import load_embedder, load_reranker, load_llm, check_servers

각 파일이 자기 환경변수를 갖는다.

    embed.py    TEI_EMBED_URL(8085)  EMBED_MODEL
    rerank.py   TEI_RERANK_URL(8086) RERANK_MODEL
    llm.py      VLLM_URL(8087)       LOCAL_LLM_MODEL  OPENAI_MODEL
"""

from models.embed import (
    EMBED_MODEL,
    TEI_EMBED_URL,
    FakeEmbeddings,
    TEIEmbeddings,
    load_embedder,
)
from models.health import check_servers
from models.llm import (
    HFLLM,
    LOCAL_LLM_MODEL,
    MODEL_SIZE_GB,
    OPENAI_MODEL,
    VLLM_URL,
    EchoLLM,
    OpenAILLM,
    load_llm,
)
from models.rerank import (
    RERANK_MODEL,
    TEI_RERANK_URL,
    FakeReranker,
    LocalReranker,
    TEIReranker,
    load_reranker,
)

__all__ = [
    "EMBED_MODEL",
    "HFLLM",
    "LOCAL_LLM_MODEL",
    "MODEL_SIZE_GB",
    "OPENAI_MODEL",
    "RERANK_MODEL",
    "TEI_EMBED_URL",
    "TEI_RERANK_URL",
    "VLLM_URL",
    "EchoLLM",
    "FakeEmbeddings",
    "FakeReranker",
    "LocalReranker",
    "OpenAILLM",
    "TEIEmbeddings",
    "TEIReranker",
    "check_servers",
    "load_embedder",
    "load_llm",
    "load_reranker",
]
