"""임베딩 — 문장을 벡터로.

    embedder = load_embedder("tei")     # 도커로 띄운 TEI (권장)
    embedder = load_embedder("local")   # 노트북 안에 모델을 올림 (강의 방식)
    embedder = load_embedder("fake")    # 아무것도 안 띄우고 배관만 확인
"""

import os

import requests
from langchain_core.embeddings import Embeddings

# 8000·8001·8080·8081 은 JupyterHub 가 쓰므로 피했다.
# 팀 공용 JupyterHub 가 도는 VM 에서는 이 포트를 절대 쓰면 안 된다.
TEI_EMBED_URL = os.environ.get("TEI_EMBED_URL", "http://localhost:8085")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "dragonkue/BGE-m3-ko")



class TEIEmbeddings(Embeddings):
    """도커로 띄운 TEI 서버에 임베딩을 맡긴다.

    하는 일은 POST 한 번이 전부다.

        POST http://localhost:8085/embed
        {"inputs": ["문장1", "문장2"]}
        → [[0.1, ...], [0.2, ...]]

    모델을 노트북 안에 안 올리므로 커널을 재시작해도 모델이 안 죽고,
    GPU 메모리를 노트북이 붙들고 있지 않는다.

    batch_size 는 TEI 의 --max-client-batch-size 보다 작아야 한다.
    TEI 기본값이 32 라서 그보다 크게 보내면 413 이 뜬다.
    """

    def __init__(self, url=TEI_EMBED_URL, batch_size=32, timeout=120, truncate=True):
        self.url = url.rstrip("/")
        self.batch_size = batch_size
        self.timeout = timeout
        self.truncate = truncate

    def _post(self, texts):
        response = requests.post(
            f"{self.url}/embed",
            json={"inputs": texts, "truncate": self.truncate},
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"TEI 임베딩 실패 {response.status_code}: {response.text[:200]}\n"
                f"서버 주소: {self.url}  (docker/docker-compose.yml 로 띄웠는지 확인)"
            )
        return response.json()

    def embed_documents(self, texts):
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._post(texts[start : start + self.batch_size]))
        return vectors

    def embed_query(self, text):
        return self._post([text])[0]

    def health(self):
        """서버가 살아 있는지, 어떤 모델인지 확인한다."""
        info = requests.get(f"{self.url}/info", timeout=10).json()
        return {
            "model": info.get("model_id"),
            "max_input_length": info.get("max_input_length"),
            "dim": len(self.embed_query("확인")),
        }


class FakeEmbeddings(Embeddings):
    """가짜 임베딩. 서버도 GPU 도 없이 배관만 확인할 때 쓴다.

    글자 3개씩 묶어 해시로 벡터를 만든다. 뜻을 전혀 담지 못하므로
    **이걸로 검색 품질을 판단하면 안 된다.** 파이프라인이 도는지만 본다.
    """

    def __init__(self, dim=256):
        self.dim = dim

    def _vector(self, text):
        import hashlib
        import math

        vector = [0.0] * self.dim
        lowered = text.lower()
        for i in range(max(len(lowered) - 2, 1)):
            gram = lowered[i : i + 3]
            slot = int.from_bytes(hashlib.md5(gram.encode()).digest()[:4], "little")
            vector[slot % self.dim] += 1.0
        length = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / length for v in vector]

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


def load_embedder(kind="tei", model=None, **kwargs):
    """임베딩 모델을 붙인다.

    tei     도커로 띄운 TEI 서버 (권장)
    local   노트북 안에 sentence-transformers 로 올림 (강의 방식)
    fake    가짜. 배관 확인용
    """
    if kind == "tei":
        return TEIEmbeddings(**kwargs)

    if kind == "local":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=model or EMBED_MODEL,
            model_kwargs={"device": kwargs.get("device", "cuda")},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": kwargs.get("batch_size", 16),
            },
        )

    if kind == "fake":
        return FakeEmbeddings(**kwargs)

    raise ValueError(f"모르는 임베딩 종류: {kind!r} (tei / local / fake 중 하나)")
