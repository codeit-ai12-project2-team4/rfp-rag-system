"""리랭커 — 질문과 문서를 쌍으로 넣고 점수를 매긴다.

    reranker = load_reranker("tei")     # 도커로 띄운 TEI (권장)
    reranker = load_reranker("local")   # 노트북 안에 크로스 인코더를 올림 (강의 L05)
    reranker = load_reranker("fake")    # 글자 겹침. 배관 확인용

부품 쪽은 score(query, texts) 하나만 부른다.
"""

import os

import requests

TEI_RERANK_URL = os.environ.get("TEI_RERANK_URL", "http://localhost:8086")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "dragonkue/bge-reranker-v2-m3-ko")



class TEIReranker:
    """TEI 리랭커 서버. 질문과 문서를 쌍으로 넣고 점수를 받는다.

    POST http://localhost:8086/rerank
    {"query": "질문", "texts": ["문서1", "문서2"]}
    → [{"index": 1, "score": 0.98}, {"index": 0, "score": 0.12}]
    """

    name = "TEI"

    def __init__(self, url=TEI_RERANK_URL, batch_size=32, timeout=120, truncate=True):
        self.url = url.rstrip("/")
        self.batch_size = batch_size
        self.timeout = timeout
        self.truncate = truncate

    def score(self, query, texts):
        """texts 순서 그대로 점수 리스트를 돌려준다."""
        scores = [0.0] * len(texts)
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = requests.post(
                f"{self.url}/rerank",
                json={"query": query, "texts": batch, "truncate": self.truncate},
                timeout=self.timeout,
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"TEI 리랭커 실패 {response.status_code}: {response.text[:200]}\n"
                    f"서버 주소: {self.url}"
                )
            for item in response.json():
                scores[start + item["index"]] = item["score"]
        return scores

    def health(self):
        info = requests.get(f"{self.url}/info", timeout=10).json()
        return {
            "model": info.get("model_id"),
            "max_input_length": info.get("max_input_length"),
        }


class LocalReranker:
    """노트북 안에 크로스 인코더를 올린다. 강의 L05 방식."""

    name = "local"

    def __init__(self, model=None, device="cuda", max_length=512):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(
            model or RERANK_MODEL, device=device, max_length=max_length
        )

    def score(self, query, texts):
        return list(self.model.predict([(query, text) for text in texts]))


class FakeReranker:
    """가짜 리랭커. 글자 겹침으로 점수를 낸다. 배관 확인용."""

    name = "fake"

    def score(self, query, texts):
        keys = set(query)
        return [len(keys & set(text)) / (len(keys) + 1) for text in texts]


def load_reranker(kind="tei", **kwargs):
    if kind == "tei":
        return TEIReranker(**kwargs)
    if kind == "local":
        return LocalReranker(**kwargs)
    if kind == "fake":
        return FakeReranker()
    raise ValueError(f"모르는 리랭커 종류: {kind!r} (tei / local / fake 중 하나)")
