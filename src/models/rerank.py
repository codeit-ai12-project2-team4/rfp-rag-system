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
    """크로스 인코더를 이 프로세스에 직접 올린다.

    **TEI 가 못 받는 모델을 재볼 때 쓴다.** TEI 는 자기가 구현한 구조만
    돌린다 — Qwen3 리랭커도 jina 도 거부당했다. sentence-transformers 는
    `trust_remote_code` 로 아무 구조나 받으므로, "그 모델이 실제로 더 나은가"를
    먼저 여기서 확인하고, 이길 때만 서빙을 고민하면 된다.

    느리다. 지표를 재는 용도이지 서비스용이 아니다.

    Args:
        model: HuggingFace 모델 이름. 생략하면 `RERANK_MODEL`.
        device: 생략하면 cuda → mps → cpu 순으로 알아서 고른다.
        max_length: 질문+문서를 이 토큰 수로 자른다.
        trust_remote_code: jina 처럼 커스텀 코드가 필요한 모델에 True.
    """

    name = "local"

    def __init__(self, model=None, device=None, max_length=512, trust_remote_code=False):
        import torch
        from sentence_transformers import CrossEncoder

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.model = CrossEncoder(
            model or RERANK_MODEL,
            device=device,
            max_length=max_length,
            trust_remote_code=trust_remote_code,
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
