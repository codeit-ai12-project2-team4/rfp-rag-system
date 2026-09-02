"""임베딩 — 문장을 벡터로.

    embedder = load_embedder("tei")     # 도커로 띄운 TEI (권장)
    embedder = load_embedder("local")   # 노트북 안에 모델을 올림 (강의 방식)
    embedder = load_embedder("fake")    # 아무것도 안 띄우고 배관만 확인
    embedder = load_embedder("openai")  # OpenAI API (시나리오 B)
"""

import os

import requests
from langchain_core.embeddings import Embeddings

TEI_EMBED_URL = os.environ.get("TEI_EMBED_URL", "http://localhost:8085")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "dragonkue/BGE-m3-ko")
# 요즘 임베딩 모델은 질의와 문서에 **서로 다른 접두어**를 요구한다. 안 붙이면
# 오류 없이 성능만 깎여서, 모델이 나쁜 줄 알고 버리게 된다. 두 번 당했다.
#
#     arctic-embed-ko    질의 "query: "                    문서 없음
#     embeddinggemma     질의 "task: search result | query: "  문서 "title: none | text: "
#     BGE-m3             둘 다 없음
#
# 질의 접두어는 검색할 때 붙으므로 바꿔도 재인덱싱이 필요 없다. **문서 접두어는
# 인덱싱 시점에 들어가므로 바꾸면 인덱스를 다시 만들어야 한다.**
# **import 시점에 읽으면 안 된다.** `.env` 를 올리는 건 `config/settings.py` 인데,
# 이 모듈이 먼저 import 되면 값이 빈 문자열로 굳어 버린다. 조용히 접두어 없이
# 돌아간다. 그래서 객체를 만들 때 읽는다.
def _prefix(name):
    """접두어를 지금 읽는다. 뒤 공백이 의미를 갖는 값이라 손대지 않는다.

    Args:
        name (str): 환경변수 이름.

    Returns:
        str: 접두어. 없으면 빈 문자열.
    """
    return os.environ.get(name, "")
# 시나리오 B — API 임베딩. GPU 를 안 굽는 대신 호출당 돈이 든다.
# 청크 9,200개(약 370만 토큰) 기준 3-small $0.07, 3-large $0.48 이다.
EMBED_OPENAI_MODEL = os.environ.get("EMBED_OPENAI_MODEL", "text-embedding-3-large")



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

    def __init__(
        self,
        url=TEI_EMBED_URL,
        batch_size=32,
        timeout=120,
        truncate=True,
        query_prefix=None,
        doc_prefix=None,
    ):
        self.url = url.rstrip("/")
        self.batch_size = batch_size
        self.timeout = timeout
        self.truncate = truncate
        # 빈 문자열이면 안 붙인다.
        self.query_prefix = (
            _prefix("EMBED_QUERY_PREFIX") if query_prefix is None else query_prefix
        )
        self.doc_prefix = (
            _prefix("EMBED_DOC_PREFIX") if doc_prefix is None else doc_prefix
        )

    @property
    def model_id(self):
        """TEI 가 실제로 서빙 중인 모델 이름. `/info` 에 물어본다.

        docker-compose 만 바꾸고 인덱스를 다시 안 만드는 사고가 세 번 났다.
        `.env` 의 EMBED_MODEL 은 그때 같이 안 고쳐지므로 믿을 수 없다.
        서버에 직접 묻는 게 유일하게 맞는 답이다.

        Returns:
            str: 모델 이름. 서버가 대답을 안 하면 EMBED_MODEL 을 쓴다.
        """
        if getattr(self, "_model_id", None) is None:
            try:
                got = requests.get(f"{self.url}/info", timeout=5).json()
                self._model_id = got.get("model_id") or EMBED_MODEL
            except Exception:
                self._model_id = EMBED_MODEL
        return self._model_id

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
        if self.doc_prefix:
            texts = [self.doc_prefix + text for text in texts]
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._post(texts[start : start + self.batch_size]))
        return vectors

    def embed_query(self, text):
        """질의를 벡터로. `EMBED_QUERY_PREFIX` 를 앞에 붙인다.

        문서 쪽은 `embed_documents` 가 `EMBED_DOC_PREFIX` 를 따로 붙인다.
        모델마다 요구하는 접두어가 다르고 질의·문서가 다른 경우가 많다.
        질의 접두어만 바꾸면 재인덱싱이 필요 없지만, **문서 접두어를 바꾸면
        인덱스를 다시 만들어야 한다.**
        """
        return self._post([self.query_prefix + text])[0]

    def health(self):
        """서버가 살아 있는지, 어떤 모델인지 확인한다."""
        info = requests.get(f"{self.url}/info", timeout=10).json()
        return {
            "model": info.get("model_id"),
            "max_input_length": info.get("max_input_length"),
            "dim": len(self.embed_query("확인")),
            # 대괄호로 감싼다. `query:` 와 `query: ` 는 다른 값인데 눈으로 구분이 안 된다
            "질의 접두어": f"[{self.query_prefix}]" if self.query_prefix else "(없음)",
            "문서 접두어": f"[{self.doc_prefix}]" if self.doc_prefix else "(없음)",
        }


class OpenAIEmbeddings(Embeddings):
    """OpenAI 임베딩 API. **시나리오 B** — 서버를 안 띄운다.

    GPU 도 도커도 없이 돌아서 배포가 쉽다. 대신 호출당 돈이 들고, 문서가
    밖으로 나간다 — 원본 RFP 가 NDA 대상이라는 걸 잊으면 안 된다.
    지금은 지표 비교용이고, 실제로 쓸지는 그 다음 문제다.

    차원이 3,072(3-large)라 BGE-m3 의 768 보다 4배다. 인덱스도 그만큼 커진다.

    Args:
        model: `text-embedding-3-large` / `text-embedding-3-small`.
        batch_size: 한 번에 보낼 문장 수. API 는 2,048개까지 받는다.
        dimensions: 주면 그 차원으로 줄여서 받는다 (3 계열만 지원).
    """

    def __init__(self, model=None, batch_size=32, dimensions=None, max_chars=80_000):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model or EMBED_OPENAI_MODEL
        self.batch_size = batch_size
        self.dimensions = dimensions
        # 요청 하나의 토큰 총량에도 한도가 있다. 청크 256개를 한 번에 보냈다가
        # `Request too large for text-embedding-3-large` 로 죽었다.
        # 개수만 세면 안 되고 **글자 수로도** 잘라야 한다.
        self.max_chars = max_chars

    def _post(self, texts):
        import time

        # 빈 문자열을 보내면 400 이 난다. 공백 하나로 바꿔 둔다.
        cleaned = [text if text.strip() else " " for text in texts]
        extra = {"dimensions": self.dimensions} if self.dimensions else {}
        for attempt in range(6):
            try:
                response = self.client.embeddings.create(
                    model=self.model, input=cleaned, **extra
                )
                return [item.embedding for item in response.data]
            except Exception as error:  # 분당 한도(429)면 기다렸다 다시
                if "429" not in str(error) or attempt == 5:
                    raise
                back = 15 * (attempt + 1)
                print(f"  OpenAI 한도 — {back}초 쉬고 재시도    ", end="\r")
                time.sleep(back)
        raise RuntimeError("도달할 수 없음")

    def _batches(self, texts):
        """개수와 글자 수 **둘 다** 넘지 않게 잘라 낸다."""
        batch, chars = [], 0
        for text in texts:
            if batch and (len(batch) >= self.batch_size
                          or chars + len(text) > self.max_chars):
                yield batch
                batch, chars = [], 0
            batch.append(text)
            chars += len(text)
        if batch:
            yield batch

    def embed_documents(self, texts):
        vectors = []
        for batch in self._batches(texts):
            vectors.extend(self._post(batch))
            print(f"  임베딩 {len(vectors):,}/{len(texts):,}", end="\r")
        print(" " * 40, end="\r")
        return vectors

    def embed_query(self, text):
        return self._post([text])[0]

    def health(self):
        return {"model": self.model, "dim": len(self.embed_query("확인"))}


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

    tei     도커로 띄운 TEI 서버 (시나리오 A, 권장)
    local   노트북 안에 sentence-transformers 로 올림
    openai  OpenAI 임베딩 API (시나리오 B)
    fake    가짜. 배관 확인용

    **임베더를 바꾸면 인덱스를 다시 만들어야 한다.** 벡터 공간이 달라서
    옛 인덱스로 조회하면 조용히 엉뚱한 결과가 나온다.
    """
    if kind == "tei":
        return TEIEmbeddings(**kwargs)

    if kind == "openai":
        return OpenAIEmbeddings(model=model, **kwargs)

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
