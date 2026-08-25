"""모델 붙이기 — 임베딩 / 리랭커 / LLM.

어떤 걸 쓰든 부품 쪽 코드는 그대로다. 여기서만 바꾸면 된다.

    embedder = load_embedder("tei")          # 도커로 띄운 TEI
    embedder = load_embedder("local")        # 노트북 안에 모델을 올림
    embedder = load_embedder("fake")         # 아무것도 안 띄우고 배관만 확인

    reranker = load_reranker("tei")

    llm = load_llm("hf")                     # 노트북 안에 LLM 을 올림 (강의 방식)
    llm = load_llm("vllm")                   # 도커로 띄운 vLLM
    llm = load_llm("openai")                 # OpenAI API

LLM 은 전부 ask(system, user) 하나만 있으면 된다. 강의의 ask() 와 같은 자리다.
"""

import os

import requests
from langchain_core.embeddings import Embeddings

# 기본 주소. .env 나 환경변수로 바꿀 수 있다.
#
# 8000·8001·8080·8081 은 JupyterHub 가 쓰므로 피했다.
# 팀 공용 JupyterHub 가 도는 VM 에서는 이 포트를 절대 쓰면 안 된다.
TEI_EMBED_URL = os.environ.get("TEI_EMBED_URL", "http://localhost:8085")
TEI_RERANK_URL = os.environ.get("TEI_RERANK_URL", "http://localhost:8086")
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8087/v1")

EMBED_MODEL = os.environ.get("EMBED_MODEL", "dragonkue/BGE-m3-ko")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "dragonkue/bge-reranker-v2-m3-ko")
LOCAL_LLM_MODEL = os.environ.get("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini")


# =========================================================================
# 임베딩
# =========================================================================


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


# =========================================================================
# 리랭커
# =========================================================================


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


# =========================================================================
# LLM
# =========================================================================


# 모델별 대략 내려받는 용량 (fp16 기준). 디스크가 부족한 VM 이 많아서 미리 경고한다.
MODEL_SIZE_GB = {
    "Qwen/Qwen2.5-7B-Instruct": 15,
    "Qwen/Qwen2.5-3B-Instruct": 6,
    "Qwen/Qwen2.5-1.5B-Instruct": 3,
    "Qwen/Qwen3-4B-Instruct-2507": 8,
    "unsloth/Llama-3.2-3B-Instruct": 6,
    "K-intelligence/Midm-2.0-Mini-Instruct": 5,
}


def free_disk_gb(path="/"):
    import shutil

    return shutil.disk_usage(path).free / 1024**3


class HFLLM:
    """노트북 안에 모델을 올려서 쓴다. 강의에서 하던 방식 그대로.

    **디스크를 많이 먹는다.** 7B 를 fp16 으로 받으면 15GB 다. 거기에 torch 와
    CUDA 라이브러리가 6~8GB. VM 디스크가 50GB 면 이것만으로 절반이 찬다.

    디스크가 빠듯하면 순서대로 검토할 것:
      1. load_llm("openai")     디스크 0. 팀 한도 $20 안에서
      2. 작은 모델              3B 는 6GB, 1.5B 는 3GB
      3. AWQ 4bit               7B 가 약 5GB 로 줄어든다
      4. 디스크를 늘린다        과제 지침상 200GB 까지 허용된다

    L4 24GB 면 7~9B 를 float16 으로 **올릴 수는** 있다. GPU 메모리와 디스크는
    다른 문제라는 걸 헷갈리지 말 것 — 여기서 막히는 건 대개 디스크다.
    """

    name = "hf"

    def __init__(self, model=None, device="cuda", dtype=None, skip_disk_check=False):
        model_name = model or LOCAL_LLM_MODEL

        # 무거운 import 전에 디스크부터 본다. 다 받고 나서 터지면 시간만 버린다.
        need = MODEL_SIZE_GB.get(model_name)
        free = free_disk_gb()
        if not skip_disk_check and need and free < need * 1.3:
            raise RuntimeError(
                f"디스크가 부족합니다.\n"
                f"  {model_name} 은 약 {need}GB 를 받습니다 (여유 공간은 그 1.3배는 있어야 함)\n"
                f"  지금 남은 공간: {free:.1f}GB\n\n"
                f"선택지:\n"
                f"  1. load_llm('openai')                 디스크 0\n"
                f"  2. load_llm('hf', model='Qwen/Qwen2.5-1.5B-Instruct')   약 3GB\n"
                f"  3. 캐시 정리: rm -rf ~/.cache/huggingface/hub/models--*\n"
                f"  4. 디스크 늘리기 (과제 지침상 200GB 까지 허용)\n\n"
                f"그래도 강행하려면 skip_disk_check=True"
            )
        if need:
            print(f"내려받기 시작: {model_name} (약 {need}GB, 남은 공간 {free:.1f}GB)")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype or torch.float16
        ).to(device)
        self.device = device
        print(f"모델 올림: {model_name} ({device})")

    def ask(self, system, user, max_tokens=800):
        text = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        output = self.model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=False
        )
        generated = output[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


class OpenAILLM:
    """OpenAI API 또는 그와 호환되는 서버(vLLM, TGI)에 물어본다.

    vLLM 을 도커로 띄우면 OpenAI 와 똑같은 형식으로 말을 받으므로
    base_url 만 바꾸면 같은 코드로 쓸 수 있다.
    """

    def __init__(
        self, model=None, base_url=None, api_key=None, temperature=0.0, name="openai"
    ):
        from openai import OpenAI

        self.name = name
        self.model = model or OPENAI_MODEL
        self.temperature = temperature
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or os.environ.get("OPENAI_API_KEY") or "unused",
        )

    def ask(self, system, user, max_tokens=800):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            max_completion_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()


class EchoLLM:
    """가짜 LLM. 프롬프트에 뭐가 들어갔는지만 보고 싶을 때.

    돈도 GPU 도 안 든다. 부품을 새로 만들고 배관을 확인할 때 쓴다.
    """

    name = "echo"

    def __init__(self, reply=None):
        self.reply = reply
        self.calls = []

    def ask(self, system, user, max_tokens=800):
        self.calls.append({"system": system, "user": user})
        if self.reply is not None:
            return self.reply
        return "\n".join([
            "(가짜 LLM 입니다. 실제 답변이 아닙니다.)",
            f"발췌 {user.count('---') + 1}덩어리를 받았습니다.",
            f"프롬프트 길이 {len(system) + len(user)}자",
        ])


def load_llm(kind="openai", model=None, **kwargs):
    """답변 생성 모델을 붙인다.

    | 종류 | 무엇 | 디스크 | 돈 |
    |---|---|---|---|
    | `openai` | OpenAI API | 0 | 팀 한도 $20 |
    | `echo`   | 가짜. 프롬프트 확인용 | 0 | 0 |
    | `vllm`   | 도커로 띄운 vLLM | 이미지 10GB + 모델 | 0 |
    | `hf`     | 노트북 안에 transformers 로 (강의 방식) | 모델 크기만큼 | 0 |

    **기본값이 openai 인 이유** — `hf` 는 7B fp16 이면 15GB 를 내려받는다.
    50GB VM 에서 그냥 부르면 디스크가 찬다. 로컬 모델은 필요할 때
    명시적으로 고르게 했다.
    """
    if kind == "hf":
        return HFLLM(model=model, **kwargs)
    if kind == "vllm":
        return OpenAILLM(
            model=model or LOCAL_LLM_MODEL,
            base_url=kwargs.pop("base_url", VLLM_URL),
            api_key="local",
            name="vllm",
            **kwargs,
        )
    if kind == "openai":
        return OpenAILLM(model=model or OPENAI_MODEL, name="openai", **kwargs)
    if kind == "echo":
        return EchoLLM(**kwargs)
    raise ValueError(f"모르는 LLM 종류: {kind!r} (hf / vllm / openai / echo 중 하나)")


# =========================================================================
# 상태 확인
# =========================================================================


def check_servers():
    """지금 무엇이 떠 있는지 한눈에. 노트북 맨 앞에서 부르면 편하다."""
    print("=" * 60)
    for label, url in [
        ("임베딩 (TEI)", TEI_EMBED_URL),
        ("리랭커 (TEI)", TEI_RERANK_URL),
    ]:
        try:
            info = requests.get(f"{url}/info", timeout=3).json()
            print(f"  O  {label:<14} {url}  →  {info.get('model_id')}")
        except Exception:
            print(f"  X  {label:<14} {url}  →  응답 없음")

    try:
        models = requests.get(f"{VLLM_URL}/models", timeout=3).json()
        served = [m["id"] for m in models.get("data", [])]
        print(f"  O  {'생성 (vLLM)':<14} {VLLM_URL}  →  {served}")
    except Exception:
        print(f"  X  {'생성 (vLLM)':<14} {VLLM_URL}  →  응답 없음")

    key = os.environ.get("OPENAI_API_KEY")
    print(f"  {'O' if key else 'X'}  {'OpenAI 키':<14} {'설정됨' if key else '없음'}")

    from pieces.search import has_kiwi

    print(
        f"  {'O' if has_kiwi() else 'X'}  {'형태소 분석기':<13} "
        f"{'kiwipiepy 동작' if has_kiwi() else 'kiwipiepy 없음 → BM25 성능 떨어짐'}"
    )

    # 메모리가 바닥나면 디스크와 달리 VM 이 통째로 멈춘다. 먼저 볼 것.
    from resources import DANGER_GB, free_disk_gb, memory, process_memory_gb

    stat = memory()
    if stat:
        mark = "X" if stat["available_gb"] < DANGER_GB else "O"
        mine = process_memory_gb()
        tail = f" · 이 커널 {mine}GB" if mine is not None else ""
        print(
            f"  {mark}  {'메모리':<15} 전체 {stat['total_gb']}GB · "
            f"남음 {stat['available_gb']}GB{tail}"
        )
        if mark == "X":
            print(
                "      ⚠ 이대로 큰 셀을 돌리면 VM 이 멈추고 SSH 가 끊깁니다. "
                "커널부터 재시작하세요."
            )
    print(f"  O  {'디스크':<15} 남음 {free_disk_gb():.1f}GB")
    print("=" * 60)
