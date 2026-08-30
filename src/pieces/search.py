"""검색 부품 — 청크를 찾아오는 부품들.

Dense   임베딩(뜻)으로 찾는다.   "클라우드로 옮기는 사업" 같은 질문에 강하다
BM25    키워드(글자)로 찾는다.   "SFR-002", "ISMP" 같은 정확한 말에 강하다
Hybrid  둘을 섞는다.             보통 이게 제일 낫다

셋 다 메서드가 두 개다.

    search(query, k)   질문 하나를 받아 청크 리스트를 돌려준다. 그냥 함수다
    __call__(state)    조립대에 끼울 때 쓴다. state.queries 를 전부 검색한다

`Hybrid` 는 자식 검색기의 `search()` 를 부르기만 한다. 그래서 섞는 코드가
열 줄이면 끝난다.

**섞는 방법(RRF)** — 강의는 두 점수를 0~1 로 정규화해 가중합했다. 여기서는
점수 대신 **등수**를 쓴다.

    점수(청크) = Σ  가중치 / (60 + 그 검색기에서의 등수)

정규화는 문서 집합이 바뀔 때마다 min/max 가 흔들려서 재현이 안 된다.
등수는 그런 문제가 없다. 60 은 관례값이고, 키우면 등수 차이가 덜 중요해진다.
"""

import hashlib
import os
import time
from collections import OrderedDict
from enum import Enum

import numpy as np
from rank_bm25 import BM25Okapi

from pieces.base import dedup_chunks

# --- 한국어 토큰 나누기 ---------------------------------------------------

_kiwi = None


def korean_tokens(text):
    """한국어를 검색용 토큰으로 자른다.

    그냥 text.split() 을 쓰면 "시스템을" 과 "시스템이" 가 다른 단어가 되어서
    BM25 가 못 찾는다. 강의에서 "텀블러 할인" 의 BM25 점수가 0이 나왔던 것과
    같은 문제다. 형태소 분석기(kiwipiepy)로 어간만 남긴다.

    kiwipiepy 가 없으면 그냥 공백으로 자른다(성능은 떨어진다).
    """
    global _kiwi
    try:
        if _kiwi is None:
            from kiwipiepy import Kiwi

            _kiwi = Kiwi()
        return [
            token.form
            for token in _kiwi.tokenize(text)
            if token.tag.startswith(("N", "V", "SL", "SN", "SH"))
            and len(token.form) > 1
        ]
    except ImportError:  # kiwipiepy 없음 — 공백으로 자른다
        return text.split()


def korean_tokens_batch(texts):
    """여러 글을 한 번에 자른다. **한 줄씩 부르는 것보다 훨씬 빠르다.**

    kiwipiepy 의 tokenize() 는 리스트를 받으면 내부에서 여러 스레드를 쓴다.
    청크가 수만 개면 이 차이가 몇 분과 몇십 초를 가른다.
    """
    global _kiwi
    try:
        if _kiwi is None:
            from kiwipiepy import Kiwi

            _kiwi = Kiwi()
        out = []
        for tokens in _kiwi.tokenize(list(texts)):
            out.append([
                t.form
                for t in tokens
                if t.tag.startswith(("N", "V", "SL", "SN", "SH")) and len(t.form) > 1
            ])
        return out
    except ImportError:  # kiwipiepy 없음
        return [korean_tokens(t) for t in texts]


def has_kiwi():
    """형태소 분석기가 실제로 동작하는지 확인한다."""
    a = set(korean_tokens("시스템을 구축한다"))
    b = set(korean_tokens("시스템이 구축되었다"))
    return bool(a & b)


# --- 검색 부품이 함께 쓰는 것 ---------------------------------------------


def keep_docs(chunks, doc_ids):
    """특정 공고들의 청크만 남긴다. doc_ids 가 없으면 그대로 돌려준다."""
    if not doc_ids:
        return chunks
    allowed = set(doc_ids)
    return [c for c in chunks if c.metadata.get("doc_id") in allowed]


def run_search(searcher, state):
    """state.queries 를 전부 검색하고 중복을 없앤다.

    Dense / BM25 / Hybrid 의 __call__ 이 이것 한 줄이다.
    """
    found = []
    for query in state.queries or [state.question]:
        found.extend(searcher.search(query))
    state.chunks = dedup_chunks(found)
    state.note(f"질문 {len(state.queries)}개로 검색 → 청크 {len(state.chunks)}개")
    return state


# --- 부품 ---------------------------------------------------------------


# Splade 는 TEI 로도 돌릴 수 있다. 로컬에서 코퍼스를 인코딩하면 맥이 뜨거우니
# 되도록 이쪽을 쓴다. 도커에 `--pooling=splade` 로 띄우고 8084 로 연다.
# **엔드포인트가 다르다** — splade pooling 모델에 `/embed` 를 부르면 424 가 난다.
TEI_SPLADE_URL = os.environ.get("TEI_SPLADE_URL", "http://localhost:8084")


class SpladeModel(Enum):
    """어휘 확장을 위해 추천하는 Splade 모델 라인업."""

    PIXIE = "telepix/PIXIE-Splade-v1.5"
    JANG = "yjoonjang/splade-ko-v1"


def chunk_signature(chunks):
    """이 청크 묶음이 무엇인지 짧은 지문으로.

    청크 이름은 그대로인데 내용만 바뀌는 일이 잦다 — 목차 제거는 **줄**을
    지우므로 청크 개수가 안 변한다. 실제로 9,189개 그대로였다. 개수만 보면
    못 잡으니 본문을 해시한다.

    Args:
        chunks: 청크 리스트.

    Returns:
        str: 12자리 지문.
    """
    digest = hashlib.md5()
    for chunk in chunks:
        digest.update(chunk.page_content.encode())
    return digest.hexdigest()[:12]


class Splade:
    """어휘 확장(Splade)으로 찾는 희소 검색기.

    BM25 는 질문에 있는 글자만 본다. Splade 는 마스크 언어모델을 돌려
    "클라우드" 문서에 "인프라·서버" 같은 안 적힌 낱말까지 가중치를 붙인다.
    키워드 검색인데 동의어를 스스로 만드는 셈이다.

    문서 벡터는 어휘 크기(수만 차원)지만 실제로 0 이 아닌 칸은 수백 개뿐이라,
    상위 `top_terms` 개만 (칸번호, 값) 두 배열로 들고 있는다. 8,381 청크 기준
    17MB 정도다. 전부 들고 있으면 수 GB 가 되고 맥이 뜨거워진다.

    인덱스는 `cache` 이름을 주면 `outputs/vectorstore/` 에 저장했다가 다음
    실행에서 그대로 읽는다. 코퍼스 인코딩이 이 부품에서 제일 비싼 일이다.

    Attributes:
        k (int): 기본으로 돌려줄 청크 개수.
        chunks (list): 검색 대상 청크 리스트.
        model_id (str): 허깅페이스 모델 ID.
        idx (np.ndarray): (청크수, top_terms) 어휘 칸번호.
        val (np.ndarray): (청크수, top_terms) 그 칸의 가중치.
    """

    def __init__(
        self,
        chunks,
        model=SpladeModel.PIXIE,
        k=5,
        batch_size=8,
        top_terms=256,
        max_length=1024,
        url=None,
        cache=None,
        refresh=False,
        verbose=False,
    ):
        """모델을 올리고 청크를 인코딩한다. 캐시가 있으면 인코딩을 건너뛴다.

        Args:
            chunks (iterable): page_content 를 가진 청크들.
            model (SpladeModel or str): 모델 Enum 또는 허깅페이스 ID.
            k (int): 기본 반환 개수. 기본값 5.
            batch_size (int): 한 번에 인코딩할 청크 수. 기본값 8.
                GPU 메모리가 모자라면 알아서 절반씩 줄여 다시 시도한다.
            top_terms (int): 문서당 남길 상위 낱말 수. 기본값 256.
            max_length (int): 자를 토큰 수. 기본값 1024.
                512 로 두면 1,200자 청크의 뒤쪽 40%가 통째로 안 보인다.
                ModernBERT 계열은 8192 까지 받으므로 여유가 있다.
            url (str, optional): TEI 서버 주소. 주면 `/embed_sparse` 로 맡기고
                로컬에 모델을 안 올린다. "tei" 를 주면 기본 주소를 쓴다.
            cache (str, optional): 캐시 이름. 보통 청크 세트 이름을 준다.
                주지 않으면 매번 코퍼스를 다시 인코딩한다.
            refresh (bool): 캐시가 청크와 안 맞을 때 다시 만들지. 만드는 쪽
                (`build_splade.py`)만 True 로 준다. 검색 쪽에서 True 로 두면
                맥에서 코퍼스를 통째로 인코딩하게 된다.
            verbose (bool): 진행 상황 출력 여부.
        """
        self.k = k
        self.chunks = list(chunks)
        self.model_id = model.value if isinstance(model, SpladeModel) else model
        self.batch_size = batch_size
        self.top_terms = top_terms
        self.max_length = max_length
        self.url = TEI_SPLADE_URL if url == "tei" else (url.rstrip("/") if url else None)
        self._model = None
        self._tokenizer = None

        path = None
        if cache:
            from config import settings

            stem = f"{cache}__splade__{self.model_id.split('/')[-1]}"
            path = settings.VECTORSTORE / f"{stem}.npz"

        signature = chunk_signature(self.chunks)
        if path and path.exists():
            saved = np.load(path)
            was = str(saved["signature"]) if "signature" in saved else "(없음)"
            if was == signature:
                self.idx, self.val = saved["idx"], saved["val"]
                if verbose:
                    print(f"Splade 인덱스 재사용 {path.name}")
                return
            if not refresh:
                raise RuntimeError(
                    f"Splade 인덱스가 지금 청크와 다릅니다: {path.name}\n"
                    f"  만들 때  {was}\n"
                    f"  지금     {signature}\n"
                    "청크 개수가 같아도 내용이 바뀌면 못 씁니다. GPU 가 있는 곳에서\n"
                    "다시 만들어 npz 를 가져오세요:\n"
                    f"  python scripts/retrieval/build_splade.py --chunks {cache}"
                )

        started = time.time()
        if verbose:
            print(f"Splade 인덱스 생성 중 … ({self.model_id}, 청크 {len(self.chunks):,}개)")
        self.idx, self.val = self._encode([c.page_content for c in self.chunks], verbose)
        if verbose:
            print(f"Splade 인덱스 {time.time() - started:.0f}초")

        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path, idx=self.idx, val=self.val, signature=signature
            )

    def _load(self):
        """모델과 토크나이저를 한 번만 올린다.

        Returns:
            tuple: (tokenizer, model, device) 세 값.
        """
        if self._model is not None:
            return self._tokenizer, self._model, self._device

        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer, logging

        logging.set_verbosity_error()
        if torch.cuda.is_available():
            self._device = "cuda"
        elif torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForMaskedLM.from_pretrained(self.model_id)
        self._model.to(self._device).eval()  # 옮기는 건 여기 한 번뿐이다
        if self._device == "cuda":
            self._model.half()  # 로짓이 (묶음 x 토큰수 x 어휘수) 라 절반이 크다
        return self._tokenizer, self._model, self._device

    def _encode(self, texts, verbose=False):
        """텍스트들을 희소 벡터로 바꿔 (칸번호, 값) 두 배열로 돌려준다.

        Args:
            texts (list[str]): 인코딩할 텍스트들.
            verbose (bool): 진행률 출력 여부.

        Returns:
            tuple[np.ndarray, np.ndarray]: 각각 (개수, top_terms) 모양.
        """
        if self.url:
            return self._encode_tei(texts, verbose)

        import torch

        tokenizer, model, device = self._load()
        n = min(self.top_terms, model.config.vocab_size)
        idx = np.zeros((len(texts), n), dtype=np.int32)
        val = np.zeros((len(texts), n), dtype=np.float32)

        # 긴 글끼리 모아 자른다. 한 묶음은 그 안에서 제일 긴 글에 맞춰 패딩되므로,
        # 섞여 있으면 300자짜리도 1,024 토큰으로 부풀어 그만큼 메모리를 쓴다.
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        batch_size = self.batch_size
        done = 0

        while done < len(order):
            picked = order[done : done + batch_size]
            try:
                top = self._forward([texts[i] for i in picked], tokenizer, model, device, n)
            except getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError):
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                if device == "cuda":
                    torch.cuda.empty_cache()
                print(f"  GPU 메모리 부족 — 묶음을 {batch_size} 로 줄인다")
                continue

            for row, i in enumerate(picked):
                idx[i] = top[0][row]
                val[i] = top[1][row]
            done += len(picked)

            if verbose and done % (batch_size * 50) < batch_size:
                print(f"  {done:,}/{len(texts):,}")

        return idx, val

    def _forward(self, batch, tokenizer, model, device, n):
        """한 묶음을 인코딩해 상위 n 개 (칸번호, 값) 을 돌려준다.

        메모리가 여기서 갈린다. MLM 머리는 `(묶음, 토큰수, 어휘수)` 를 뱉는데
        32 x 1024 x 50,368 이면 float32 로 **6.6GB** 다. 여기에
        `log1p(relu(...))` 와 마스크 곱을 차례로 하면 사본이 셋이 되어 20GB 가
        된다. 실제로 그렇게 죽었다.

        `log1p(relu(x))` 는 단조증가라 **max pooling 을 먼저 해도 결과가 같다.**
        먼저 `(묶음, 어휘수)` 로 줄이고 나서 계산하면 사본이 안 생긴다.

        Args:
            batch (list[str]): 한 묶음의 텍스트.
            tokenizer: 토크나이저.
            model: MLM 모델.
            device (str): 연산 장치.
            n (int): 남길 상위 낱말 수.

        Returns:
            tuple[np.ndarray, np.ndarray]: (칸번호, 값). 각각 (묶음, n) 모양.
        """
        import torch

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(device)

        with torch.inference_mode():
            logits = model(**inputs).logits
            # 패딩 자리가 max 를 이기지 못하게 눌러 둔다. 제자리 연산이라 사본이 없다
            logits.masked_fill_(inputs.attention_mask.unsqueeze(-1) == 0, -1e4)
            pooled = logits.max(dim=1).values  # 여기서 (묶음, 어휘수) 로 줄어든다
            del logits
            pooled = torch.log1p(torch.relu(pooled))
            top = pooled.topk(n, dim=-1)  # 어차피 대부분 0 이라 상위만 남긴다
            return (
                top.indices.cpu().numpy(),
                top.values.float().cpu().numpy(),
            )

    def _encode_tei(self, texts, verbose=False):
        """인코딩을 TEI 에 맡긴다. `/embed_sparse` 는 (칸번호, 값) 쌍만 돌려준다.

        Args:
            texts (list[str]): 인코딩할 텍스트들.
            verbose (bool): 진행률 출력 여부.

        Returns:
            tuple[np.ndarray, np.ndarray]: 각각 (개수, top_terms) 모양.
        """
        import requests

        n = self.top_terms
        idx = np.zeros((len(texts), n), dtype=np.int32)
        val = np.zeros((len(texts), n), dtype=np.float32)

        for at in range(0, len(texts), self.batch_size):
            batch = texts[at : at + self.batch_size]
            response = requests.post(
                f"{self.url}/embed_sparse",
                json={"inputs": batch, "truncate": True},
                timeout=120,
            )
            if response.status_code == 424:
                raise RuntimeError(
                    f"{self.url} 가 splade 모델이 아니다. "
                    "도커에서 --pooling=splade 로 띄웠는지 본다."
                )
            response.raise_for_status()

            for row, pairs in enumerate(response.json()):
                pairs = sorted(pairs, key=lambda p: -p["value"])[:n]
                idx[at + row, : len(pairs)] = [p["index"] for p in pairs]
                val[at + row, : len(pairs)] = [p["value"] for p in pairs]

            if verbose and at and at % (self.batch_size * 50) == 0:
                print(f"  {at:,}/{len(texts):,}")

        return idx, val

    def search(self, query, k=None):
        """질문과 내적이 큰 상위 k 개 청크를 돌려준다.

        Args:
            query (str): 질문.
            k (int, optional): 반환 개수. 없으면 self.k.

        Returns:
            list: 점수 높은 순 청크 리스트.
        """
        k = k or self.k
        q_idx, q_val = self._encode([query])
        # 질문 쪽 낱말만 모아 한 번에 훑는다. 청크마다 np.dot 하면 8천 번이다
        size = max(int(self.idx.max()), int(q_idx.max())) + 1
        lookup = np.zeros(size, dtype=np.float32)
        lookup[q_idx[0]] = q_val[0]
        scores = (lookup[self.idx] * self.val).sum(axis=1)

        order = np.argsort(scores)[::-1][:k]
        return [self.chunks[i] for i in order if scores[i] > 0]

    def __call__(self, state):
        """조립대에 끼울 때 쓴다.

        Args:
            state: 파이프라인 상태 객체.

        Returns:
            검색 결과가 담긴 상태 객체.
        """
        return run_search(self, state)

    def __repr__(self):
        return f"Splade(model={self.model_id.split('/')[-1]}, k={self.k})"


class Dense:
    """임베딩으로 검색한다. FAISS 인덱스가 필요하다.

    doc_ids 를 주면 그 공고들 안에서만 찾는다. 요약 카드를 만들 때 쓴다.
    """

    def __init__(self, store, k=5, mmr=False, fetch_k=30, doc_ids=None):
        self.store = store
        self.k = k
        self.mmr = mmr  # 결과를 서로 덜 겹치게 뽑고 싶으면 True
        self.fetch_k = fetch_k
        self.doc_ids = doc_ids

    def search(self, query, k=None):
        k = k or self.k
        # 공고를 좁혔으면 넉넉히 뽑아야 거르고 나서 남는 게 있다
        take = k * 10 if self.doc_ids else k
        if self.mmr:
            hits = self.store.max_marginal_relevance_search(
                query, k=take, fetch_k=max(self.fetch_k, take * 2)
            )
        else:
            hits = self.store.similarity_search(query, k=take)
        return keep_docs(hits, self.doc_ids)[:k]

    def __call__(self, state):
        return run_search(self, state)

    def __repr__(self):
        scope = f", 공고 {len(self.doc_ids)}건" if self.doc_ids else ""
        return f"Dense(k={self.k}, {'MMR' if self.mmr else '유사도'}{scope})"


# 같은 청크로 만든 BM25 인덱스는 한 번만 만들고 돌려쓴다.
# 조립대를 여러 개 만들면 BM25(chunks, k=20) 도 여러 번 쓰게 되는데,
# 인덱스 하나가 수백 MB 다. 열 개면 VM 이 멈춘다(실제로 멈췄다).
# k 만 다른 건 같은 인덱스로 충분하다 — k 는 검색할 때 쓰는 값이니까.
#
# 다만 무한정 쌓아 두면 그것대로 메모리를 먹는다. 청킹 설정을 바꿔 가며
# 도는 루프(노트북 4의 설정 비교)에서는 매번 다른 청크 묶음이 들어오기 때문이다.
# 그래서 최근 것 몇 개만 남긴다.
_BM25_CACHE = OrderedDict()
BM25_CACHE_SIZE = 2


def _bm25_key(chunks, tokenizer):
    """같은 청크 묶음인지 싸게 알아보는 지문."""
    head = chunks[0].page_content[:80] if chunks else ""
    tail = chunks[-1].page_content[:80] if chunks else ""
    name = getattr(tokenizer, "__name__", None) or repr(tokenizer)
    return (len(chunks), head, tail, name)


def clear_bm25_cache():
    """돌려쓰던 인덱스를 버린다. 메모리를 되찾고 싶을 때."""
    count = len(_BM25_CACHE)
    _BM25_CACHE.clear()
    import gc

    gc.collect()
    print(f"BM25 인덱스 {count}개를 버렸습니다.")


class BM25:
    """키워드로 검색한다. 인덱스도 API 키도 필요 없다.

    **만드는 게 비싸다.** 청크를 전부 형태소 분석해야 해서 수만 개면
    수십 초가 걸리고 메모리도 수백 MB 를 먹는다.

    같은 청크 묶음으로 두 번째부터는 **자동으로 돌려쓴다.** 그래서 이렇게 써도
    인덱스는 하나만 만들어진다.

        Pipeline([Hybrid([Dense(store, k=20), BM25(chunks, k=20)], k=5)])
        Pipeline([BM25(chunks, k=3)])          # ← 위에서 만든 걸 그대로 쓴다

    그래도 밖에서 한 번 만들어 이름을 붙여 두는 쪽이 읽기 좋다.

        bm25 = BM25(chunks, k=20)
        compare({"BM25": lambda q: bm25.search(q, 5)}, questions)

    메모리를 되찾으려면 `clear_bm25_cache()`.
    """

    def __init__(
        self, chunks, k=5, tokenizer=None, doc_ids=None, verbose=False, cache=True
    ):
        import time

        self.k = k
        self.tokenizer = tokenizer or korean_tokens
        self.doc_ids = doc_ids

        key = _bm25_key(chunks, self.tokenizer) if cache else None
        if key is not None and key in _BM25_CACHE:
            _BM25_CACHE.move_to_end(key)
            self.chunks, self.bm25 = _BM25_CACHE[key]
            if verbose:
                print(f"BM25 인덱스 재사용: 청크 {len(self.chunks):,}개 (새로 안 만듦)")
            return

        from resources import need_memory

        # 청크 1만 개당 대략 0.5GB. 없으면 만들기 전에 멈춘다.
        need_memory(max(1.0, len(chunks) / 10000 * 0.5), what="BM25 인덱스")

        self.chunks = list(chunks)
        started = time.time()
        if tokenizer is None:
            tokenized = korean_tokens_batch([c.page_content for c in self.chunks])
        else:
            tokenized = [tokenizer(c.page_content) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)
        del tokenized

        elapsed = time.time() - started
        if verbose or elapsed > 20:
            print(f"BM25 인덱스 준비: 청크 {len(self.chunks):,}개, {elapsed:.1f}초")

        if key is not None:
            _BM25_CACHE[key] = (self.chunks, self.bm25)
            while len(_BM25_CACHE) > BM25_CACHE_SIZE:
                _BM25_CACHE.popitem(last=False)  # 오래된 것부터 버린다

    def search(self, query, k=None):
        k = k or self.k
        scores = self.bm25.get_scores(self.tokenizer(query))
        allowed = set(self.doc_ids) if self.doc_ids else None

        picked = []
        for i in np.argsort(scores)[::-1]:
            if scores[i] <= 0:
                break
            chunk = self.chunks[i]
            if allowed and chunk.metadata.get("doc_id") not in allowed:
                continue
            picked.append(chunk)
            if len(picked) >= k:
                break
        return picked

    def __call__(self, state):
        return run_search(self, state)

    def __repr__(self):
        scope = f", 공고 {len(self.doc_ids)}건" if self.doc_ids else ""
        return f"BM25(k={self.k}{scope})"


class Hybrid:
    """검색기 여러 개를 돌리고 등수로 합친다.

        Hybrid([Dense(store), BM25(chunks)], weights=[0.5, 0.5], k=5)

    pool 은 자식 검색기에게 몇 개씩 받아올지다. 넉넉히 받아서 합친 뒤
    k 개로 좁힌다.
    """

    def __init__(self, searchers, weights=None, k=5, pool=20, rrf_k=60):
        self.searchers = list(searchers)
        self.weights = weights or [1.0 / len(self.searchers)] * len(self.searchers)
        self.k = k
        self.pool = pool
        self.rrf_k = rrf_k

    def search(self, query, k=None):
        scores = {}
        found = {}
        for searcher, weight in zip(self.searchers, self.weights, strict=False):
            for rank, chunk in enumerate(searcher.search(query, self.pool), 1):
                key = chunk.metadata.get("chunk_id") or chunk.page_content[:120]
                scores[key] = scores.get(key, 0.0) + weight / (self.rrf_k + rank)
                found.setdefault(key, chunk)

        best = sorted(scores, key=scores.get, reverse=True)[: k or self.k]
        for key in best:
            found[key].metadata["score"] = round(scores[key], 6)
        return [found[key] for key in best]

    def __call__(self, state):
        return run_search(self, state)

    def __repr__(self):
        inner = ", ".join(
            f"{s!r}×{w:.2f}" for s, w in zip(self.searchers, self.weights, strict=False)
        )
        return f"Hybrid([{inner}], k={self.k})"


class FilterBy:
    """이미 찾아온 청크를 조건으로 거른다. **검색 뒤에** 놓는다.

        Pipeline([
            Hybrid([dense, bm25], k=30),
            FilterBy(budget_min=3e8),         # 예산 3억 이상만
            TopK(5),
        ])

    "예산 3억 이상인 공고 중에서 클라우드 전환 요구가 있는 것" 같은 질문은
    의미 검색만으로는 안 된다. 숫자 조건은 이 부품이 지킨다.

    뒤에서 거르는 방식이라 **앞 단계 k 를 넉넉히** 잡아야 남는 게 있다.

    검색 자체를 특정 공고 안으로 좁히고 싶으면 이게 아니라 검색기에 직접 준다.
    요약 카드가 그 경우다.

        Dense(store, k=15, doc_ids=["20241001798-0"])
        BM25(chunks, k=15, doc_ids=["20241001798-0"])
    """

    def __init__(
        self,
        doc_ids=None,
        agency_has=None,
        budget_min=None,
        budget_max=None,
        close_after=None,
        close_before=None,
    ):
        self.doc_ids = doc_ids
        self.agency_has = agency_has
        self.budget_min = budget_min
        self.budget_max = budget_max
        self.close_after = close_after
        self.close_before = close_before

    def keep(self, meta):
        """이 청크를 남길지 판단한다. 조건을 늘리려면 여기에 한 줄씩 더한다."""
        if self.doc_ids and meta.get("doc_id") not in self.doc_ids:
            return False
        if self.agency_has and self.agency_has not in (meta.get("agency") or ""):
            return False

        budget = meta.get("budget")
        if self.budget_min is not None and (budget is None or budget < self.budget_min):
            return False
        if self.budget_max is not None and (budget is None or budget > self.budget_max):
            return False

        close = meta.get("bid_close_at")
        if self.close_after or self.close_before:
            if not close:
                return False
            if self.close_after and close < self.close_after:
                return False
            if self.close_before and close > self.close_before:
                return False
        return True

    def __call__(self, state):
        before = len(state.chunks)
        state.chunks = [c for c in state.chunks if self.keep(c.metadata)]
        state.note(f"조건으로 거름 {before} → {len(state.chunks)}개")
        return state

    def __repr__(self):
        conds = []
        if self.doc_ids:
            conds.append(f"공고 {len(self.doc_ids)}건")
        if self.agency_has:
            conds.append(f"기관~'{self.agency_has}'")
        if self.budget_min:
            conds.append(f"예산≥{self.budget_min / 1e8:.1f}억")
        if self.budget_max:
            conds.append(f"예산≤{self.budget_max / 1e8:.1f}억")
        return f"FilterBy({', '.join(conds) or '조건없음'})"


if __name__ == "__main__":
    # Splade 점수 계산만 확인한다. 모델은 안 올린다.
    import types

    class _Chunk:
        def __init__(self, text):
            self.page_content = text

    s = Splade.__new__(Splade)
    s.chunks = [_Chunk("a"), _Chunk("b"), _Chunk("c")]
    s.k = 2
    s.idx = np.array([[0, 1], [1, 2], [3, 4]], np.int32)
    s.val = np.array([[1.0, 1.0], [2.0, 1.0], [5.0, 5.0]], np.float32)
    s._model = types.SimpleNamespace(config=types.SimpleNamespace(vocab_size=8))
    s._encode = lambda texts: (
        np.array([[1, 2]], np.int32),
        np.array([[1.0, 1.0]], np.float32),
    )
    # b = 2*1 + 1*1 = 3, a = 1, c = 0 이라 탈락
    assert [c.page_content for c in s.search("질문")] == ["b", "a"]
    print("Splade 점수 계산 통과")
