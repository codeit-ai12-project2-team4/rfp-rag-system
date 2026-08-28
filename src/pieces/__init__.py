"""부품과 조립대.

    from pieces import *

    found = Pipeline([
        Dense(store, k=30),
        Rerank(reranker, k=8),
    ])("과업기간이 어떻게 되나요?").chunks

조립해 놓은 것을 그냥 쓰려면 `src/retriever.py`. 답을 만드는 건 그 다음 단계인
`src/generation.py` 다. 여기는 검색까지다.

부품을 새로 만들려면 __call__(self, state) 하나만 쓴다. 상속할 부모는 없다.
자세한 건 base.py 맨 위 주석.
"""

from pieces.base import Pipeline, State, dedup_chunks, name_of
from pieces.expand import AddKeywords
from pieces.refine import Rerank, TopK, Widen
from pieces.search import (
    BM25,
    Dense,
    FilterBy,
    Hybrid,
    clear_bm25_cache,
    has_kiwi,
    keep_docs,
    korean_tokens,
    run_search,
)

__all__ = [
    "BM25",
    "AddKeywords",
    "Dense",
    "FilterBy",
    "Hybrid",
    "Pipeline",
    "Rerank",
    "State",
    "TopK",
    "Widen",
    "clear_bm25_cache",
    "dedup_chunks",
    "has_kiwi",
    "keep_docs",
    "korean_tokens",
    "name_of",
    "run_search",
]
