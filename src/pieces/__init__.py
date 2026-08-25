"""부품과 조립대.

    from pieces import *

    rag = Pipeline([
        Hybrid([Dense(store, k=20), BM25(chunks, k=20)], k=5),
        Rerank(reranker, k=3),
        Generate(llm),
    ])

부품을 새로 만들려면 __call__(self, state) 하나만 쓴다. 상속할 부모는 없다.
자세한 건 base.py 맨 위 주석.
"""

from pieces.base import Pipeline, State, dedup_chunks, name_of
from pieces.expand import AddKeywords, MultiQuery, QueryRewrite
from pieces.generate import (
    CARD_FIELDS,
    Generate,
    MakeCard,
    format_context,
    render_card,
)
from pieces.refine import Compress, Rerank, TopK, Widen
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
    "CARD_FIELDS",
    "AddKeywords",
    "Compress",
    "Dense",
    "FilterBy",
    "Generate",
    "Hybrid",
    "MakeCard",
    "MultiQuery",
    "Pipeline",
    "QueryRewrite",
    "Rerank",
    "State",
    "TopK",
    "Widen",
    "clear_bm25_cache",
    "dedup_chunks",
    "format_context",
    "has_kiwi",
    "keep_docs",
    "korean_tokens",
    "name_of",
    "render_card",
    "run_search",
]
