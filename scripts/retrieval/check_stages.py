"""질문 하나가 단계마다 몇 초를 쓰는지. **빼기로 알아내지 않는다.**

    python scripts/retrieval/check_stages.py
    python scripts/retrieval/check_stages.py --splade --pool 30
    python scripts/retrieval/check_stages.py -q "제안서 제출 마감" -n 5

전에 `Hybrid` 377초 vs `Hybrid(BM25+Splade)` 221초를 검색 속도 차이로 읽었는데,
실제로는 먼저 도는 쪽이 BM25 색인을 짓고 있었다. **합계를 빼서 추정하면 틀린다.**
여기서는 `Pipeline` 이 부품마다 남긴 시간을 그대로 찍는다.

Splade 를 채택할지 정할 때 필요한 것도 이것이다 — Dense 를 대체해야 하는지
BM25 를 대체해야 하는지는 둘 중 어느 쪽이 비싼지로 갈린다.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import chunking  # noqa: E402
from config import retrieval as cfg  # noqa: E402
from models import load_embedder, load_reranker  # noqa: E402
from pieces import BM25, Dense, Hybrid, Pipeline, Rerank  # noqa: E402
from retriever import open_index  # noqa: E402

QUESTIONS = [
    "제안서 제출 마감이 언제야",
    "가격 평가 배점이 몇 점인가",
    "이 사업의 예산이 얼마야",
    "입찰 참가 자격이 어떻게 되나",
    "과업 기간은 얼마나 되나",
]


def main():
    parser = argparse.ArgumentParser(description="단계별 시간을 잰다.")
    parser.add_argument("--chunks", default=cfg.chunk_name())
    parser.add_argument("--pool", type=int, default=cfg.POOL)
    parser.add_argument("--top-k", type=int, default=cfg.TOP_K)
    parser.add_argument("--splade", action="store_true", help="Dense 대신 Splade")
    parser.add_argument("--splade-only", action="store_true", help="Splade 단독")
    parser.add_argument("-q", "--question", action="append", help="질문 (여러 번)")
    parser.add_argument("-n", "--repeat", type=int, default=3, help="질문마다 몇 번")
    args = parser.parse_args()

    questions = args.question or QUESTIONS
    index = cfg.index_name(args.chunks)

    print(f"코퍼스 {args.chunks} · pool {args.pool} · top_k {args.top_k}")
    print("준비 중 …")
    started = time.time()

    chunks = chunking.load_chunks(args.chunks)
    bm25 = BM25(chunks, k=args.pool)
    reranker = load_reranker("tei")

    splade = None
    if args.splade or args.splade_only:
        from pieces import Splade

        splade = Splade(chunks, k=args.pool, cache=args.chunks, verbose=True)

    if args.splade_only:
        searcher = splade
        label = "Splade 단독"
    elif args.splade:
        # 9/4 에 잰 "방법 A" — 어휘 매칭 + 어휘 확장. Dense 를 안 쓴다.
        searcher = Hybrid([bm25, splade], weights=[0.4, 0.6], k=args.pool, pool=args.pool)
        label = "Hybrid(BM25+Splade)"
    else:
        store = open_index(index, chunks=chunks)
        if store is None:
            sys.exit(f"인덱스가 없습니다: {index}")
        searcher = Hybrid(
            [Dense(store, k=args.pool), bm25], k=args.pool, pool=args.pool
        )
        label = "Hybrid(Dense+BM25)"

    pipeline = Pipeline([searcher, Rerank(reranker, k=args.top_k)])
    print(f"준비 {time.time() - started:.1f}초 · {label}\n")

    # 첫 질문은 캐시·연결을 데우느라 느리다. 버린다.
    pipeline(questions[0])

    rows = {}
    for question in questions:
        for _ in range(args.repeat):
            state = pipeline(question)
            for name, sec in state.timings:
                rows.setdefault(name, []).append(sec)

    print(f"질문 {len(questions)}개 × {args.repeat}회 (첫 회는 버림)\n")
    total = 0.0
    for name, times in rows.items():
        median = statistics.median(times)
        if not name.startswith("  └"):
            total += median
        print(f"  {name:24s} {median:7.3f}초  (최소 {min(times):.3f} 최대 {max(times):.3f})")
    print(f"  {'─' * 24} {'─' * 7}")
    print(f"  {'질문당 합계':24s} {total:7.3f}초")
    print("\n  `└` 는 Hybrid 안의 자식이라 합계에 두 번 안 센다.")


if __name__ == "__main__":
    main()
