"""pool 만 바꿔가며 확정 설정(Dense+머리말+Rerank)을 잰다.

리랭커에 후보를 몇 개 넘길지가 `pool` 이다. 크면 정답이 후보에 들어올 확률이
오르고 리랭커가 그만큼 느려진다. 지금 기본값 30 은 재보고 고른 값이 아니다.

인덱스를 다시 만들 필요가 없어서 싸다. `compare_retrieval.py` 를 pool 별로
여러 번 돌리면 BM25 를 매번 다시 짓고 안 쓸 설정 7개까지 같이 재게 되므로,
여기서는 인덱스와 리랭커를 한 번만 올리고 pool 만 바꾼다.

    python scripts/sweep_pool.py --chunks cleaned_documents_v3__recursive_1200_200
    python scripts/sweep_pool.py --chunks ... --pools 10,30,60,100
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import pandas as pd

import evaluation as ev
from config import settings
from models import load_embedder, load_reranker
from pieces import Dense, Pipeline, Rerank
from vectorstore import list_stores, load_store


def main():
    """pool 값마다 유형별 지표를 재고 표로 찍는다."""
    parser = argparse.ArgumentParser(description="pool 을 바꿔가며 잰다.")
    parser.add_argument("--chunks", required=True, help="청크 이름 (__header 없이)")
    parser.add_argument("--pools", default="10,30,60,100", help="쉼표로 구분한 pool 값")
    parser.add_argument("--evalset", default="eval_qa")
    parser.add_argument("--embed", default="tei", choices=["tei", "local", "fake"])
    parser.add_argument("--rerank", default="tei", choices=["tei", "local", "fake"])
    parser.add_argument("--budget", type=int, default=settings.MAX_CONTEXT_CHARS)
    parser.add_argument("--out", default=str(settings.EVAL_RESULTS / "pool.csv"))
    args = parser.parse_args()

    index = f"{args.chunks}__header__{args.embed}"
    if index not in list_stores():
        sys.exit(f"인덱스가 없습니다: {index}\n"
                 f"  python src/vectorstore.py --chunks {args.chunks}__header")

    pools = [int(p) for p in args.pools.split(",")]
    store = load_store(index, load_embedder(args.embed))
    reranker = load_reranker(args.rerank)

    pairs = [p for p in ev.load_evalset(args.evalset) if p.get("answerable", True)]
    by_type = {}
    for pair in pairs:
        by_type.setdefault(pair.get("type", "기타"), []).append(pair)

    # 검색기는 루프 밖에서 만든다. lambda 안에서 만들면 질문마다 새로 만든다.
    setups = {}
    for pool in pools:
        pipeline = Pipeline([Dense(store, k=pool), Rerank(reranker, k=pool)])
        setups[f"pool={pool}"] = (lambda p: lambda q: p(q).chunks)(pipeline)

    frames = []
    for kind, group in by_type.items():
        print(f"\n[{kind}] {len(group)}문항")
        frame = ev.compare(setups, group, budget=args.budget)
        frame.insert(0, "유형", kind)
        frames.append(frame)

    table = pd.concat(frames, ignore_index=True)
    hit = [c for c in table.columns if c.startswith("적중률")][0]

    print("\n" + "=" * 70)
    for metric in (hit, "MRR", "초"):
        print(f"\n유형 × pool — {metric}")
        print(table.pivot(index="설정", columns="유형", values=metric))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\n저장 → {args.out}")
    print("\n적중률이 안 오르면 pool 은 30 그대로 두세요. 시간만 늘어납니다.")


if __name__ == "__main__":
    main()
