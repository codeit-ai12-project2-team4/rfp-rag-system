"""공고 검색(1단계)을 잰다. 지금까지 한 번도 안 재본 절반이다.

2단계(공고 안에서 청크 찾기)는 133문항·80문항으로 여러 번 쟀지만, 그 앞의
"어떤 공고를 볼지 고르는" 화면은 정답 세트가 없다고 미뤄 뒀다. **틀린 판단이었다.**
평가 세트의 모든 문항에 `doc_id` 가 붙어 있고, 그게 곧 정답 공고다.

    질문을 search_notices 에 넣는다 → 정답 공고가 몇 위로 나오나

`compare_retrieval.py` 의 `doc_hits` 와 다르다. 그건 "발췌 안에 정답 공고
청크가 있나"고, 이건 "공고 목록에서 몇 위인가"다. 1단계 UI 가 보여주는 건 후자다.

    python scripts/eval_notices.py --chunks cleaned_documents_strip__recursive_1200_200
    python scripts/eval_notices.py --chunks ... --evalset eval_qa --pool 100

리랭킹은 후보 200개를 다시 채점하므로 느리다. 질문당 몇 초씩 걸린다.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import pandas as pd

import retriever
from config import settings
from evaluation import load_evalset


def rank_of(notices, gold):
    """정답 공고가 몇 위인지. 없으면 None."""
    for rank, notice in enumerate(notices, 1):
        if str(notice.get("doc_id")) == str(gold):
            return rank
    return None


def score(pairs, top_n, **kwargs):
    """한 설정으로 전 문항을 돌리고 지표를 낸다.

    Args:
        pairs: `{"question", "doc_id"}` 를 가진 문항 리스트.
        top_n: 공고를 몇 개까지 돌려받을지. 이 밖으로 나가면 못 찾은 것이다.
        **kwargs: `search_notices()` 인자 (pool, chunks, rerank …).

    Returns:
        MRR·Top1·Top5·Top10·질문수·초 dict.
    """
    started = time.time()
    ranks = []
    for i, pair in enumerate(pairs, 1):
        notices = retriever.search_notices(pair["question"], top_n=top_n, **kwargs)
        ranks.append(rank_of(notices, pair["doc_id"]))
        print(f"  {i}/{len(pairs)}", end="\r")
    print(" " * 20, end="\r")

    n = len(pairs)
    return {
        "MRR": round(sum(1 / r for r in ranks if r) / n, 3),
        "Top1": round(sum(1 for r in ranks if r == 1) / n, 3),
        "Top5": round(sum(1 for r in ranks if r and r <= 5) / n, 3),
        f"Top{top_n}": round(sum(1 for r in ranks if r) / n, 3),
        "질문수": n,
        "초": round(time.time() - started, 1),
    }


def main():
    """설정을 바꿔 가며 공고 검색을 재고 표로 찍는다."""
    parser = argparse.ArgumentParser(description="공고 검색(1단계)을 잰다.")
    parser.add_argument("--chunks", required=True, help="청크 이름 (__header 없이)")
    parser.add_argument("--evalset", default="eval_qa_80")
    parser.add_argument("--pool", type=int, default=100, help="훑어볼 청크 수")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument(
        "--embed", default="tei", choices=["tei", "local", "openai", "fake"]
    )
    parser.add_argument("--rerank", default="tei", choices=["tei", "local"])
    parser.add_argument("--no-rerank", action="store_true", help="리랭커 설정을 뺀다")
    parser.add_argument("--out", default=str(settings.EVAL_RESULTS / "notices.csv"))
    args = parser.parse_args()

    index = f"{args.chunks}__{args.embed}"
    pairs = [
        p for p in load_evalset(args.evalset)
        if p.get("answerable", True) and p.get("doc_id")
    ]
    print(f"{len(pairs)}문항 · 인덱스 {index}")

    # 정답 공고가 코퍼스에 없으면 잴 수가 없다. 먼저 센다.
    import chunking

    known = {c.metadata.get("doc_id") for c in chunking.load_chunks(args.chunks)}
    missing = [p for p in pairs if str(p["doc_id"]) not in {str(d) for d in known}]
    if missing:
        print(f"⚠ 정답 공고가 코퍼스에 없는 문항 {len(missing)}개는 뺍니다")
        pairs = [p for p in pairs if p not in missing]

    setups = {
        "Dense": {},
        "Hybrid": {"chunks": args.chunks},
    }
    if not args.no_rerank:
        setups["Dense+Rerank"] = {"rerank": args.rerank}
        setups["Hybrid+Rerank"] = {"chunks": args.chunks, "rerank": args.rerank}

    rows = []
    for name, extra in setups.items():
        print(f"\n[{name}]")
        row = {"설정": name, **score(
            pairs, args.top_n, pool=args.pool, index=index,
            embed=args.embed, **extra
        )}
        print(f"  MRR {row['MRR']}  Top1 {row['Top1']}  "
              f"Top{args.top_n} {row[f'Top{args.top_n}']}  {row['초']}초")
        rows.append(row)

    table = pd.DataFrame(rows).sort_values("MRR", ascending=False)
    print("\n" + "=" * 70)
    print(table.to_string(index=False))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\n저장 → {args.out}")
    print("\nTop1 이 낮으면 1단계에서 사용자가 공고를 못 고른다는 뜻이다.")
    print("Top10 이 높은데 Top1 이 낮으면 순위 문제 — 리랭커가 값을 한다.")
    print("Top10 도 낮으면 후보에 아예 못 드는 것 — pool 이나 임베딩 문제다.")


if __name__ == "__main__":
    main()
