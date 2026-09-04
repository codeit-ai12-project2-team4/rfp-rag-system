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
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # scripts/retrieval/ 아래다
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


# 조사·접미어. 질문에 사업명이 들어 있는지 볼 때 이런 건 세면 안 된다.
_STOP = {"사업", "용역", "구축", "시스템", "관련", "위한", "대한", "및", "등"}


def naming(question, title, agency):
    """이 질문이 **공고를 특정하는가.** 사업명·기관 낱말이 두 개 이상 들어 있나.

    1단계는 "클라우드 전환 사업" 같은 주제어로 공고를 고르는 화면이다. 그런데
    평가 세트는 2단계용으로 만든 것이라 "가격 평가 배점이 몇 점인가" 같은
    질문이 많다. **그건 어느 공고를 말하는지 문장만으로 알 수 없다** — 배점
    문항은 거의 모든 공고에 있다. 그런 문항의 Top10 은 검색 성능이 아니라
    운을 잰다.

    9/5 "검토 중" 의 `공고특정불가 필터의 구멍` 이 이 얘기다. 그때는 정답의
    유일성만 봤고 질문의 특정성은 안 봤다.

    Args:
        question: 질문 문장.
        title: 정답 공고의 사업명.
        agency: 정답 공고의 발주기관.

    Returns:
        bool: 특정형이면 True.
    """
    words = {
        w for w in re.split(r"[^가-힣A-Za-z0-9]+", f"{title or ''} {agency or ''}")
        if len(w) >= 2 and w not in _STOP
    }
    return sum(1 for w in words if w in question) >= 2


def metrics(ranks, top_n, seconds=None):
    """순위 목록 하나를 지표로. 전체와 부분 집합에 같은 계산을 쓴다."""
    n = len(ranks) or 1
    out = {
        "MRR": round(sum(1 / r for r in ranks if r) / n, 3),
        "Top1": round(sum(1 for r in ranks if r == 1) / n, 3),
        "Top5": round(sum(1 for r in ranks if r and r <= 5) / n, 3),
        f"Top{top_n}": round(sum(1 for r in ranks if r) / n, 3),
        "질문수": len(ranks),
    }
    if seconds is not None:
        out["초"] = round(seconds, 1)
    return out


def score(pairs, top_n, named=None, **kwargs):
    """한 설정으로 전 문항을 돌리고 지표를 낸다.

    Args:
        pairs: `{"question", "doc_id"}` 를 가진 문항 리스트.
        top_n: 공고를 몇 개까지 돌려받을지. 이 밖으로 나가면 못 찾은 것이다.
        named: 문항별 `naming()` 결과. 주면 특정형/일반형을 나눠서도 낸다.
        **kwargs: `search_notices()` 인자 (pool, chunks, rerank …).

    Returns:
        전체 지표 dict. `named` 를 주면 `"특정형"`·`"일반형"` 이 더 붙는다.
    """
    started = time.time()
    ranks = []
    for i, pair in enumerate(pairs, 1):
        notices = retriever.search_notices(pair["question"], top_n=top_n, **kwargs)
        ranks.append(rank_of(notices, pair["doc_id"]))
        print(f"  {i}/{len(pairs)}", end="\r")
    print(" " * 20, end="\r")

    out = metrics(ranks, top_n, time.time() - started)
    if named:
        out["특정형"] = metrics([r for r, ok in zip(ranks, named) if ok], top_n)
        out["일반형"] = metrics([r for r, ok in zip(ranks, named) if not ok], top_n)
    return out


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
    parser.add_argument("--splade", action="store_true",
                        help="Splade 행을 추가한다. npz 캐시가 있어야 한다")
    parser.add_argument("--splade-model", default="telepix/PIXIE-Splade-v1.5")
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

    # chunks=None 이어야 BM25 를 안 짓는다. 안 주면 search_notices 의 기본값
    # (retriever.CHUNKS)으로 지어서, Dense 행이 **다른 코퍼스의 BM25 와 섞인다.**
    setups = {
        "Dense": {"chunks": None},
        "Hybrid": {"chunks": args.chunks},
    }
    if not args.no_rerank:
        setups["Dense+Rerank"] = {"chunks": None, "rerank": args.rerank}
        setups["Hybrid+Rerank"] = {"chunks": args.chunks, "rerank": args.rerank}

    # Splade 는 1단계에서 한 번도 안 쟀다. 2단계에서 뺀 근거(리랭커가 이득을
    # 먹는다)는 리랭커가 없는 이 화면에 그대로 안 온다.
    if args.splade:
        from pieces import Splade

        splade = Splade(chunking.load_chunks(args.chunks), model=args.splade_model,
                        k=args.pool, cache=args.chunks, verbose=True)
        setups["Splade"] = {"chunks": None, "splade": splade}
        setups["Hybrid(BM25+Splade)"] = {"chunks": args.chunks, "splade": splade}

    # 문항이 공고를 특정하는가. 청크 메타에서 정답 공고의 사업명·기관을 가져온다.
    titles = {}
    for chunk in chunking.load_chunks(args.chunks):
        doc_id = str(chunk.metadata.get("doc_id") or "")
        titles.setdefault(doc_id, (chunk.metadata.get("title"),
                                   chunk.metadata.get("agency")))
    named = [naming(p["question"], *titles.get(str(p["doc_id"]), (None, None)))
             for p in pairs]
    print(f"특정형 {sum(named)}문항 · 일반형 {len(named) - sum(named)}문항")

    rows = []
    splits = []
    for name, extra in setups.items():
        print(f"\n[{name}]")
        got = score(pairs, args.top_n, named=named, pool=args.pool, index=index,
                    embed=args.embed, **extra)
        by_kind = {k: got.pop(k) for k in ("특정형", "일반형") if k in got}
        row = {"설정": name, **got}
        print(f"  MRR {row['MRR']}  Top1 {row['Top1']}  "
              f"Top{args.top_n} {row[f'Top{args.top_n}']}  {row['초']}초")
        rows.append(row)
        for kind, values in by_kind.items():
            splits.append({"설정": name, "질문": kind, **values})

    table = pd.DataFrame(rows).sort_values("MRR", ascending=False)

    # **이 표가 진짜 얘기다.** 전체 Top10 이 낮아도, 공고를 특정하는 질문에서
    # 높고 그렇지 않은 질문에서 낮다면 그건 검색이 아니라 평가 세트 문제다.
    # 1단계 화면에 실제로 들어오는 건 "클라우드 전환 사업" 같은 특정형이다.
    if splits:
        print("\n" + "=" * 70)
        print("질문이 공고를 특정하는가로 나눠 보면")
        print(pd.DataFrame(splits).to_string(index=False))

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
