#!/usr/bin/env python
"""청킹 설정을 비교한다. 전처리본 두 개를 나란히 놓고 볼 수도 있다.

    python scripts/compare_chunking.py
    python scripts/compare_chunking.py --docs documents cleaned_documents
    python scripts/compare_chunking.py --limit 20        먼저 20건으로 시간 재기

## 공정하게 비교하려면

전처리본이 다르면 본문이 다르다. A 에서 뽑은 질문의 정답이 B 본문에 아예 없으면
B 가 부당하게 손해를 본다. 그래서 **두 본문에 다 있는 질문만 남긴다.** 몇 개를
버렸는지도 찍는다.

## 청크가 크면 적중률이 그냥 오른다

극단적으로 문서 하나를 청크 하나로 만들면 적중률@5 가 1.0 이 된다. 크기가 다른
설정을 개수로 비교하면 **큰 쪽이 무조건 이긴다.** 그래서 기본값은 상위 k개가
아니라 **글자 예산**(`settings.MAX_CONTEXT_CHARS`, 6000자)이다. 생성 단계가
실제로 받는 것도 개수가 아니라 글자 수다. 예전 방식으로 보려면 `--budget 0`.

## BM25 만 쓴다

청킹 설정을 고르는 데는 BM25 로 충분하고 공짜다 (LLM·GPU·서버 안 씀).
Dense 는 설정마다 인덱스를 새로 만들어야 해서 비싸다. **BM25 로 후보를 좁힌 뒤
이긴 설정 하나만 Dense 로 확인하는 순서**가 맞다.

주의 — BM25 인덱스는 만드는 게 비싸다 (청크를 전부 형태소 분석한다).
설정당 한 번만 만든다. 설정 6개 × 소스 2개면 12번이다. 오래 걸리면 --limit 로 줄여라.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))   # 평평한 import: chunking, preprocessing …
sys.path.insert(0, str(ROOT))           # config.settings

import pandas as pd

import chunking
import evaluation as ev
from config import settings
from preprocessing import load_documents
from pieces import BM25

# (자르는 법, 크기, 겹침)
SETTINGS = [
    ("recursive", 600, 0),
    ("recursive", 1000, 150),
    ("recursive", 1500, 200),
    ("recursive", 2000, 300),
    ("recursive", 3000, 400),
    ("section", 600, 100),
    ("section", 1000, 150),
    ("section", 1500, 200),
    ("section", 3000, 400),
]


def split(documents, how, size, overlap):
    fn = chunking.split_by_section if how == "section" else chunking.split_recursive
    return fn(documents, size=size, overlap=overlap)


def shared_pairs(sources, max_per_doc=2):
    """두 본문에 다 답이 있는 질문만 남긴다. (질문들, 버린 수)"""
    first = next(iter(sources.values()))
    pairs = ev.make_pairs_from_documents(first, max_per_doc=max_per_doc)
    if len(sources) == 1:
        return pairs, 0

    texts = {name: {d["meta"]["doc_id"]: d["text"] for d in docs}
             for name, docs in sources.items()}
    kept = []
    for p in pairs:
        body = [t.get(p["doc_id"], "") for t in texts.values()]
        if all(ev.matches(b, p["keywords"]) for b in body):
            kept.append(p)
    return kept, len(pairs) - len(kept)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", nargs="+", default=["documents"],
                        help="data/processed 안의 jsonl 이름 (확장자 없이)")
    parser.add_argument("--limit", type=int, help="문서 수를 줄여 먼저 시간을 잰다")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--budget", type=int, default=settings.MAX_CONTEXT_CHARS,
                        help="컨텍스트 글자 예산. 개수 대신 글자 수를 맞춰 공정하게 잰다. "
                             "0 이면 예전처럼 상위 k개로 잰다")
    parser.add_argument("--out", default=str(settings.EVAL_RESULTS / "chunking.csv"))
    args = parser.parse_args()

    sources = {name: load_documents(name) for name in args.docs}

    # 두 소스에 다 있는 문서만 남긴다. doc_id 가 없는 쪽은 비교에서 빠진다.
    # --limit 도 여기서 건다 — 소스마다 순서가 달라서 앞에서 N개씩 자르면
    # 서로 다른 문서를 비교하게 된다.
    ids = [{d["meta"]["doc_id"] for d in docs} for docs in sources.values()]
    common = sorted(set.intersection(*ids))
    if args.limit:
        common = common[: args.limit]
    keep = set(common)
    for name in sources:
        before = len(sources[name])
        sources[name] = [d for d in sources[name] if d["meta"]["doc_id"] in keep]
        docs = sources[name]
        lengths = sorted(d["chars"] for d in docs)
        print(f"{name:22} 문서 {len(docs)}건 (전체 {before}건) · "
              f"본문 {sum(lengths):,}자 · 중앙 {lengths[len(lengths) // 2]:,}자")
    if len(sources) > 1:
        print(f"{'':22} 두 소스에 다 있는 문서만 비교한다 ({len(common)}건)")

    pairs, dropped = shared_pairs(sources)
    print(f"\n질문 {len(pairs)}개 (정규식으로 자동 생성)", end="")
    print(f" · 한쪽 본문에만 있어 버린 질문 {dropped}개" if dropped else "")
    if not pairs:
        raise SystemExit("두 본문에 공통으로 답이 있는 질문이 없습니다.")

    metric = f"적중률@{args.budget}자" if args.budget else f"적중률@{args.k}"
    rows = []
    for name, docs in sources.items():
        for how, size, overlap in SETTINGS:
            t = time.time()
            chunks = split(docs, how, size, overlap)
            bm = BM25(chunks, k=10)          # 루프 안이지만 설정당 한 번뿐이라 괜찮다
            score = ev.score_all(pairs, lambda q: bm.search(q, 30),
                                 k=args.k, budget=args.budget or None)
            rows.append({
                "소스": name,
                "설정": f"{how}/{size}/{overlap}",
                "청크수": len(chunks),
                "평균길이": sum(len(c.page_content) for c in chunks) // max(len(chunks), 1),
                **score,
                "초": round(time.time() - t, 1),
            })
            tail = (f"  발췌 {score['평균청크수']}개/{score['평균글자']:,}자"
                    if "평균청크수" in score else "")
            print(f"  {name} · {how}/{size}/{overlap}  청크 {len(chunks):,}개  "
                  f"{metric} {score[metric]:.3f}  MRR {score['MRR']:.3f}{tail}  "
                  f"{time.time() - t:.0f}초")

    df = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print(df.to_string(index=False))

    print(f"\n소스별 최고 설정 ({metric})")
    for name in sources:
        best = df[df["소스"] == name].nlargest(1, metric).iloc[0]
        print(f"  {name:22} {best['설정']:20} {metric} {best[metric]:.3f} · MRR {best['MRR']:.3f}")

    if len(sources) > 1:
        print("\n같은 설정끼리 소스 비교")
        wide = df.pivot(index="설정", columns="소스", values=metric)
        wide["차이"] = (wide[args.docs[1]] - wide[args.docs[0]]).round(3)
        print(wide.to_string())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
