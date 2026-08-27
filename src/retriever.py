"""질문 → 발췌. generation 파트에 넘기는 창구.

`src/generation.py` 의 `generate_answer(model_key, query, context)` 는 `context`
를 **문자열 하나**로 받는다. 여기서 그 문자열을 만들어 준다.

    from src.retriever import retrieve_context
    from src.generation import generate_answer

    context = retrieve_context("이 사업의 예산이 얼마야?")
    result = generate_answer(model_key="mini", query="이 사업의 예산이 얼마야?",
                             context=context)

공고 하나 안에서만 찾을 때 (요약 카드):

    context = retrieve_context(질문, doc_ids=["20240330003-0"])

근거를 같이 쓰고 싶으면 두 단계로 나눈다.

    chunks = retrieve(질문)
    context = build_context(chunks)      # [1] [2] … 번호가 붙는다
    출처 = sources(chunks)                # 번호 → 공고 정보

generation 파트에 넘길 때는 파일로 뽑는다. 그쪽은 TEI 도 인덱스도 필요 없다.

    python src/retriever.py --export
    → outputs/eval_results/contexts_eval_qa.jsonl

    for row in map(json.loads, open(path, encoding="utf-8")):
        generate_answer(model_key="mini", query=row["question"], context=row["context"])

공고를 먼저 찾는 화면(1단계)이라면:

    from src.retriever import search_notices
    notices = search_notices("클라우드 전환 사업", min_budget=300_000_000)
    # → [{doc_id, title, agency, budget, bid_close_at, score, excerpt}, ...]

명령줄로 확인:

    python src/retriever.py "이 사업의 예산이 얼마야?"
    python src/retriever.py --notices "클라우드 전환" --min-budget 300000000

## 왜 이 설정인가 (2026-08-26 실측)

`data/eval_qa.json` 133문항(요구사항·배점·의역·없음)으로 일곱 조합을 쟀다.

    설정                    배점    요구사항   의역
    BM25                   0.050  0.500  0.550
    Dense                  0.150  0.725  0.550
    Dense+머리말             0.575  0.925  0.475
    Dense+머리말+Rerank      0.950  0.975  0.600   ← 전 유형 1위
    Hybrid                 0.100  0.775  0.550
    Hybrid+Rerank          0.400  0.675  0.575

- **머리말(`[사업명]`)을 붙인 인덱스**를 쓴다. 안 붙이면 배점이 0.150 으로 떨어진다.
  공고를 하나로 좁혀서(`doc_ids`) 재도 이 차이가 그대로라, 단순히 "그 공고로
  몰아주는" 효과가 아니다.
- **BM25 를 섞지 않는다.** RRF 는 순위 융합이라, 배점에서 0.050 밖에 못 찾는 BM25 가
  엉뚱한 청크를 올리면 Dense 의 좋은 1등이 밀린다 (Hybrid+머리말 0.275 < Dense+머리말 0.575).
- **리랭커가 승패를 가른다.** 배점 0.575 → 0.950. 다만 3배 느리다(0.42초 → 1.2초/질문).
- 자르기는 `recursive/1200/200`. 3000/400 은 쉬운 질문 세트로 골랐던 값이고,
  어려운 세트에서는 1200 이 이긴다 (의역 0.475 → 0.600).
"""

import argparse
import sys
from functools import lru_cache
from pathlib import Path

# 프로젝트 루트와 src/ 를 경로에 넣는다. 이래야 `python src/retriever.py` 도,
# 다른 폴더에서 `from src.retriever import ...` 도 똑같이 된다.
_ROOT = Path(__file__).resolve().parents[1]
for _folder in (_ROOT / "src", _ROOT):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

from config import settings
from evaluation import fit_budget
from models import load_embedder, load_reranker
from pieces import Dense, Pipeline, Rerank, format_context
from vectorstore import load_store

# 실측으로 고른 기본값. 바꾸려면 scripts/compare_retrieval.py 로 다시 재고 바꾼다.
CHUNKS = "cleaned_documents__recursive_1200_200"
INDEX = f"{CHUNKS}__header__tei"
POOL = 30  # 리랭커에 넘길 후보 수
TOP_K = 8  # 리랭커가 남길 수. 예산에서 다시 잘리므로 넉넉히 준다


@lru_cache(maxsize=4)
def _load(index, embed, rerank):
    """인덱스와 리랭커를 한 번만 올린다.

    질문마다 다시 올리면 FAISS 를 매번 디스크에서 읽는다. 캐시가 이걸 막는다.

    Args:
        index: 인덱스 이름.
        embed: 임베딩 종류 (tei / local / fake).
        rerank: 리랭커 종류 (tei / local / fake).

    Returns:
        `(FAISS 인덱스, 리랭커)`.
    """
    return load_store(index, load_embedder(embed)), load_reranker(rerank)


def retrieve(
    query, doc_ids=None, top_k=TOP_K, pool=POOL, index=INDEX, embed="tei", rerank="tei"
):
    """질문에 맞는 청크를 찾는다.

    Args:
        query: 사용자 질문.
        doc_ids: 주면 그 공고들 안에서만 찾는다. 요약 카드를 만들 때 쓴다.
        top_k: 리랭커가 남길 청크 수.
        pool: 리랭커에 넘길 후보 수. 크면 정확하고 느리다.
        index: 인덱스 이름. 기본값은 머리말이 붙은 것이다.
        embed: 임베딩 종류.
        rerank: 리랭커 종류.

    Returns:
        점수 순 Document 리스트. `metadata` 에 doc_id·title·agency·chunk_id 가 있다.
    """
    store, reranker = _load(index, embed, rerank)
    pipeline = Pipeline([
        Dense(store, k=pool, doc_ids=doc_ids),
        Rerank(reranker, k=top_k),
    ])
    return pipeline(query).chunks


def fit_context(chunks, budget=None):
    """예산 안에 들어가는 청크만 남긴다.

    `fit_budget` 은 본문 글자만 센다. 그런데 `format_context` 가 `[1] 사업명 ·
    발주기관` 머리와 `---` 구분선을 덧붙이므로 최종 문자열은 그보다 길어진다.
    실제로 6,000자 예산에 6,084자가 나왔다. `generation.py` 가
    `context[:MAX_CONTEXT_CHARS]` 로 뒤를 그냥 잘라내니 마지막 발췌가 중간에서
    끊긴다. 그래서 **붙인 뒤에 다시 재서** 넘치면 하나씩 뺀다.

    Args:
        chunks: `retrieve()` 가 돌려준 청크.
        budget: 최대 글자 수. 생략하면 `settings.MAX_CONTEXT_CHARS`.

    Returns:
        머리와 구분선까지 세어도 예산 안에 들어가는 청크 리스트.
        하나도 안 들어가면 첫 청크는 남긴다.
    """
    budget = budget or settings.MAX_CONTEXT_CHARS
    kept = fit_budget(chunks, budget)
    while len(kept) > 1 and len(format_context(kept)) > budget:
        kept = kept[:-1]
    return kept


def _passes(row, min_budget, max_budget, agency, closes_after):
    """공고 하나가 조건을 통과하는지.

    Args:
        row: `search_notices` 가 모으는 공고 dict.
        min_budget: 최소 사업금액(원). None 이면 안 본다.
        max_budget: 최대 사업금액(원).
        agency: 발주기관 이름 일부.
        closes_after: 이 날짜 이후 마감. `"2024-04-01"` 처럼 준다.

    Returns:
        통과하면 True.
    """
    budget = row.get("budget")
    known = isinstance(budget, (int, float)) and budget == budget  # noqa NaN 거르기
    if min_budget is not None and (not known or budget < min_budget):
        return False
    if max_budget is not None and (not known or budget > max_budget):
        return False
    if agency and agency not in str(row.get("agency") or ""):
        return False
    if closes_after and str(row.get("bid_close_at") or "") < closes_after:  # noqa
        return False
    return True


def search_notices(
    query,
    top_n=10,
    pool=200,
    min_budget=None,
    max_budget=None,
    agency=None,
    closes_after=None,
    index=INDEX,
    embed="tei",
):
    """자연어로 공고를 찾는다. **1단계 — 어떤 공고를 볼지 고르는 화면.**

    청크를 검색한 뒤 공고 단위로 묶어 점수를 합친다. 한 공고에서 여러 청크가
    상위에 들면 그만큼 점수가 올라간다(RRF). 예산·기관·마감일 같은 조건은
    임베딩이 아니라 **메타데이터로 거른다** — 숫자 비교를 벡터에 맡기면 틀린다.

    리랭커는 안 쓴다. 후보가 200개라 느리고, 공고 점수는 여러 청크의 합이라
    한 청크를 다시 채점해도 순위가 크게 안 바뀐다.

    Args:
        query: 자연어 질의. "클라우드 전환", "장애인 접근성 개선" 같은 것.
        top_n: 돌려줄 공고 수.
        pool: 훑어볼 청크 수. 크면 넓게 보고 느리다.
        min_budget: 최소 사업금액(원).
        max_budget: 최대 사업금액(원).
        agency: 발주기관 이름 일부.
        closes_after: 이 날짜 이후 마감 (`"2024-04-01"`).
        index: 인덱스 이름.
        embed: 임베딩 종류.

    Returns:
        점수 순 공고 리스트.
        `[{doc_id, title, agency, budget, bid_close_at, summary, score, 청크수, excerpt}]`
    """
    store, _ = _load(index, embed, "fake")  # 리랭커는 안 쓴다
    hits = Dense(store, k=pool).search(query, pool)

    found = {}
    for rank, chunk in enumerate(hits, 1):
        meta = chunk.metadata
        doc_id = meta.get("doc_id")
        row = found.get(doc_id)
        if row is None:
            row = found[doc_id] = {
                "doc_id": doc_id,
                "title": meta.get("title"),
                "agency": meta.get("agency"),
                "budget": meta.get("budget"),
                "bid_close_at": meta.get("bid_close_at"),
                "summary": meta.get("summary"),
                "score": 0.0,
                "청크수": 0,
                "excerpt": " ".join(chunk.page_content.split())[:200],
            }
        row["score"] += 1.0 / (60 + rank)  # RRF. 상수 60 은 관례값
        row["청크수"] += 1

    rows = [
        r
        for r in found.values()
        if _passes(r, min_budget, max_budget, agency, closes_after)
    ]
    rows.sort(key=lambda r: r["score"], reverse=True)
    for row in rows:
        row["score"] = round(row["score"], 6)
    return rows[:top_n]


def build_context(chunks, budget=None):
    """청크를 `generate_answer(context=...)` 에 넣을 문자열로 만든다.

    `[1] 사업명 · 발주기관` 머리가 붙는다. 이 번호가 곧 인용 번호다.

    Args:
        chunks: `retrieve()` 가 돌려준 청크.
        budget: 최대 글자 수. 생략하면 `settings.MAX_CONTEXT_CHARS`.

    Returns:
        발췌를 이어 붙인 문자열. 예산을 넘지 않는다.
    """
    return format_context(fit_context(chunks, budget))


def sources(chunks):
    """인용 번호 → 공고 정보. 답변에 출처를 붙일 때 쓴다.

    Args:
        chunks: `build_context()` 에 넣은 것과 **같은** 청크 리스트.

    Returns:
        `[{"n", "doc_id", "title", "agency", "chunk_id"}]`. 번호는 1부터.
    """
    return [
        {
            "n": i,
            "doc_id": chunk.metadata.get("doc_id"),
            "title": chunk.metadata.get("title"),
            "agency": chunk.metadata.get("agency"),
            "chunk_id": chunk.metadata.get("chunk_id"),
        }
        for i, chunk in enumerate(chunks, 1)
    ]


def retrieve_context(query, doc_ids=None, budget=None, **kwargs):
    """질문 하나 → 컨텍스트 문자열. generation 쪽에서 부를 한 줄.

    Args:
        query: 사용자 질문.
        doc_ids: 주면 그 공고들 안에서만 찾는다.
        budget: 최대 글자 수.
        **kwargs: `retrieve()` 의 나머지 인자.

    Returns:
        `generate_answer(context=...)` 에 그대로 넣을 문자열.
    """
    return build_context(retrieve(query, doc_ids=doc_ids, **kwargs), budget=budget)


def preview(text, query, width=220):
    """질의어가 나온 자리를 잘라 보여준다.

    **앞에서부터 자르면 안 된다.** 1,200자 청크에서 답이 뒤쪽에 있으면 앞부분만
    보고 "엉뚱한 청크"로 오판한다. 실제로 그렇게 두 번 틀렸다 — 1위 청크 끝에
    `Ⅵ 제안안내 사항 1 입찰 참가자격 …` 이 있었는데 앞 160자에는 웹 접근성
    교육 이야기만 있었다.

    Args:
        text: 청크 본문.
        query: 질문. 여기서 두 글자 이상인 토막을 뽑아 찾는다.
        width: 보여줄 길이.

    Returns:
        질의어 주변을 자른 한 줄. 못 찾으면 앞에서부터.
    """
    flat = " ".join(text.split())
    words = sorted((w.strip("?!.,'\"") for w in query.split()), key=len, reverse=True)
    for word in words:
        if len(word) < 2:
            continue
        at = flat.find(word)
        if at >= 0:
            start = max(0, at - width // 3)
            head = "…" if start else ""
            return head + flat[start:start + width]
    return flat[:width]


def export_contexts(evalset, out_path, **kwargs):
    """평가 질문마다 발췌를 뽑아 파일로 저장한다.

    **generation 파트가 검색을 안 돌려도 되게 하려는 것이다.** 브랜치를 가져갈
    필요도, TEI 를 띄울 필요도 없다. jsonl 한 줄이 `generate_answer()` 한 번에
    그대로 들어간다.

    Args:
        evalset: `data/` 의 평가 세트 이름.
        out_path: 저장 경로.
        **kwargs: `retrieve()` 인자 (index, embed, rerank, top_k …).

    Returns:
        저장한 줄 수.
    """
    import json
    import time

    from evaluation import load_evalset

    pairs = load_evalset(evalset)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    with open(out_path, "w", encoding="utf-8") as f:
        for i, pair in enumerate(pairs, 1):
            doc_ids = [pair["doc_id"]] if pair.get("doc_id") else None
            chunks = fit_context(retrieve(pair["question"], doc_ids=doc_ids, **kwargs))
            context = format_context(chunks)
            f.write(json.dumps({
                "qid": f"{pair.get('type', 'q')}-{i:03d}",
                "question": pair["question"],
                "type": pair.get("type"),
                "answerable": pair.get("answerable", True),
                "doc_ids": doc_ids,
                "keywords": pair.get("keywords"),   # 검색 정답. 채점 참고용
                "context": context,
                "sources": sources(chunks),
                "chunks": len(chunks),
                "chars": len(context),
            }, ensure_ascii=False) + "\n")
            print(f"  {i}/{len(pairs)}", end="\r")

    print(" " * 30, end="\r")
    print(f"질문 {len(pairs)}개 · {time.time() - started:.0f}초 → {out_path}")
    return len(pairs)


def main():
    """명령줄에서 검색 결과를 눈으로 확인한다."""
    parser = argparse.ArgumentParser(description="질문을 넣고 무엇이 뽑히는지 본다.")
    parser.add_argument(
        "query",
        nargs="?",
        help="질문 (--notices 와 함께면 공고 찾기 질의, --export 면 생략)",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="평가 질문 전체의 발췌를 파일로 뽑는다 (generation 전달용)",
    )
    parser.add_argument("--evalset", default="eval_qa", help="data/ 의 평가 세트 이름")
    parser.add_argument("--out", help="--export 저장 경로")
    parser.add_argument(
        "--notices",
        action="store_true",
        help="공고를 찾는다 (1단계). 안 주면 발췌를 찾는다 (2단계)",
    )
    parser.add_argument("--min-budget", type=float)
    parser.add_argument("--max-budget", type=float)
    parser.add_argument("--agency")
    parser.add_argument("--closes-after")
    parser.add_argument(
        "--doc",
        action="append",
        dest="doc_ids",
        help="공고를 좁힌다. 여러 번 줄 수 있다",
    )
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--index", default=INDEX)
    parser.add_argument("--embed", default="tei", choices=["tei", "local", "fake"])
    parser.add_argument("--rerank", default="tei", choices=["tei", "local", "fake"])
    args = parser.parse_args()

    if args.export:
        out = args.out or settings.EVAL_RESULTS / f"contexts_{args.evalset}.jsonl"
        export_contexts(args.evalset, out, top_k=args.top_k, index=args.index,
                        embed=args.embed, rerank=args.rerank)
        return

    if not args.query:
        parser.error("질문을 주거나 --export 를 쓰세요.")

    if args.notices:
        notices = search_notices(
            args.query,
            min_budget=args.min_budget,
            max_budget=args.max_budget,
            agency=args.agency,
            closes_after=args.closes_after,
            index=args.index,
            embed=args.embed,
        )
        print(f"공고 {len(notices)}건\n")
        for i, notice in enumerate(notices, 1):
            budget = notice["budget"]
            known = isinstance(budget, (int, float)) and budget == budget  # noqa
            money = f"{budget:,.0f}원" if known else "미상"
            print(f"[{i}] {notice['title']}")
            print(f"    {notice['agency']} · {money} · 마감 {notice['bid_close_at']}")
            print(
                f"    점수 {notice['score']} (청크 {notice['청크수']}개)  {notice['doc_id']}"
            )
            print(f"    {notice['excerpt'][:110]}")
        print("\n다음:  retrieve_context(질문, doc_ids=[위 doc_id])")
        return

    chunks = retrieve(
        args.query,
        doc_ids=args.doc_ids,
        top_k=args.top_k,
        index=args.index,
        embed=args.embed,
        rerank=args.rerank,
    )
    kept = fit_context(chunks)
    context = format_context(kept)

    print(f"찾은 청크 {len(chunks)}개 · 예산 안에 {len(kept)}개 · {len(context):,}자\n")
    for source, chunk in zip(sources(kept), kept, strict=False):
        score = chunk.metadata.get("score")
        mark = f"{score:.4f}" if isinstance(score, float) else "-"
        print(f"[{source['n']}] {mark}  {source['title']} · {source['agency']}")
        print(f"    {preview(chunk.page_content, args.query)}")
    print(
        f"\ngenerate_answer(model_key='mini', query=..., context=...) 에 넣을 "
        f"문자열 {len(context):,}자"
    )


if __name__ == "__main__":
    main()
