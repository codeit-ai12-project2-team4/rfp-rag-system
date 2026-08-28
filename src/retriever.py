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

## 왜 이 설정인가 (2026-08-28 실측)

팀원이 만든 `data/eval_qa_80.json` 으로 여덟 조합을 쟀다. MRR 이다.
공고를 한정한 경우(scoped)가 실제 UI 2단계 조건이다.

    설정                       배점   요구사항   의역
    BM25                     0.611  0.704  0.618
    Dense                    0.608  0.648  0.586
    Dense+머리말               0.633  0.694  0.624
    Dense+머리말+Rerank        0.667  0.722  0.679
    Hybrid                   0.621  0.711  0.632
    Hybrid+Rerank            0.867  0.778  0.712   ← 전 유형 1위
    Hybrid+머리말              0.686  0.705  0.650

- **BM25 를 섞는다.** RFP 는 글자의 60~80%가 표 안에 있고, 표가 많은 문서에서는
  어휘 매칭이 강하다는 게 문헌과도 맞는다. 단독으로도 Dense 를 이긴다.
- **머리말(`[사업명]`)은 안 쓴다.** BM25 와 섞을 때는 머리말 없는 인덱스가 낫다.
  예전에 머리말이 결정적으로 보였던 건 그때 평가 질문이 100% 「사업명」으로
  시작했기 때문이다. 팀원 세트는 13% 뿐이고 실사용도 그쪽에 가깝다.
- **리랭커가 승패를 가른다.** 배점 0.621 → 0.867. 3배 느리다.
- 공고를 한정하지 않으면 요구사항만 Dense+머리말+Rerank 가 낫다(0.722 vs 0.648).
  BM25 가 `SFR` 같은 공용 어휘로 엉뚱한 공고를 끌어오기 때문이다. 2단계는
  이미 공고가 정해져 있으니 문제되지 않는다.
- 자르기는 `recursive/1200/200`, 전처리본은 마크다운 표 문법을 걷어낸 것.

**남은 문제** — 의역 40문항 중 못 찾는 14개는 순위 문제가 아니다. 후보밖 8개
(정답 청크가 후보 30개에 못 듦), 정답없음 6개(청크 경계가 정답을 자름).
순위밀림은 0개다. parent-child 청킹이 다음 후보다.
"""

import argparse
import sys
import time
from functools import lru_cache
from pathlib import Path

# 프로젝트 루트와 src/ 를 경로에 넣는다. 이래야 `python src/retriever.py` 도,
# 다른 폴더에서 `from src.retriever import ...` 도 똑같이 된다.
_ROOT = Path(__file__).resolve().parents[1]
for _folder in (_ROOT / "src", _ROOT):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import chunking
from config import settings
from evaluation import fit_budget
from models import load_embedder, load_reranker
from pieces import BM25, Dense, Hybrid, Pipeline, Rerank, State
from vectorstore import load_store

# 실측으로 고른 기본값. 바꾸려면 scripts/compare_retrieval.py 로 다시 재고 바꾼다.
CHUNKS = "cleaned_documents_strip__recursive_1200_200"
INDEX = f"{CHUNKS}__tei"  # 머리말 없는 쪽. BM25 와 섞을 때는 이게 낫다
POOL = 30  # 리랭커에 넘길 후보 수. 10/30/60/100 을 재고 고른 값이다 —
#          10 은 부족하고(배점 0.850), 60·100 은 30 과 ±1문항 차이인데 시간만 2~3배다.
TOP_K = 8  # 리랭커가 남길 수. 예산에서 다시 잘리므로 넉넉히 준다


@lru_cache(maxsize=2)
def _store(index, embed):
    """FAISS 인덱스만 연다. 공고 찾기(1단계)는 이것만 있으면 된다."""
    return load_store(index, load_embedder(embed))


@lru_cache(maxsize=2)
def _load(index, chunks, embed, rerank):
    """인덱스·청크·리랭커를 한 번만 올린다.

    질문마다 다시 올리면 FAISS 를 매번 디스크에서 읽고 BM25 를 다시 짓는다.
    **BM25 가 비싸다** — 청크 9,500개를 형태소 분석해야 해서 수십 초 걸리고
    메모리도 수백 MB 다. 그래서 첫 호출만 느리고 그 뒤로는 캐시가 받는다.
    서비스에서는 뜰 때 한 번 불러 두는 게 낫다.

    Args:
        index: FAISS 인덱스 이름.
        chunks: BM25 가 쓸 청크 이름 (머리말 없는 쪽).
        embed: 임베딩 종류 (tei / local / fake).
        rerank: 리랭커 종류 (tei / local / fake).

    Returns:
        `(FAISS 인덱스, 청크 리스트, 리랭커)`.
    """
    started = time.time()
    store = _store(index, embed)
    chunk_list = chunking.load_chunks(chunks)
    BM25(chunk_list, k=POOL)  # 여기서 색인을 지어 캐시에 넣는다
    reranker = load_reranker(rerank)
    print(f"검색기 준비 {time.time() - started:.1f}초 (청크 {len(chunk_list):,}개)")
    return store, chunk_list, reranker


def retrieve(
    query,
    doc_ids=None,
    top_k=TOP_K,
    pool=POOL,
    index=INDEX,
    chunks=CHUNKS,
    embed="tei",
    rerank="tei",
):
    """질문에 맞는 청크를 찾는다.

    Args:
        query: 사용자 질문.
        doc_ids: 주면 그 공고들 안에서만 찾는다. 요약 카드를 만들 때 쓴다.
        top_k: 리랭커가 남길 청크 수.
        pool: 리랭커에 넘길 후보 수. 크면 정확하고 느리다.
        index: FAISS 인덱스 이름.
        chunks: BM25 가 쓸 청크 이름.
        embed: 임베딩 종류.
        rerank: 리랭커 종류.

    Returns:
        점수 순 Document 리스트. `metadata` 에 doc_id·title·agency·chunk_id 가 있다.
    """
    store, chunk_list, reranker = _load(index, chunks, embed, rerank)
    # BM25 는 같은 청크 묶음이면 색인을 돌려쓴다. 그래서 질문마다 만들어도 싸다.
    pipeline = Pipeline([
        Hybrid(
            [
                Dense(store, k=pool, doc_ids=doc_ids),
                BM25(chunk_list, k=pool, doc_ids=doc_ids),
            ],
            k=pool,
            pool=pool,
        ),
        Rerank(reranker, k=top_k),
    ])
    return pipeline(query).chunks


def format_context(chunks):
    """청크를 번호 붙여 프롬프트용 문자열로 잇는다.

    이 `[1] [2]` 번호가 그대로 인용 번호가 된다. `sources()` 가 돌려주는
    `n` 과 짝이 맞으므로, 답변에 달린 번호로 출처를 되짚을 수 있다.

    Args:
        chunks: `retrieve()` 가 돌려준 청크.

    Returns:
        `[1] 사업명 · 발주기관 · 절제목` 머리를 붙이고 `---` 로 이은 문자열.
    """
    blocks = []
    for i, chunk in enumerate(chunks, 1):
        title = chunk.metadata.get("title", "")
        agency = chunk.metadata.get("agency", "")
        section = chunk.metadata.get("section", "")
        head = f"[{i}] {title} · {agency}"
        if section:
            head += f" · {section}"
        blocks.append(head + "\n" + chunk.page_content)
    return "\n\n---\n\n".join(blocks)


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
    chunks=CHUNKS,
    rerank=None,
):
    """자연어로 공고를 찾는다. **1단계 — 어떤 공고를 볼지 고르는 화면.**

    청크를 검색한 뒤 공고 단위로 묶어 점수를 합친다. 한 공고에서 여러 청크가
    상위에 들면 그만큼 점수가 올라간다(RRF). 예산·기관·마감일 같은 조건은
    임베딩이 아니라 **메타데이터로 거른다** — 숫자 비교를 벡터에 맡기면 틀린다.

    **리랭커는 안 쓴다 (2026-08-28 실측, 62문항).** 이 화면은 사람이 목록에서
    고르므로 1위 정확도보다 **목록 안에 있는지(Top10)** 가 중요하다.

        설정            MRR   Top1  Top10   질문당
        Dense          0.633 0.532 0.806   0.4초
        Hybrid         0.663 0.565 0.839   0.7초   ← 채택
        Dense+Rerank   0.687 0.613 0.839   3.0초
        Hybrid+Rerank  0.680 0.581 0.871   3.3초

    리랭커는 Top10 을 2문항 더 올리는 대신 질문당 3초를 더 쓴다. 검색창에서
    3초는 못 쓴다. BM25 를 섞는 건 0.3초라 그건 켠다.
    2단계(`retrieve`)는 모델이 직접 골라야 하므로 리랭커를 쓴다 — 거기선 순위가
    곧 답이다. 다시 재려면 `scripts/eval_notices.py`.

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
        chunks: BM25 가 쓸 청크 이름. None 이면 Dense 만 쓴다.
        rerank: 리랭커 종류 (tei / local). 주면 묶기 전에 다시 채점한다.
            기본은 끔 — 3초가 더 든다.

    Returns:
        점수 순 공고 리스트.
        `[{doc_id, title, agency, budget, bid_close_at, summary, score, 청크수, excerpt}]`
    """
    store = _store(index, embed)
    searcher = Dense(store, k=pool)
    if chunks:
        searcher = Hybrid(
            [searcher, BM25(chunking.load_chunks(chunks), k=pool)], k=pool, pool=pool
        )
    hits = searcher.search(query, pool)
    if rerank:
        hits = Rerank(load_reranker(rerank), k=pool)(
            State(question=query, chunks=hits)
        ).chunks

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
            return head + flat[start : start + width]
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
            f.write(
                json.dumps(
                    {
                        "qid": f"{pair.get('type', 'q')}-{i:03d}",
                        "question": pair["question"],
                        "type": pair.get("type"),
                        "answerable": pair.get("answerable", True),
                        "doc_ids": doc_ids,
                        "keywords": pair.get("keywords"),  # 검색 정답. 채점 참고용
                        "context": context,
                        "sources": sources(chunks),
                        "chunks": len(chunks),
                        "chars": len(context),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
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
    parser.add_argument("--chunks", default=CHUNKS, help="BM25 가 쓸 청크 이름")
    parser.add_argument("--embed", default="tei", choices=["tei", "local", "fake"])
    parser.add_argument("--rerank", default="tei", choices=["tei", "local", "fake"])
    args = parser.parse_args()

    if args.export:
        out = args.out or settings.EVAL_RESULTS / f"contexts_{args.evalset}.jsonl"
        export_contexts(
            args.evalset,
            out,
            top_k=args.top_k,
            index=args.index,
            chunks=args.chunks,
            embed=args.embed,
            rerank=args.rerank,
        )
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
        chunks=args.chunks,
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
