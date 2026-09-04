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

## 왜 이 설정인가 (9/8 기준. 상세는 `docs/tries/retrieval/`)

2단계 scoped 가중평균 MRR. 코퍼스를 v5→v6→v7 로 갈아엎어도 리랭커를 통과하면
0.92x 로 수렴한다 — **리랭커 바운드**다. 앞단이 0.08 벌어져도 결과가 같다.

    설정                        v5     v6     v7
    BM25                      0.783  0.720  0.700
    Dense                     0.720  0.710  0.701
    Hybrid                    0.779  0.760  0.729
    Hybrid+Rerank             0.919  0.919  0.919
    용어추가+Hybrid+Rerank      0.924  0.921  0.924   ← 채택

    pool 80 (9/3, 30 → 80 이 그 주 최대 개선 +0.043)
    BM25 가중치 0.5 (0.7·0.9 와 동일 — RRF 는 어느 지점부터 순위가 포화)
    Splade 제외 (9/4·9/5 두 번 결정. 리랭커를 붙이면 -0.006, 운영 비용만 는다)

**발표 숫자**  1단계 Top10 0.717 · Top1 0.414 / 2단계 scoped MRR 0.924
**병목은 1단계다.** 2단계와 생성은 천장인데 공고 검색에서 28% 를 잃는다.

---

아래는 8/28 옛 세트(62문항) 기록이다. **9/1 에 이 세트를 폐기했다** — 사업명이
질문에 그대로 있어 BM25 를 부풀렸고, pool 30 안에 정답이 늘 있어 상류 차이가
안 보였다("눈이 먼 도구"). 절대값을 지금 숫자와 비교하지 말 것.

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
from config import retrieval as cfg
from config import settings
from evaluation import body, fit_budget
from models import load_embedder, load_reranker
from pieces import BM25, Dense, Hybrid, Pipeline, Rerank, State
from vectorstore import load_store

# 실측으로 고른 기본값. 바꾸려면 scripts/compare_retrieval.py 로 다시 재고 바꾼다.
# **설정은 config/settings.py 한 곳에만 있다.** 여기에 상수를 다시 적으면
# 그게 굳어서, settings 를 고쳐도 API 는 옛 코퍼스로 답하게 된다. 실제로 그랬다.
CHUNKS = cfg.chunk_name()
INDEX = cfg.index_name()
POOL = cfg.POOL
TOP_K = cfg.TOP_K


@lru_cache(maxsize=2)
def _store(index, embed):
    """벡터 저장소만 연다. 공고 찾기(1단계)는 이것만 있으면 된다.

    `STORE=lance` 면 LanceDB 를 연다. 둘 다 `similarity_search(query, k)` 하나만
    있으면 되므로 `Dense` 부품은 어느 쪽인지 모른다.
    """
    if cfg.STORE == "lance":
        import lance_store

        return lance_store.load_store(index, load_embedder(embed))
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


def money(value):
    """`150000000.0` → `1억 5,000만원`. 못 읽으면 원래 문자열 그대로.

    프롬프트에 `150000000.0` 을 그대로 넣으면 모델이 자릿수를 자주 틀린다.
    `.0` 은 pandas 가 숫자로 읽어서 붙는 꼬리다.

    Args:
        value: 금액 문자열이나 숫자.

    Returns:
        str: 사람이 읽는 꼴.
    """
    try:
        won = int(float(str(value).replace(",", "")))
    except (TypeError, ValueError):
        return str(value)
    eok, man = divmod(won, 10**8)
    man //= 10**4
    if not eok and not man:
        return f"{won:,}원"
    # 억만 있고 만 자리가 0 이면 `15억 원` 처럼 공백이 남는다. 붙여서 짓는다.
    return " ".join(
        part for part in (f"{eok}억" if eok else "", f"{man:,}만" if man else "")
        if part
    ) + "원"


def format_context(chunks, generation=False):
    """청크를 번호 붙여 프롬프트용 문자열로 잇는다.

    이 `[1] [2]` 번호가 그대로 인용 번호가 된다. `sources()` 가 돌려주는
    `n` 과 짝이 맞으므로, 답변에 달린 번호로 출처를 되짚을 수 있다.

    Args:
        chunks: `retrieve()` 가 돌려준 청크.
        generation: True 면 표 마크업이 살아 있는 생성용 본문을 쓴다.
            검색은 마크업 없는 쪽으로 하고 프롬프트에는 있는 쪽을 넣는
            A/B 를 이 인자 하나로 켠다.

    Returns:
        `[1] 사업명 · 발주기관 · 공고번호 · 마감 …` 머리를 붙이고 `---` 로 이은 문자열.
    """
    blocks = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.metadata
        # **NaN 을 걸러야 한다.** 메타데이터가 pandas 에서 오므로 빈 칸이
        # float('nan') 이다. 그대로 f-string 에 넣으면 프롬프트에 "nan" 이
        # 박히고, 모델이 그걸 사업명으로 읽는다.
        def value(key):
            got = meta.get(key)
            return "" if got is None or got != got else str(got).strip()

        # 사업명·발주기관만 넣고 있었다. 컨설턴트가 출처를 되짚을 때 필요한
        # 건 공고번호와 마감일과 금액이다. 전부 CSV(나라장터 API) 값이다.
        #
        # **금액은 `배정예산 1억 5,000만원` 처럼 어느 쪽인지 붙여서만 넣는다.**
        # 배정예산은 발주처가 잡아 둔 돈이고 추정가격은 조달청 산정치라 뜻이
        # 다르다. 라벨 없이 "사업금액 …" 이라고만 주면 모델은 그대로 옮겨 쓰고,
        # 읽는 사람은 어느 쪽인지 모른 채 숫자만 받는다 — 컨설턴트가 제일
        # 먼저 확인하는 값이라 그게 제일 나쁘다. 모델이 라벨을 못 잡는 게
        # 아니라, **우리가 그 구분을 안 갖고 있었다**(크롤러가 버렸다).
        #
        # 9/3 실측(135건): 본문과 일치 63% · 불일치 13% · **본문에 금액 표기가
        # 아예 없음 19%.** 불일치 18건은 틀린 값이 아니라 기준이 다른 값이다 —
        # 본문은 부가세 포함액과 공급가액(÷1.1)을 나란히 적고, CSV 값은 그보다
        # 0.6~1.0% 크다(한 건은 정확히 ×1.1). 라벨이 있으면 읽는 사람이 가른다.
        # 넣는 진짜 이유는 19% 다 — 그 문서들은 머리가 유일한 출처다.
        # 구분을 모르는 행(2026-09-03 이전에 받은 135건)은 `공고 금액` 으로 적는다.
        # 라벨 없이 숫자만 주는 것보다 낫고, 모르는 걸 아는 척하지도 않는다.
        #
        # `section` 은 뺐다. `split_by_section` 으로 자른 청크에만 있는데
        # 전처리팀 파이프라인은 recursive 라 **늘 비어 있었다.**
        close = value("bid_close_at")[:10]
        amount = value("budget")
        kind = value("budget_kind") or "공고 금액"
        parts = [
            value("title"),
            value("agency"),
            value("notice_no"),
            f"마감 {close}" if close else "",
            f"{kind} {money(amount)}" if amount else "",
        ]
        # `[n]` 은 붙여 쓴다. 이 번호가 그대로 인용 번호이고 `sources()` 의
        # n 과 짝이 맞아야 한다.
        head = (f"[{i}] " + " · ".join(part for part in parts if part)).rstrip()
        blocks.append(head + "\n" + body(chunk, generation))
    return "\n\n---\n\n".join(blocks)


def fit_context(chunks, budget=None, generation=False):
    """예산 안에 들어가는 청크만 남긴다.

    `fit_budget` 은 본문 글자만 센다. 그런데 `format_context` 가 `[1] 사업명 ·
    발주기관` 머리와 `---` 구분선을 덧붙이므로 최종 문자열은 그보다 길어진다.
    실제로 6,000자 예산에 6,084자가 나왔다. `generation.py` 가
    `context[:MAX_CONTEXT_CHARS]` 로 뒤를 그냥 잘라내니 마지막 발췌가 중간에서
    끊긴다. 그래서 **붙인 뒤에 다시 재서** 넘치면 하나씩 뺀다.

    Args:
        chunks: `retrieve()` 가 돌려준 청크.
        budget: 최대 글자 수. 생략하면 `settings.MAX_CONTEXT_CHARS`.
        generation: True 면 생성용 본문 길이로 잰다. 프롬프트에 들어가는
            것이 그쪽이므로 여기서 재야 예산이 맞는다.

    Returns:
        머리와 구분선까지 세어도 예산 안에 들어가는 청크 리스트.
        하나도 안 들어가면 첫 청크는 남긴다.
    """
    budget = budget or settings.MAX_CONTEXT_CHARS
    kept = fit_budget(chunks, budget, generation)
    while len(kept) > 1 and len(format_context(kept, generation)) > budget:
        kept = kept[:-1]
    return kept


def _drop_nan(row):
    """dict 안의 NaN 을 None 으로 바꾼다.

    메타데이터가 pandas 에서 와서 값이 비면 float('nan') 이 섞여 나온다.
    Starlette 의 JSONResponse 는 `allow_nan=False` 라 NaN 하나에 응답 전체가
    끊긴다. 500 도 아니고 연결이 그냥 끊겨서 브라우저에는 "Failed to fetch"
    로만 보인다 — 원인을 찾는 데 한참 걸린다.

    Args:
        row: 메타데이터에서 만든 dict.

    Returns:
        NaN 자리가 None 으로 바뀐 새 dict.
    """
    return {k: None if isinstance(v, float) and v != v else v for k, v in row.items()}


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


def _plain(value):
    """JSON 으로 나갈 수 있는 값으로 바꾼다.

    예산이 없는 공고가 있어서 `budget` 이 NaN 으로 온다. 파이썬 `json` 은
    기본으로 NaN 을 통과시키지만 **Starlette 은 `allow_nan=False`** 라
    ValueError 로 죽는다. 응답을 만드는 중이라 스택만 남고 원인이 안 보인다.

    Args:
        value: 메타데이터 값.

    Returns:
        NaN 이면 None, 아니면 그대로.
    """
    return None if isinstance(value, float) and value != value else value


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

    **리랭커는 안 쓴다. 다만 그 근거는 만료됐다 — 아직 다시 안 쟀다.**

    뺀 근거는 8/28 옛 세트(62문항) 하나뿐이다. 이 화면은 사람이 목록에서
    고르므로 1위 정확도보다 목록 안에 있는지(Top10)가 중요한데, 리랭커가
    Top10 을 2문항 올리는 대신 질문당 3초를 더 썼다.

        설정            MRR   Top1  Top10   질문당      ← 폐기된 세트의 숫자
        Dense          0.633 0.532 0.806   0.4초
        Hybrid         0.663 0.565 0.839   0.7초   ← 채택
        Dense+Rerank   0.687 0.613 0.839   3.0초
        Hybrid+Rerank  0.680 0.581 0.871   3.3초

    **그런데 9/1 에 이 세트를 폐기했다.** pool 30 안에 정답이 늘 들어 있어
    상류가 무엇을 주든 결과가 같았다 — 리랭커 유무를 판정할 수 없는 도구였다.
    9/5 "검토 중" 에 "1단계 재측정, 리랭커를 넣을지도 다시 판단" 이 그대로 남아
    있고, 9/8 현재 1단계는 Top10 0.717 · Top1 0.414 로 **전 구간의 병목**이다.

    현재 세트로 다시 재는 명령은 아래 한 줄이다. 재고 나서 정한다.

        python scripts/retrieval/eval_notices.py \
            --chunks $CHUNKS --evalset $EVALSET --pool 80

    2단계(`retrieve`)는 모델이 직접 골라야 하므로 리랭커를 쓴다 — 거기선 순위가
    곧 답이다.

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
            row = found[doc_id] = _drop_nan(
                {
                    "doc_id": doc_id,
                    "title": _plain(meta.get("title")),
                    "agency": _plain(meta.get("agency")),
                    "budget": _plain(meta.get("budget")),
                    "bid_close_at": _plain(meta.get("bid_close_at")),
                    "summary": _plain(meta.get("summary")),
                    "score": 0.0,
                    "청크수": 0,
                    "excerpt": " ".join(chunk.page_content.split())[:200],
                }
            )
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


@lru_cache(maxsize=1)
def _files():
    """doc_id → `data/raw` 의 원본 파일. **CSV 가 기준이다.**

    청크 메타에는 파일명이 없고, 있더라도 믿으면 안 된다 — 사용자가 준
    문자열로 경로를 만들면 상위 폴더로 빠져나갈 수 있다. **CSV 에 적힌 이름만**
    쓰면 그 문제가 통째로 없어진다.

    파일명 규칙이 두 가지로 섞여 있다. 처음 받은 100건은 `{기관}_{사업명}.hwp`,
    크롤러가 받은 건 `{doc_id}.hwp`. 그래서 규칙으로 만들지 않고 표를 읽는다.
    """
    table = {}
    if not settings.META_CSV.exists():
        return table
    import csv

    with open(settings.META_CSV, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            number = (row.get("공고 번호") or "").strip().removesuffix(".0")
            name = (row.get("파일명") or "").strip()
            if not number or not name:
                continue
            try:
                order = int(float((row.get("공고 차수") or "0").strip() or 0))
            except ValueError:
                order = 0
            table[f"{number}-{order}"] = (name, (row.get("사업명") or "").strip())
    return table


def file_for(doc_id):
    """공고 하나의 원본 파일. `(경로, 내려줄 이름)` 또는 None.

    내려줄 이름은 사업명으로 만든다 — `20241001798-0.hwp` 를 받으면 뭔지 모른다.

    Returns:
        (Path, str) 또는 None (표에 없거나 파일이 사라졌을 때).
    """
    found = _files().get(str(doc_id))
    if not found:
        return None
    stored, title = found
    path = (settings.RAW / stored).resolve()
    # CSV 에 적힌 이름만 쓰므로 원래 못 빠져나가지만, 경계에서 한 번 더 본다.
    if not path.is_file() or settings.RAW.resolve() not in path.parents:
        return None
    safe = "".join(c for c in (title or path.stem) if c not in '/\\:*?"<>|').strip()
    return path, f"{safe or path.stem}{path.suffix}"


def build_context(chunks, budget=None, generation=True):
    """청크를 `generate_answer(context=...)` 에 넣을 문자열로 만든다.

    `[1] 사업명 · 발주기관` 머리가 붙는다. 이 번호가 곧 인용 번호다.

    **기본이 생성용 본문이다.** 9/8 A/B 에서 표 구조를 살린 쪽이 충실성
    0.963 대 0.937 로 이겼다 (191문항, 3칸 중 2칸). 대가는 발췌 3.52 → 3.27 개다.
    청크에 생성용 본문이 없으면 `body()` 가 검색용으로 알아서 떨어진다.

    Args:
        chunks: `retrieve()` 가 돌려준 청크.
        budget: 최대 글자 수. 생략하면 `settings.MAX_CONTEXT_CHARS`.
        generation: False 면 검색용 평문을 넣는다. A/B 를 다시 잴 때만 쓴다.

    Returns:
        발췌를 이어 붙인 문자열. 예산을 넘지 않는다.
    """
    return format_context(fit_context(chunks, budget, generation), generation)


@lru_cache(maxsize=1)
def _notices(chunks=None):
    """doc_id → 공고 한 건. 청크 메타에서 한 번만 모은다.

    **화면이 sessionStorage 에만 기대면 안 된다.** 지금 공고 화면은 목록에서
    넘겨준 값을 세션에 담아 읽는데, 그러면 새로고침·주소 직접 입력·답변의
    출처를 눌러 들어온 경우에 제목도 요약도 빈다. 실제로 "어떤 건 요약이
    보이고 어떤 건 안 보인다" 로 나타났다. 서버가 주면 그 경우가 없어진다.

    Args:
        chunks: 청크 이름. 생략하면 config 기본값.

    Returns:
        dict: doc_id → `search_notices` 와 같은 모양의 dict.
    """
    found = {}
    for chunk in chunking.load_chunks(chunks or CHUNKS):
        doc_id = str(chunk.metadata.get("doc_id") or "")
        if not doc_id or doc_id in found:
            continue
        meta = chunk.metadata
        found[doc_id] = _drop_nan({
            "doc_id": doc_id,
            "title": _plain(meta.get("title")),
            "agency": _plain(meta.get("agency")),
            "budget": _plain(meta.get("budget")),
            "budget_kind": _plain(meta.get("budget_kind")),
            "bid_close_at": _plain(meta.get("bid_close_at")),
            "summary": _plain(meta.get("summary")),
            "score": 0.0,
            "청크수": 0,
            # **요약이 없는 공고가 많다.** `사업 요약` 은 나라장터 API 에 없는
            # 값이라 크롤러가 빈칸으로 둔다(처음 받은 100건만 사람이 넣었다).
            # 그러면 공고 화면이 제목만 있고 텅 빈다. 본문 첫 대목이라도 준다 —
            # 없는 값을 지어내는 것보다 원문 한 조각을 보여주는 게 맞다.
            "excerpt": " ".join(chunk.page_content.split())[:400],
        })
    for chunk in chunking.load_chunks(chunks or CHUNKS):
        row = found.get(str(chunk.metadata.get("doc_id") or ""))
        if row is not None:
            row["청크수"] += 1
    return found


def notice(doc_id, chunks=None):
    """공고 하나. 없으면 None."""
    return _notices(chunks).get(str(doc_id))


def sources(chunks):
    """인용 번호 → 공고 정보. 답변에 출처를 붙일 때 쓴다.

    Args:
        chunks: `build_context()` 에 넣은 것과 **같은** 청크 리스트.

    Returns:
        `[{"n", "doc_id", "title", "agency", "chunk_id", "excerpt"}]`. 번호는 1부터.
    """
    return [
        _drop_nan(
            {
                "n": i,
                "doc_id": chunk.metadata.get("doc_id"),
                "title": chunk.metadata.get("title"),
                "agency": chunk.metadata.get("agency"),
                "chunk_id": chunk.metadata.get("chunk_id"),
                # **근거로 쓴 원문.** 이게 없으면 화면이 제목만 보여주게 되고,
                # 그러면 "이 답이 어디서 나왔나" 를 확인할 방법이 사라진다.
                # 이 제품에서 그건 기능 하나가 아니라 존재 이유다.
                #
                # 검색용 본문을 준다. 모델이 읽은 건 생성용(표 구조 유지)이지만
                # 내용은 같고 표 마크업이 없어 사람이 읽기 좋다.
                "excerpt": " ".join(chunk.page_content.split())[:400],
            }
        )
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


def export_contexts(evalset, out_path, generation=False, on_progress=None,
                    scoped=True, **kwargs):
    """평가 질문마다 발췌를 뽑아 파일로 저장한다.

    **generation 파트가 검색을 안 돌려도 되게 하려는 것이다.** 브랜치를 가져갈
    필요도, TEI 를 띄울 필요도 없다. jsonl 한 줄이 `generate_answer()` 한 번에
    그대로 들어간다.

    **`scoped` 가 무엇을 재는지 가른다.**

        scoped=True   문항의 `doc_id` 로 공고를 고정하고 그 안에서만 찾는다.
                      1단계를 건너뛴다. "공고가 정해진 뒤의 E2E" 다.
        scoped=False  질문만 주고 검색부터 시킨다. **전 구간이다.**
                      실제 사용자 흐름과 같다.

    `scoped=True` 로 낸 숫자를 그냥 "E2E" 라고 쓰면 실제보다 후하다 — 1단계
    Top10 을 곱해야 전 구간이 된다. `scoped=False` 는 그 곱을 직접 잰다.

    `scoped=False` 일 때는 각 줄에 `found_doc` 을 적는다. 정답 공고의 청크가
    발췌에 하나라도 들었는가다. **틀린 답이 "공고를 못 찾아서" 인지 "찾았는데
    못 읽어서" 인지 이걸로 갈린다.** 없으면 전 구간 숫자만 보고 어디를 고쳐야
    할지 알 수 없다.

    Args:
        evalset: `data/` 의 평가 세트 이름.
        out_path: 저장 경로.
        on_progress: `(끝난 개수, 전체)` 로 부른다. UI 가 진행률을 보여줄 때 쓴다.
            `\r` 로 덮어쓰는 화면 출력은 파일 로그에서 한 줄로 뭉개진다.
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
    empty = 0  # 발췌가 하나도 안 잡힌 문항. doc_id 가 코퍼스와 다르면 전부 여기로 온다
    with open(out_path, "w", encoding="utf-8") as f:
        for i, pair in enumerate(pairs, 1):
            gold = pair.get("doc_id")
            doc_ids = [gold] if (scoped and gold) else None
            chunks = fit_context(
                retrieve(pair["question"], doc_ids=doc_ids, **kwargs), generation=generation
            )
            context = format_context(chunks, generation)
            f.write(
                json.dumps(
                    {
                        "qid": f"{pair.get('type', 'q')}-{i:03d}",
                        "question": pair["question"],
                        "type": pair.get("type"),
                        "answerable": pair.get("answerable", True),
                        "doc_ids": doc_ids,
                        "gold_doc": gold,
                        # scoped 면 늘 True 라 뜻이 없다. unscoped 에서만 적는다.
                        "found_doc": None if scoped else any(
                            c.metadata.get("doc_id") == gold for c in chunks
                        ),
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
            if not chunks:
                empty += 1
            print(f"  {i}/{len(pairs)}", end="\r")
            if on_progress:
                on_progress(i, len(pairs))

    print(" " * 30, end="\r")
    print(f"질문 {len(pairs)}개 · {time.time() - started:.0f}초 → {out_path}")
    if not scoped:
        print(f"※ unscoped — 1단계부터 잽니다. 전 구간 숫자입니다")
    if empty:
        # 빈 발췌로 답변을 만들면 모델은 "확인되지 않습니다" 밖에 못 낸다.
        # 채점은 그걸 물러섬 1.0 · 충실성 0.0 으로 적는다 — 성능처럼 보이지만
        # 입력이 빈 것이다. 여기서 세서 말해 준다.
        print(f"⚠ 발췌가 하나도 안 잡힌 문항 {empty}/{len(pairs)}개. "
              f"평가 세트의 doc_id 가 코퍼스와 다를 수 있습니다")
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
    parser.add_argument(
        "--unscoped",
        action="store_true",
        help="공고를 안 알려주고 검색부터 시킨다 (전 구간 E2E). 기본은 scoped",
    )
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
    parser.add_argument(
        "--generation",
        action="store_true",
        help="발췌 본문을 생성용(표 마크업 유지)으로 뽑는다. 검색은 그대로다",
    )
    parser.add_argument(
        "--embed", default="tei", choices=["tei", "local", "openai", "fake"]
    )
    parser.add_argument(
        "--rerank", default="tei", choices=["tei", "local", "cohere", "fake"]
    )
    args = parser.parse_args()

    if args.export:
        out = args.out or settings.EVAL_RESULTS / f"contexts_{args.evalset}.jsonl"
        export_contexts(
            args.evalset,
            out,
            generation=args.generation,
            scoped=not args.unscoped,
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
    kept = fit_context(chunks, generation=True)
    context = format_context(kept, generation=True)

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
