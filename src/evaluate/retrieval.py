"""검색 평가 — 공짜다. LLM 을 안 부른다.

    적중률(k)   top k개 안에 정답이 있는 질문의 비율.  높을수록 좋다. 0~1
    MRR         정답이 1등이면 1.0, 2등이면 0.5, 3등이면 0.33 …  높을수록 좋다
    doc적중률    정답이 아니어도 같은 공고 문서를 물어 왔는지

적중률은 "안에 있냐 없냐"만 본다. 1위로 찾은 것과 5위로 겨우 찾은 것이 같은
점수다. **리랭커는 적중률이 아니라 등수를 올리는 부품이므로** MRR 로 봐야
쓸모가 보인다.

공짜니까 청킹 설정이나 검색 방법을 바꿔 가며 마음껏 돌려도 된다.
"""

from evaluate.evalset import matches


def _search_once(pairs, search, k):
    """질문마다 검색을 **한 번만** 돌리고 결과를 모아 둔다.

    지표를 세 개 재려고 검색을 세 번 돌리면 그만큼 느려진다. 청크가 수만 개면
    검색 한 번이 꽤 비싸다. 한 번 돌려서 세 지표를 다 계산한다.
    """
    return [(pair, _run_search(search, pair["question"])[:k]) for pair in pairs]


def hit_rate(pairs, search, k=5, verbose=False):
    """top k개 안에 정답이 들어온 질문의 비율.

    search 는 "질문을 받아 청크 리스트를 돌려주는 것"이면 뭐든 된다.
    조립대도 되고, 함수도 되고, 검색 부품 하나도 된다.

        hit_rate(pairs, lambda q: store.similarity_search(q, k=5))
        hit_rate(pairs, lambda q: rag(q).chunks)
    """
    hits = 0
    for pair in pairs:
        chunks = _run_search(search, pair["question"])[:k]
        hit = any(matches(c.page_content, pair["keywords"]) for c in chunks)
        hits += hit
        if verbose:
            mark = "O" if hit else "X"
            print(f"[{mark}] {pair['question'][:50]}")
            for rank, chunk in enumerate(chunks, 1):
                found = (
                    " ← 정답" if matches(chunk.page_content, pair["keywords"]) else ""
                )
                snippet = chunk.page_content[:45].replace("\n", " ")
                print(f"     {rank}위 {snippet}{found}")
    return hits / len(pairs) if pairs else 0.0


def mrr(pairs, search, k=10):
    """정답이 몇 등이었는지. 1등이면 1.0, 2등이면 0.5, 못 찾으면 0.

    리랭커를 켜고 끄며 비교할 때 이걸 본다. 적중률은 잘 안 움직이는데
    MRR 이 오르면 리랭커가 일한 것이다.
    """
    total = 0.0
    for pair in pairs:
        chunks = _run_search(search, pair["question"])[:k]
        for rank, chunk in enumerate(chunks, 1):
            if matches(chunk.page_content, pair["keywords"]):
                total += 1.0 / rank
                break
    return total / len(pairs) if pairs else 0.0


def doc_hit_rate(pairs, search, k=5):
    """정답 **문서**를 찾았는지. 청크 단위 매칭이 까다로울 때 쓴다.

    "예산 5억 이상 클라우드 공고" 같이 여러 공고를 가로지르는 질문은
    청크가 아니라 문서가 맞았는지를 봐야 한다.
    """
    hits = 0
    for pair in pairs:
        gold = pair.get("doc_id")
        if not gold:
            continue
        chunks = _run_search(search, pair["question"])[:k]
        hits += any(c.metadata.get("doc_id") == gold for c in chunks)
    return hits / len(pairs) if pairs else 0.0


def score_all(pairs, search, k=5, mrr_k=10):
    """지표 세 개를 한 번에. 표로 비교할 때 쓴다.

    검색은 질문당 한 번만 돈다. hit_rate / mrr / doc_hit_rate 를 따로 부르면
    세 번 돌아서 세 배 느리다.
    """
    if not pairs:
        return {}

    hits = rank_sum = doc_hits = 0.0
    for pair, chunks in _search_once(pairs, search, max(k, mrr_k)):
        top = chunks[:k]
        hits += any(matches(c.page_content, pair["keywords"]) for c in top)

        for rank, chunk in enumerate(chunks[:mrr_k], 1):
            if matches(chunk.page_content, pair["keywords"]):
                rank_sum += 1.0 / rank
                break

        gold = pair.get("doc_id")
        if gold:
            doc_hits += any(c.metadata.get("doc_id") == gold for c in top)

    n = len(pairs)
    return {
        f"적중률@{k}": round(hits / n, 3),
        "MRR": round(rank_sum / n, 3),
        f"doc_hits@{k}": round(doc_hits / n, 3),
        "질문수": n,
    }


def compare(setups, pairs, k=5, verbose=True):
    """여러 설정을 한 표로 비교한다.

        bm25 = BM25(chunks, k=5)          # ← 루프 밖에서 미리 만든다
        compare({
            "BM25만":   lambda q: bm25.search(q, 5),
            "임베딩만": lambda q: store.similarity_search(q, k=5),
            "합친 것":  lambda q: rag(q).chunks,
        }, pairs)

    **주의** — lambda 안에서 `BM25(chunks)` 나 `Pipeline([...])` 를 만들면 안 된다.
    질문마다 인덱스를 새로 만들게 되어 수십 배 느려진다. 검색기는 밖에서 한 번
    만들어 두고 lambda 는 그걸 부르기만 하게 한다.
    """
    import time

    import pandas as pd

    rows = []
    for name, search in setups.items():
        started = time.time()
        row = {"설정": name, **score_all(pairs, search, k=k)}
        elapsed = time.time() - started
        row["초"] = round(elapsed, 1)
        rows.append(row)
        if verbose:
            print(f"  {name} … {elapsed:.1f}초")
            if elapsed > 60:
                print(
                    "     느립니다. lambda 안에서 검색기를 새로 만들고 있지 않은지 확인하세요."
                )
    return pd.DataFrame(rows).sort_values(f"적중률@{k}", ascending=False)


def _run_search(search, question):
    """조립대든 함수든 검색 부품이든 다 받아서 청크 리스트를 얻는다."""
    result = search(question)
    if hasattr(result, "chunks"):  # Pipeline 이 돌려준 State
        return result.chunks
    return result
