"""검색 평가 — 공짜다. LLM 을 안 부른다.

    적중률(k)   top k개 안에 정답이 있는 질문의 비율.  높을수록 좋다. 0~1
    MRR         정답이 1등이면 1.0, 2등이면 0.5, 3등이면 0.33 …  높을수록 좋다
    doc적중률    정답이 아니어도 같은 공고 문서를 물어 왔는지

적중률은 "안에 있냐 없냐"만 본다. 1위로 찾은 것과 5위로 겨우 찾은 것이 같은
점수다. **리랭커는 적중률이 아니라 등수를 올리는 부품이므로** MRR 로 봐야
쓸모가 보인다.

공짜니까 청킹 설정이나 검색 방법을 바꿔 가며 마음껏 돌려도 된다.
"""

from evaluation.evalset import matches


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


def body(chunk, generation=False):
    """청크에서 쓸 본문을 고른다. 생성용이 없으면 검색용으로 떨어진다.

    Args:
        chunk: Document.
        generation: True 면 생성용 본문을 고른다.

    Returns:
        str: 본문.
    """
    if generation:
        return chunk.metadata.get("gen") or chunk.page_content
    return chunk.page_content


def fit_budget(chunks, budget, generation=False):
    """컨텍스트 예산 안에 들어가는 청크만 앞에서부터 남긴다.

    **청크가 크면 적중률@5 는 그냥 올라간다.** 극단적으로 문서 하나를 청크
    하나로 만들면 적중률이 1.0 이 된다. 크기가 다른 설정을 비교하려면 개수가
    아니라 **글자 수를 맞춰야** 공정하다. 생성 단계가 실제로 받는 것도 개수가
    아니라 글자 수다 (`settings.MAX_CONTEXT_CHARS`).

    Args:
        chunks: 검색 결과 Document 리스트 (점수 순).
        budget: 넣을 수 있는 최대 글자 수.
        generation: True 면 생성용 본문(`metadata["gen"]`) 길이로 잰다.
            프롬프트에 실제로 들어가는 건 그쪽이라, 검색용으로 재면 표
            마크업만큼 예산을 조용히 넘긴다.

    Returns:
        예산 안에 들어가는 앞쪽 청크들. 첫 청크가 예산보다 커도 하나는 넣는다.
    """
    kept, used = [], 0
    for chunk in chunks:
        size = len(body(chunk, generation))
        if kept and used + size > budget:
            break
        kept.append(chunk)
        used += size
    return kept


def score_all(pairs, search, k=5, mrr_k=10, budget=None):
    """지표를 한 번에 잰다. 표로 비교할 때 쓴다.

    검색은 질문당 한 번만 돈다. hit_rate / mrr / doc_hit_rate 를 따로 부르면
    세 번 돌아서 세 배 느리다.

    Args:
        pairs: `{"question", "keywords", "doc_id"}` 꼴 질문 리스트.
        search: 질문을 받아 Document 리스트를 돌려주는 함수.
        k: 적중률과 doc 적중률을 볼 상위 개수.
        mrr_k: MRR 을 볼 상위 개수.
        budget: 글자 수 예산. 주면 **상위 k개 대신 예산에 들어가는 만큼**을 본다.
            청크 크기가 다른 설정을 공정하게 비교할 때 쓴다.

    Returns:
        적중률·MRR·doc적중률·질문수를 담은 dict. budget 을 주면 평균 청크수와
        평균 글자수도 함께 담는다. pairs 가 비면 빈 dict.
    """
    if not pairs:
        return {}

    hits = rank_sum = doc_hits = 0.0
    used_chunks = used_chars = 0
    for pair, chunks in _search_once(pairs, search, max(k, mrr_k)):
        top = fit_budget(chunks, budget) if budget else chunks[:k]
        used_chunks += len(top)
        used_chars += sum(len(c.page_content) for c in top)
        hits += any(matches(c.page_content, pair["keywords"]) for c in top)

        for rank, chunk in enumerate(chunks[:mrr_k], 1):
            if matches(chunk.page_content, pair["keywords"]):
                rank_sum += 1.0 / rank
                break

        gold = pair.get("doc_id")
        if gold:
            doc_hits += any(c.metadata.get("doc_id") == gold for c in top)

    n = len(pairs)
    label = f"@{budget}자" if budget else f"@{k}"
    out = {
        f"적중률{label}": round(hits / n, 3),
        "MRR": round(rank_sum / n, 3),
        f"doc_hits{label}": round(doc_hits / n, 3),
        "질문수": n,
    }
    if budget:
        out["평균청크수"] = round(used_chunks / n, 1)
        out["평균글자"] = int(used_chars / n)
    return out


def compare(setups, pairs, k=5, verbose=True, budget=None):
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

    Args:
        setups: `{설정 이름: 검색 함수}`. 검색 함수는 질문을 받아 Document 리스트.
        pairs: 질문 리스트.
        k: 상위 몇 개를 볼지. budget 을 주면 무시된다.
        verbose: 설정마다 걸린 시간을 찍을지.
        budget: 글자 예산. 청크 크기가 다른 설정을 공정하게 비교할 때 준다.

    Returns:
        설정별 지표 DataFrame. 적중률 → MRR 순으로 내림차순 정렬.
    """
    import time

    import pandas as pd

    rows = []
    for name, search in setups.items():
        started = time.time()
        row = {"설정": name, **score_all(pairs, search, k=k, budget=budget)}
        elapsed = time.time() - started
        row["초"] = round(elapsed, 1)
        rows.append(row)
        if verbose:
            print(f"  {name} … {elapsed:.1f}초")
            if elapsed > 60:
                print(
                    "     느립니다. lambda 안에서 검색기를 새로 만들고 있지 않은지 확인하세요."
                )
    label = f"@{budget}자" if budget else f"@{k}"
    return pd.DataFrame(rows).sort_values([f"적중률{label}", "MRR"], ascending=False)


def _run_search(search, question):
    """조립대든 함수든 검색 부품이든 다 받아서 청크 리스트를 얻는다."""
    result = search(question)
    if hasattr(result, "chunks"):  # Pipeline 이 돌려준 State
        return result.chunks
    return result
