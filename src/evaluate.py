"""평가.

강의 L06 에서 배운 것을 그대로 쓰되, 두 가지를 손봤다.

1. **정답 키워드를 여러 개 넣을 수 있다.**
   L07 에서 `"3,000원 5만 원"` 처럼 여러 어절을 붙여 놓으면
   `answer_keyword in page_content` 가 절대 안 맞는다. 정답 청크가 1위인데도
   [X] 로 찍힌다. 이제 키워드를 리스트로 주고, 공백을 무시하고 맞춰 본다.

2. **적중률 말고 MRR 도 잰다.**
   적중률은 "top k개 안에 있냐 없냐"만 본다. 1위로 찾은 것과 5위로 겨우
   찾은 것이 같은 점수다. MRR 은 몇 등이었는지를 본다. 리랭커가 쓸모 있는지
   판단하려면 이게 필요하다 — 리랭커는 적중률이 아니라 등수를 올리는 부품이다.

## 무슨 지표를 봐야 하나

    적중률(k)   top k개 안에 정답이 있는 질문의 비율.  높을수록 좋다. 0~1
    MRR         정답이 1등이면 1.0, 2등이면 0.5, 3등이면 0.33 …  높을수록 좋다
    충실성      답변이 발췌에 있는 내용만 말했는지.  LLM 이 채점한다

검색 지표(적중률, MRR)는 LLM 을 안 부르므로 **공짜다.** 청킹 설정이나
검색 방법을 바꿔 가며 마음껏 돌려도 된다. 충실성만 돈이 든다.
"""

import json
import re

from bidmate import paths

# =========================================================================
# 평가용 질문 세트
# =========================================================================


def normalize(text):
    """공백을 다 지운다. '5만 원' 과 '5만원' 을 같게 보려고."""
    return re.sub(r"\s+", "", str(text))


def matches(chunk_text, keywords):
    """청크 안에 정답 키워드가 들어 있는지.

    keywords 가 리스트면 **하나라도** 맞으면 통과. 여러 표현 중 아무거나
    나오면 되는 경우에 쓴다.
    """
    if isinstance(keywords, str):
        keywords = [keywords]
    body = normalize(chunk_text)
    return any(normalize(keyword) in body for keyword in keywords)


def save_evalset(pairs, name="evalset"):
    """평가 질문을 파일로. 팀원이 같은 세트로 재야 비교가 된다."""
    paths.make_dirs()
    path = paths.EVALSETS / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    return path


def load_evalset(name="evalset"):
    path = paths.EVALSETS / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"평가 세트가 없습니다: {path}\n"
            "노트북 4번에서 만들고 저장하세요."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --- 질문 자동 만들기 (LLM 없이) -------------------------------------------

# RFP 는 서식이 정형화돼 있어서 정규식으로 질문/정답 쌍을 뽑을 수 있다.
# LLM 이 만든 질문보다 정답이 정확하다 — 원문을 그대로 쓰기 때문이다.
_MONEY = r"[\d,]+\s*(?:천원|백만원|억원|원)"
_LINE = r"[^\n]{4,60}"

FIELD_PATTERNS = [
    (r"사\s*업\s*(?:규모|예산|금액)", "{title} 사업의 예산은 얼마인가?", _MONEY),
    (r"배\s*정\s*예\s*산", "{title} 사업의 배정예산은 얼마인가?", _MONEY),
    (r"과\s*업\s*기\s*간", "{title} 사업의 과업기간은?", _LINE),
    (r"계\s*약\s*(?:방법|방식)", "{title} 사업의 계약방식은?", _LINE),
    (r"낙\s*찰\s*방\s*식", "{title} 사업의 낙찰방식은?", _LINE),
]

# "□ 사업규모 : 87,000,000원" — 앞의 글머리표와 콜론을 흡수한다.
# 한글 문서는 "사 업 명" 처럼 자간을 띄우는 일이 잦아 글자 사이 공백을 허용한다.
_FIELD = r"[□■○◦oㅇ*\-\s]*{key}\s*[:：]\s*({value})"


def make_pairs_from_documents(documents, max_per_doc=2):
    """문서에서 (질문, 정답 키워드, doc_id) 짝을 정규식으로 뽑는다.

    **반드시 사람이 눈으로 훑어야 한다.** 정규식이 엉뚱한 줄을 잡거나
    값이 잘리는 경우가 있다. 노트북 4번에서 표로 보고 이상한 걸 지운다.
    """
    pairs = []
    for document in documents:
        made = 0
        for key, template, value in FIELD_PATTERNS:
            if made >= max_per_doc:
                break
            match = re.search(_FIELD.format(key=key, value=value), document["text"])
            if not match:
                continue
            answer = match.group(1).strip().rstrip(".,")
            if len(answer) < 2:
                continue
            pairs.append({
                "question": template.format(title=document["meta"]["title"][:40]),
                "keywords": [answer],
                "doc_id": document["meta"]["doc_id"],
                "note": "자동 생성 · 원문: " + match.group(0).strip()[:70],
                "checked_by": None,
            })
            made += 1
    return pairs


# =========================================================================
# 검색 평가 — 공짜
# =========================================================================


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
                found = " ← 정답" if matches(chunk.page_content, pair["keywords"]) else ""
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
                print("     느립니다. lambda 안에서 검색기를 새로 만들고 있지 않은지 확인하세요.")
    return pd.DataFrame(rows).sort_values(f"적중률@{k}", ascending=False)


def _run_search(search, question):
    """조립대든 함수든 검색 부품이든 다 받아서 청크 리스트를 얻는다."""
    result = search(question)
    if hasattr(result, "chunks"):  # Pipeline 이 돌려준 State
        return result.chunks
    return result


# =========================================================================
# 생성 평가
# =========================================================================


def cite_rate(answer):
    """답변에 근거 번호 [1] [2] 를 달았는지. LLM 없이 잰다."""
    lines = [line for line in answer.split("\n") if len(line.strip()) > 15]
    if not lines:
        return 0.0
    return sum(1 for line in lines if re.search(r"\[\d+\]", line)) / len(lines)


def said_no_info(answer):
    """근거가 없을 때 물러섰는지. 지어내는 것보다 이게 낫다."""
    return bool(re.search(r"확인되지\s*않|찾을\s*수\s*없|명시되어\s*있지\s*않|정보가\s*없", answer))


def number_match(answer, keywords):
    """정답에 있는 숫자가 답변에도 그대로 있는지.

    RFP 에서 금액·기간을 틀리면 치명적이다. 이건 LLM 없이 잰다.
    """
    if isinstance(keywords, str):
        keywords = [keywords]
    gold_numbers = set()
    for keyword in keywords:
        gold_numbers |= {n.replace(",", "") for n in re.findall(r"[\d,]+", keyword)}
    if not gold_numbers:
        return float(matches(answer, keywords))
    answer_numbers = {n.replace(",", "") for n in re.findall(r"[\d,]+", answer)}
    return len(gold_numbers & answer_numbers) / len(gold_numbers)


# --- 충실성 (LLM 이 채점) ---------------------------------------------------

JUDGE_SYSTEM = "\n".join([
    "당신은 답변 채점자입니다.",
    "답변의 내용이 [문서] 에서 확인되면 YES, 문서에 없는 내용이 하나라도 있으면 NO 를 출력합니다.",
    "YES 또는 NO 한 단어만 출력합니다. 설명하지 않습니다.",
])


def judge_faithfulness(llm, question, context, answer):
    """답변이 발췌 안에서만 말했는지 LLM 에게 물어본다. 강의 L06 방식.

    YES / NO 로 나온다. 값이 애매하면 None.
    싼 모델(gpt-5-nano 등)로 돌려도 충분하다.
    """
    user = "\n".join([
        "[문서]", context, "",
        "[질문]", question, "",
        "[답변]", answer,
    ])
    verdict = llm.ask(JUDGE_SYSTEM, user, max_tokens=10).strip().upper()
    if "YES" in verdict:
        return True
    if "NO" in verdict:
        return False
    return None


def evaluate_answers(pipeline, pairs, judge_llm=None, limit=None, verbose=True):
    """조립대를 돌려 답변을 만들고 채점한다. **LLM 호출이 든다.**

    호출 수 = 질문 수 × (1 + 채점 1). limit 으로 먼저 몇 개만 해 볼 것.
    """
    import pandas as pd

    rows = []
    for pair in (pairs[:limit] if limit else pairs):
        state = pipeline(pair["question"])
        row = {
            "질문": pair["question"][:40],
            "정답": " / ".join(pair["keywords"])[:30],
            "숫자일치": round(number_match(state.answer or "", pair["keywords"]), 2),
            "근거인용": round(cite_rate(state.answer or ""), 2),
            "물러섬": said_no_info(state.answer or ""),
            "답변": (state.answer or "")[:60],
        }
        if judge_llm:
            row["충실성"] = judge_faithfulness(
                judge_llm, pair["question"], state.context, state.answer or ""
            )
        rows.append(row)
        if verbose:
            print(f"  {len(rows)}/{limit or len(pairs)} …", end="\r")

    if verbose:
        print()
    return pd.DataFrame(rows)
