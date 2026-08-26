"""생성 평가 — 충실성. 여기만 돈이 든다.

    근거표시율   답변에 [1] [2] 를 달았는지.        LLM 없이 잰다
    물러섬       모르면 모른다고 했는지.            LLM 없이 잰다
    숫자일치     답변의 숫자가 발췌에 있는 것인지.   LLM 없이 잰다
    충실성       발췌에 있는 내용만 말했는지.        LLM 이 YES/NO 로 채점

앞의 셋은 공짜다. 충실성만 LLM 을 부르므로 팀 한도($20)를 잡아먹는다.
"""

import re

from evaluation.evalset import matches


def cite_rate(answer):
    """답변에 근거 번호 [1] [2] 를 달았는지. LLM 없이 잰다."""
    lines = [line for line in answer.split("\n") if len(line.strip()) > 15]
    if not lines:
        return 0.0
    return sum(1 for line in lines if re.search(r"\[\d+\]", line)) / len(lines)


def said_no_info(answer):
    """근거가 없을 때 물러섰는지. 지어내는 것보다 이게 낫다."""
    return bool(
        re.search(
            r"확인되지\s*않|찾을\s*수\s*없|명시되어\s*있지\s*않|정보가\s*없", answer
        )
    )


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
        "[문서]",
        context,
        "",
        "[질문]",
        question,
        "",
        "[답변]",
        answer,
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
    for pair in pairs[:limit] if limit else pairs:
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
