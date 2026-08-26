"""평가용 질문 세트 만들고 저장하기.

정답 키워드는 **리스트**다. 강의 L07 처럼 `"3,000원 5만 원"` 을 한 덩어리로
넣으면 `keyword in page_content` 가 절대 안 맞는다. 정답 청크가 1위인데도
[X] 로 찍힌다. 공백을 무시하고 하나씩 맞춰 본다.
"""

import json
import re

from config import settings


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


def save_evalset(pairs, name="eval_qa"):
    """평가 질문을 파일로. 팀원이 같은 세트로 재야 비교가 된다."""
    settings.make_dirs()
    path = settings.DATA / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    return path


def load_evalset(name="eval_qa"):
    path = settings.DATA / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"평가 세트가 없습니다: {path}\n노트북 4번에서 만들고 저장하세요."
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
