"""평가용 질문 세트 만들고 저장하기.

명령줄로 돌리면 `data/eval_qa.json` 을 만든다.

    python src/evaluation/evalset.py
    python src/evaluation/evalset.py --docs cleaned_documents --n 40


정답 키워드는 **리스트**다. 강의 L07 처럼 `"3,000원 5만 원"` 을 한 덩어리로
넣으면 `keyword in page_content` 가 절대 안 맞는다. 정답 청크가 1위인데도
[X] 로 찍힌다. 공백을 무시하고 하나씩 맞춰 본다.
"""

import argparse
import json
import re
import sys
from pathlib import Path

# `python src/evaluation/evalset.py` 로 직접 돌릴 때 config 와 src 를 찾게 한다.
_ROOT = Path(__file__).resolve().parents[2]
for _folder in (_ROOT / "src", _ROOT):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

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


# =========================================================================
# 어려운 질문 만들기 — 유형별
#
# 위 `make_pairs_from_documents` 는 라벨을 그대로 인용한다("과업기간은?").
# 본문에 같은 말이 있으니 BM25 만으로도 다 맞는다. 실제로 재 보니 여섯 가지
# 검색 방법이 전부 0.88~0.99 로 뭉쳐서 우열이 안 보였다.
#
# 아래 세 유형은 **질문 글자와 본문 글자가 일부러 다르게** 만든다.
#   요구사항  ID 로 묻고 명칭을 답하게 한다 — 표 한 줄을 정확히 찾아야 한다
#   배점      항목 이름으로 묻고 점수를 답하게 한다 — 평가표 안을 찾아야 한다
#   의역      라벨 단어를 안 쓰고 묻는다 — BM25 가 불리하고 Dense 가 유리하다
# =========================================================================

# "요구사항 고유번호\n\nSFR-001\n\n요구사항 분류\n\n기능 요구사항\n\n요구사항 명칭\n\n사용자 인증"
_REQUIREMENT = re.compile(
    r"요\s*구\s*사\s*항\s*고\s*유\s*번\s*호[\s\n]*([A-Z]{2,4}-\d{2,4})[\s\n]+"
    r"(?:요\s*구\s*사\s*항\s*분\s*류[\s\n]*[^\n]{2,20}[\s\n]+)?"
    r"요\s*구\s*사\s*항\s*명\s*칭[\s\n]*([^\n]{3,40})"
)

# "신용평가 등급에 의한 경영상태(4점)" / "입찰가격평가 : 10점"
# 항목 이름은 한글·영문·공백만 받는다. 숫자나 괄호가 섞이면 정규식이 엉뚱한
# 조각을 집은 것이라 질문이 말이 안 된다 ("정량20, 정성70)] +가격평가" 같은 것).
_SCORE = re.compile(
    r"([가-힣][가-힣A-Za-z ]{3,20}?)\s*(?:[(（]|[:：]\s*)\s*(\d{1,3})\s*점"
)
# 조사로 시작하면 문장 중간을 집은 것이다 ("를 실시하여 종합평가점수")
_SCORE_SKIP = (
    "총",
    "합",
    "만",
    "이상",
    "미만",
    "를",
    "을",
    "이",
    "가",
    "의",
    "에",
    "로",
    "와",
    "과",
    "은",
    "는",
    "및",
    "또",
    "해",
    "여",
)

# 라벨 단어를 안 쓰고 묻는다. (본문 라벨 패턴, 질문, 값 패턴)
_PARAPHRASE = [
    (r"과\s*업\s*기\s*간", "「{title}」 사업은 며칠 안에 끝내야 하나?", _LINE),
    (r"계\s*약\s*기\s*간", "「{title}」 사업은 얼마 동안 하나?", _LINE),
    (r"사\s*업\s*(?:규모|예산|금액)", "「{title}」 사업에 돈이 얼마나 드나?", _MONEY),
    (r"배\s*정\s*예\s*산", "「{title}」 사업에 쓸 수 있는 돈은?", _MONEY),
    (r"계\s*약\s*(?:방법|방식)", "「{title}」 사업은 업체를 어떻게 고르나?", _LINE),
    (r"낙\s*찰\s*방\s*식", "「{title}」 사업은 누가 따내나?", _LINE),
    (
        r"입\s*찰\s*참\s*가\s*자\s*격",
        "「{title}」 사업에 아무나 들어갈 수 있나?",
        _LINE,
    ),
]

# 문서에 있을 리 없는 것들. 물러서야 정답이다 (검색 지표 대상이 아니다).
_UNANSWERABLE = [
    "「{title}」 사업의 작년 낙찰 업체는 어디인가?",
    "「{title}」 사업 담당자의 휴대폰 번호는?",
    "「{title}」 사업에 몇 개 업체가 입찰했나?",
    "「{title}」 사업의 실제 계약 체결일은 언제인가?",
    "「{title}」 사업을 수주한 업체의 매출액은?",
]


def _pair(document, question, keywords, kind, note):
    """질문 하나를 만든다.

    Args:
        document: 전처리 레코드.
        question: 질문 문장.
        keywords: 정답 키워드 리스트. 하나라도 청크에 있으면 맞은 것으로 본다.
        kind: 유형 이름. 유형별로 나눠 봐야 어디서 갈리는지 보인다.
        note: 원문 근거. 사람이 훑을 때 쓴다.

    Returns:
        질문 dict.
    """
    return {
        "question": question.format(title=document["meta"]["title"][:40]),
        "keywords": keywords,
        "doc_id": document["meta"]["doc_id"],
        "type": kind,
        "answerable": bool(keywords),
        "note": note[:90],
        "checked_by": None,
    }


def make_hard_pairs(documents, per_type=40, per_doc=1):
    """유형별로 어려운 질문을 만든다. LLM 없이 원문에서 뽑는다.

    정답이 원문 그대로라 라벨링이 필요 없다. 다만 **사람이 표본은 훑어야 한다** —
    정규식이 엉뚱한 줄을 잡는 경우가 있다. `note` 에 근거를 남겨 뒀다.

    Args:
        documents: 전처리 레코드 리스트.
        per_type: 유형당 몇 개까지 만들지.
        per_doc: 한 문서에서 유형당 몇 개까지 뽑을지. 문서에 골고루 퍼지게 한다.

    Returns:
        질문 리스트. 각 항목에 `type` 과 `answerable` 이 있다.
    """
    made = {"요구사항": [], "배점": [], "의역": [], "없음": []}

    for document in documents:
        text = document["text"]

        # 요구사항 — ID 로 묻고 명칭을 답하게 한다
        if len(made["요구사항"]) < per_type:
            for match in list(_REQUIREMENT.finditer(text))[:per_doc]:
                req_id, name = match.group(1), match.group(2).strip()
                if len(name) < 3:
                    continue
                made["요구사항"].append(
                    _pair(
                        document,
                        f"「{{title}}」의 {req_id} 요구사항 명칭은 무엇인가?",
                        [name],
                        "요구사항",
                        match.group(0).replace("\n", " "),
                    )
                )

        # 배점 — 항목 이름으로 묻고 점수를 답하게 한다
        if len(made["배점"]) < per_type:
            seen = 0
            for match in _SCORE.finditer(text):
                item = match.group(1).strip()
                answer = " ".join(match.group(0).split())  # 칸 사이 줄바꿈을 편다
                if seen >= per_doc or len(item) < 4 or item.startswith(_SCORE_SKIP):
                    continue
                made["배점"].append(
                    _pair(
                        document,
                        f"「{{title}}」 평가에서 '{item}' 의 배점은 몇 점인가?",
                        [answer],
                        "배점",
                        answer,
                    )
                )
                seen += 1

        # 의역 — 라벨 단어를 안 쓰고 묻는다
        if len(made["의역"]) < per_type:
            seen = 0
            for key, template, value in _PARAPHRASE:
                if seen >= per_doc:
                    break
                match = re.search(_FIELD.format(key=key, value=value), text)
                if not match:
                    continue
                answer = match.group(1).strip().rstrip(".,")
                if len(answer) < 3:
                    continue
                made["의역"].append(
                    _pair(document, template, [answer], "의역", match.group(0).strip())
                )
                seen += 1

    # 없음 — 문서를 골고루 섞어 뽑는다
    step = max(len(documents) // max(len(_UNANSWERABLE) * 3, 1), 1)
    for i, document in enumerate(documents[::step]):
        if len(made["없음"]) >= per_type // 3:
            break
        made["없음"].append(
            _pair(
                document,
                _UNANSWERABLE[i % len(_UNANSWERABLE)],
                [],
                "없음",
                "문서에 없는 내용 — 물러서야 정답",
            )
        )

    return [q for kind in made for q in made[kind][:per_type]]


def main():
    """명령줄에서 `data/eval_qa.json` 을 만든다."""
    parser = argparse.ArgumentParser(description="평가용 질문 세트를 만든다.")
    parser.add_argument(
        "--docs", default="cleaned_documents", help="질문을 뽑을 전처리본"
    )
    parser.add_argument("--n", type=int, default=40, help="유형당 문항 수")
    parser.add_argument(
        "--per-doc", type=int, default=1, help="한 문서에서 유형당 몇 개까지"
    )
    parser.add_argument("--name", default="eval_qa")
    parser.add_argument(
        "--easy", action="store_true", help="예전 방식(라벨 그대로 인용)도 같이 넣는다"
    )
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="질문에서 「사업명」을 뺀다. 머리말의 이득이 "
        "질문 형식에 대한 과적합인지 확인할 때 쓴다",
    )
    args = parser.parse_args()

    from preprocessing import load_documents  # 명령줄로 쓸 때만 필요하다

    documents = load_documents(args.docs)
    pairs = make_hard_pairs(documents, per_type=args.n, per_doc=args.per_doc)
    if args.easy:
        for pair in make_pairs_from_documents(documents):
            pair.update(type="라벨인용", answerable=True)
            pairs.append(pair)

    if args.no_title:
        # 머리말 청크는 [사업명] 으로 시작한다. 질문에도 사업명이 있으면 그것만으로
        # 맞을 수 있다. 사업명을 빼고 재야 진짜 이득인지 알 수 있다.
        for pair in pairs:
            body = pair["question"].split("」", 1)[-1]
            pair["question"] = re.sub(
                r"^\s*(의|사업은|사업에서|사업에|사업의|평가에서)\s*", "", body
            ).strip()

    counts = {}
    for pair in pairs:
        counts[pair["type"]] = counts.get(pair["type"], 0) + 1
    for kind, n in counts.items():
        print(f"  {kind:<8} {n:>4}개")
    print(f"  {'합계':<8} {len(pairs):>4}개")

    path = save_evalset(pairs, args.name)
    print(f"\n→ {path}")
    print("\n표본을 눈으로 훑으세요. note 에 원문 근거가 있습니다.")
    for pair in pairs[:3]:
        print(f"  [{pair['type']}] {pair['question']}")
        print(f"      정답 {pair['keywords']}  근거 {pair['note'][:60]}")


if __name__ == "__main__":
    main()
