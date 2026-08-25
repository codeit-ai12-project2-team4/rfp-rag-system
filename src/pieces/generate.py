"""답을 만드는 부품 — 조립대 맨 끝.

Generate      찾은 청크를 근거로 자연어 답변
MakeCard      공고 하나를 요약 카드로 (과업기간·자격·제출방식·평가배점…)

프롬프트는 강의처럼 "\\n".join([...]) 로 쓴다. 리스트로 쓰면 줄 단위로
넣고 빼기 쉽고, 무엇이 프롬프트에 들어가는지 그대로 보인다.
"""

import json


# --- 프롬프트 ------------------------------------------------------------

SYSTEM_ANSWER = "\n".join([
    "당신은 B2G 입찰 컨설팅 회사 '입찰메이트'의 분석 담당자입니다.",
    "함께 주어지는 제안요청서(RFP) 발췌만을 근거로 답합니다.",
    "규칙:",
    "- 발췌에 없는 내용은 지어내지 말고 '제공된 문서에서 확인되지 않습니다'라고 답할 것",
    "- 금액, 기간, 마감일, 면허 요건은 원문 표기 그대로 옮길 것",
    "- 근거가 된 발췌 번호를 [1] [2] 형태로 문장 끝에 달 것",
    "- 서론 없이 결론부터, 짧게 쓸 것",
])

SYSTEM_CARD = "\n".join([
    "당신은 제안요청서(RFP)에서 정해진 항목을 뽑아내는 추출기입니다.",
    "주어진 발췌에서만 값을 찾습니다. 없으면 null 을 넣습니다. 추측하지 않습니다.",
    "반드시 JSON 하나만 출력합니다. 설명이나 코드블록 표시를 붙이지 않습니다.",
])

# 요약 카드에 채울 항목.
# 기관·예산·마감일은 여기 없다 — CSV 에 정확한 값이 이미 있어서
# LLM 에게 물어볼 이유가 없다. 문서 안에만 있는 것만 뽑는다.
CARD_FIELDS = {
    "headline": "이 사업이 무엇인지 한 문장 (40자 이내)",
    "project_period": "과업기간 (예: 계약체결일로부터 120일)",
    "contract_method": "계약방식 (제한경쟁입찰, 협상에 의한 계약 등)",
    "eligibility": "입찰 참가자격 요건 목록",
    "submission": "제안서 제출 방법·장소·close_date",
    "required_docs": "제출서류 목록",
    "evaluation": "평가 방법과 배점 (기술:가격 비율 등)",
    "key_requirements": "주요 요구사항 목록 (있으면 SFR-001 같은 번호 포함)",
    "tech_stack": "언급된 기술·플랫폼·표준 목록",
    "risks": "주의할 조건 (지체상금, 지식재산권 귀속, 하자보수 등) 목록",
}


def format_context(chunks):
    """청크를 번호 붙여 프롬프트용 문자열로. 이 번호가 곧 인용 번호가 된다."""
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


# --- 부품 ---------------------------------------------------------------


class Generate:
    """찾은 청크를 근거로 답변을 만든다. LLM 호출 1회."""

    def __init__(self, llm, system=SYSTEM_ANSWER, max_tokens=800):
        self.llm = llm
        self.system = system
        self.max_tokens = max_tokens

    def __call__(self, state):
        if not state.chunks:
            state.answer = "검색된 문서가 없습니다."
            state.note("청크가 없어 생성 건너뜀")
            return state

        user = "\n".join([
            "[문서 발췌]",
            format_context(state.chunks),
            "",
            "[질문]",
            state.question,
        ])
        state.answer = self.llm.ask(self.system, user, max_tokens=self.max_tokens)
        state.note(f"발췌 {len(state.chunks)}개로 답변 생성 ({len(state.answer)}자)")
        return state

    def __repr__(self):
        return f"Generate({self.llm.name})"


class MakeCard:
    """공고 하나를 요약 카드로 만든다. LLM 호출 1회.

    과제의 미션 문장은 "추출하고 **요약하여** 필요한 정보를 제공"이다.
    질의응답만으로는 절반이다. 컨설턴트가 하루 수백 건을 훑으려면 공고를
    던졌을 때 카드 한 장이 나와야 한다.

    쓰는 법 — 항목마다 다른 질문으로 검색해 발췌를 모은 뒤 한 번에 채운다.
    카드를 만들 때는 조립대를 이렇게 세운다:

        card_pipe = Pipeline([
            Hybrid([Dense(store, k=15, doc_ids=[공고번호]),      # 그 공고 안에서만
                    BM25(chunks, k=15, doc_ids=[공고번호])], k=12),
            Rerank(reranker, k=8),
            MakeCard(llm),
        ])
        result = card_pipe("과업기간 계약방식 참가자격 제출방법 평가배점")
        result.card
    """

    def __init__(self, llm, fields=None, max_tokens=1500):
        self.llm = llm
        self.fields = fields or CARD_FIELDS
        self.max_tokens = max_tokens

    def __call__(self, state):
        if not state.chunks:
            state.card = {}
            state.note("청크가 없어 카드 생성 건너뜀")
            return state

        schema_lines = [f'  "{key}": {desc}' for key, desc in self.fields.items()]
        user = "\n".join([
            "[문서 발췌]",
            format_context(state.chunks),
            "",
            "아래 항목을 채운 JSON 하나만 출력하세요.",
            "발췌에서 확인되지 않는 항목은 null (목록이면 []) 로 둡니다.",
            "{",
            ",\n".join(schema_lines),
            "}",
        ])

        raw = self.llm.ask(SYSTEM_CARD, user, max_tokens=self.max_tokens)
        card = parse_json(raw)

        card["doc_id"] = state.chunks[0].metadata.get("doc_id")
        card["evidence"] = [c.metadata.get("chunk_id") for c in state.chunks]
        state.card = card

        filled = sum(1 for k in self.fields if card.get(k))
        state.note(f"카드 항목 {filled}/{len(self.fields)} 채움")
        return state

    def __repr__(self):
        return f"MakeCard({len(self.fields)}개 항목)"


def parse_json(text):
    """LLM 이 뱉은 것에서 JSON 을 건져낸다.

    ```json 으로 감싸거나 앞뒤에 설명을 붙이는 일이 흔해서 그냥
    json.loads 하면 자주 터진다. 제일 바깥 중괄호 쌍만 잘라 쓴다.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return {"_parse_error": text[:300]}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as error:
        return {"_parse_error": f"{error}", "_raw": text[start : end + 1][:300]}


def render_card(card, meta=None):
    """카드를 사람이 읽는 형태로. 검수할 때 쓴다."""
    lines = []
    if meta:
        budget = f"{meta['budget']:,.0f}원" if meta.get("budget") else "-"
        lines.append(f"■ {meta.get('title', '')}")
        close_date = meta.get("bid_close_at", "-")
        lines.append(f"  {meta.get('agency', '')} · {budget} · close_date {close_date}")
        lines.append("")

    if card.get("_parse_error"):
        lines.append(f"  [JSON 파싱 실패] {card['_parse_error'][:150]}")
        return "\n".join(lines)

    def row(label, key):
        value = card.get(key)
        if not value:
            return
        if isinstance(value, list):
            if isinstance(value[0], dict):
                lines.append(f"  {label}")
                for item in value[:10]:
                    code = item.get("code") or item.get("번호") or ""
                    title = item.get("title") or item.get("명칭") or str(item)
                    lines.append(f"    - {('[' + code + '] ') if code else ''}{title}")
            else:
                lines.append(f"  {label:<9} " + " / ".join(str(v) for v in value))
        else:
            lines.append(f"  {label:<9} {value}")

    row("한줄요약", "headline")
    row("과업기간", "project_period")
    row("계약방식", "contract_method")
    row("제출방식", "submission")
    row("평가배점", "evaluation")
    row("참가자격", "eligibility")
    row("제출서류", "required_docs")
    row("기술스택", "tech_stack")
    row("주의사항", "risks")
    row("주요 요구사항", "key_requirements")
    return "\n".join(lines)
