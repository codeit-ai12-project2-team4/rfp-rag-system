"""답을 만드는 부품 — 조립대 맨 끝.

Generate  찾은 청크를 근거로 자연어 답변. 조립대(`Pipeline`) 안에서 쓴다.

팀 파이프라인의 생성은 `src/generation.py` 의 `generate_answer()` 다.
이건 조립대 실험용이다. 프롬프트는 양쪽을 맞춰 둔다.

프롬프트는 강의처럼 "\\n".join([...]) 로 쓴다. 리스트로 쓰면 줄 단위로
넣고 빼기 쉽고, 무엇이 프롬프트에 들어가는지 그대로 보인다.
"""

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
