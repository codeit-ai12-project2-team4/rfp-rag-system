"""질문을 손보는 부품 — 검색 **전에** 끼운다.

사람이 던지는 질문은 문서에 쓰인 말과 다르다.

    사람: "이거 언제까지 내야 돼요?"
    문서: "제안서 제출 마감일시 : 2024. 10. 15. 17:00"

겹치는 단어가 없어서 검색이 실패한다. 질문 쪽을 문서 말투로 고치거나,
여러 갈래로 불려서 그중 하나라도 걸리게 한다.

강의 L05 의 "질문 재작성", "다중 질의" 와 같은 것이다.
"""



class QueryRewrite:
    """질문을 검색하기 좋은 문장으로 고친다. LLM 호출 1회."""

    def __init__(self, llm, max_tokens=80):
        self.llm = llm
        self.max_tokens = max_tokens

    def __call__(self, state):
        system = "\n".join([
            "당신은 공공 입찰 제안요청서(RFP) 검색을 돕는 도우미입니다.",
            "사용자 질문을 문서에서 찾기 좋은 한 문장으로 고쳐 씁니다.",
            "규칙:",
            "- 질문에 대답하지 말 것",
            "- 문서에 실제로 쓰일 법한 공문 용어로 바꿀 것",
            "  (예: '언제까지 내요' → '제안서 제출 마감일시')",
            "- 고친 문장 하나만 출력할 것",
        ])
        user = "\n".join(["원래 질문: " + state.question, "고친 질문:"])

        rewritten = self.llm.ask(system, user, max_tokens=self.max_tokens)
        rewritten = rewritten.split("질문:")[-1].strip().split("\n")[0]

        if rewritten:
            state.queries = [rewritten]
            state.note(f"질문 고침 → {rewritten}")
        else:
            state.note("고치기 실패, 원래 질문 유지")
        return state

    def __repr__(self):
        return "QueryRewrite(질문 다듬기)"


class MultiQuery:
    """질문을 여러 갈래로 불린다. LLM 호출 1회.

    원래 질문도 남긴다. 고친 질문이 엉뚱해도 원래 질문이 받쳐 준다.
    """

    def __init__(self, llm, n=3, keep_original=True, max_tokens=150):
        self.llm = llm
        self.n = n
        self.keep_original = keep_original
        self.max_tokens = max_tokens

    def __call__(self, state):
        base = state.queries[0] if state.queries else state.question
        system = "\n".join([
            "당신은 공공 입찰 제안요청서(RFP) 검색을 돕는 도우미입니다.",
            f"주어진 질문을 표현만 바꾼 검색용 질문 {self.n}개로 다시 씁니다.",
            "규칙:",
            "- 뜻은 같게, 쓰는 단어는 다르게",
            "- 한 줄에 하나씩, 번호 없이",
            "- 질문에 대답하지 말 것",
        ])
        user = "\n".join(["질문: " + base, "", f"검색용 질문 {self.n}개:"])

        generated = self.llm.ask(system, user, max_tokens=self.max_tokens)

        queries = []
        for line in generated.split("\n"):
            line = line.lstrip(" -*0123456789.)").strip()
            if len(line) > 4:
                queries.append(line)
        queries = queries[: self.n]

        if self.keep_original:
            queries = [state.question, *queries]

        # 중복 제거 (강의에서 dict 로 했던 것과 같다)
        state.queries = list(dict.fromkeys(q for q in queries if q))
        state.note(f"질문 {len(state.queries)}개로 불림")
        return state

    def __repr__(self):
        return f"MultiQuery(n={self.n})"


class AddKeywords:
    """질문에 RFP 공문 용어를 덧붙인다. LLM 없이, 규칙만으로.

    MultiQuery 는 LLM 호출이 들어간다. 이건 공짜다.
    자주 쓰는 말만 사전에 넣어 두고 붙인다. 먼저 이걸 써 보고,
    부족하면 MultiQuery 로 바꾸는 게 비용 면에서 낫다.
    """

    SYNONYMS = {
        "기간": "과업기간 계약기간 사업기간",
        "언제까지": "제출 마감일시 마감",
        "얼마": "사업규모 배정예산 사업금액",
        "예산": "사업규모 배정예산 사업금액",
        "자격": "입찰 참가자격 면허 실적",
        "제출": "제안서 제출 방법 장소 마감",
        "평가": "제안서 평가방법 배점 기술평가 가격평가",
        "요구사항": "요구사항 고유번호 기능 요구사항 SFR",
        "계약": "계약방법 낙찰자 결정방법",
        "벌금": "지체상금 손해배상",
        "저작권": "지식재산권 소유권",
    }

    def __init__(self, synonyms=None):
        self.synonyms = synonyms or self.SYNONYMS

    def __call__(self, state):
        base = state.queries[0] if state.queries else state.question
        extra = [words for key, words in self.synonyms.items() if key in base]
        if extra:
            state.queries = [base, base + " " + " ".join(extra)]
            state.note(f"공문 용어 덧붙임: {' '.join(extra)[:50]}")
        else:
            state.note("붙일 용어 없음")
        return state

    def __repr__(self):
        return f"AddKeywords(사전 {len(self.synonyms)}개)"
