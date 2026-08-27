"""부품과 조립대.

강의 해설의 `RAGPipeline` 과 같은 것이다. 부품들이 `state` 하나를 돌려가며
조금씩 채우고, 마지막에 답이 나온다.

    class Retriever:                          여기서도 똑같이
        def __call__(self, state):            __call__ 하나만 쓴다
            ...
            return state

강의와 다른 건 둘뿐이다.

1. state 가 dict 가 아니라 dataclass 다. `state["retrieved_chunks"]` 대신
   `state.chunks` 로 쓴다. 칸 이름을 잘못 쓰면 그 자리에서 에러가 난다.
2. 청크가 문자열이 아니라 langchain Document 다. 어느 공고 어느 절에서
   왔는지가 metadata 에 붙어 있어야 근거를 표시하고 요약 카드를 만들 수 있다.

**부품을 새로 만들려면 `__call__(self, state)` 하나만 쓰면 된다.**
상속할 부모 클래스는 없다.

    class DropShort:
        '''글자 수가 적은 청크를 버린다.'''

        def __init__(self, min_chars=100):
            self.min_chars = min_chars

        def __call__(self, state):
            state.chunks = [c for c in state.chunks
                            if len(c.page_content) >= self.min_chars]
            return state

        def __repr__(self):
            return f"DropShort(min_chars={self.min_chars})"

조립대는 nn.Sequential 자리다.

    rag = Pipeline([
        Dense(store, k=20),
        Rerank(reranker, k=5),
    ])
    result = rag("과업기간이 가장 짧은 공고는?")
"""

from dataclasses import dataclass, field


@dataclass
class State:
    """부품 사이를 흘러가는 상자.

    question  사용자가 처음 물어본 것. 부품이 바꾸지 않는다.
    queries   실제로 검색에 쓸 질문들. 지금은 [question] 하나다.
    chunks    지금까지 모인 청크. 검색이 채우고 정제가 줄인다.
    log       부품마다 "내가 뭘 했는지" 한 줄씩. trace() 로 본다.
    """

    question: str
    queries: list = field(default_factory=list)
    chunks: list = field(default_factory=list)
    log: list = field(default_factory=list)

    def note(self, message):
        """부품이 한 일을 기록한다."""
        self.log.append(message)

    @property
    def context(self):
        """청크들을 프롬프트에 넣을 하나의 문자열로 이어 붙인다."""
        return "\n\n".join(chunk.page_content for chunk in self.chunks)

    def show(self, chars=200):
        """지금 상태를 사람이 보기 좋게 출력한다."""
        print("질문:", self.question)
        if self.queries and self.queries != [self.question]:
            print("검색에 쓴 질문:")
            for query in self.queries:
                print("   -", query)
        print(f"청크 {len(self.chunks)}개")
        for i, chunk in enumerate(self.chunks, 1):
            title = chunk.metadata.get("title", "")[:30]
            score = chunk.metadata.get("score")
            score_text = f" (점수 {score:.4f})" if score is not None else ""
            body = chunk.page_content[:chars].replace("\n", " ⏎ ")
            print(f"  {i}. [{title}]{score_text}")
            print(f"     {body}…")

    def trace(self):
        """부품들이 남긴 기록을 순서대로 출력한다."""
        for line in self.log:
            print(" ", line)


def name_of(piece):
    """부품 이름. 클래스 이름을 그대로 쓴다. 상속이 없으니 이걸로 충분하다."""
    return type(piece).__name__


class Pipeline:
    """부품을 순서대로 실행한다. nn.Sequential 자리다."""

    def __init__(self, pieces):
        self.pieces = list(pieces)

    def run(self, question, verbose=False):
        state = State(question=question, queries=[question])
        for piece in self.pieces:
            state = piece(state)
            if verbose:
                last = state.log[-1] if state.log else ""
                print(f"[{name_of(piece)}] {last}")
        return state

    def __call__(self, question, verbose=False):
        return self.run(question, verbose=verbose)

    # --- 끼우고 빼기 ---------------------------------------------------
    #
    # 전부 **새 조립대**를 돌려준다. 원래 것은 그대로 남으므로
    # 하나를 여러 갈래로 변형해 나란히 비교할 수 있다.

    def without(self, name):
        """이름이 같은 부품을 뺀다.   rag.without("Rerank")"""
        return Pipeline([p for p in self.pieces if name_of(p) != name])

    def replace(self, name, new_piece):
        """이름이 같은 부품을 다른 것으로 바꾼다."""
        return Pipeline([new_piece if name_of(p) == name else p for p in self.pieces])

    def insert(self, position, piece):
        """position 자리에 부품을 끼운다."""
        pieces = list(self.pieces)
        pieces.insert(position, piece)
        return Pipeline(pieces)

    def add(self, piece):
        """맨 뒤에 부품을 붙인다."""
        return Pipeline([*self.pieces, piece])

    def __getitem__(self, i):
        if isinstance(i, slice):
            return Pipeline(self.pieces[i])
        return self.pieces[i]

    def __len__(self):
        return len(self.pieces)

    def __repr__(self):
        lines = ["Pipeline("]
        lines += [f"  ({i}) {piece!r}" for i, piece in enumerate(self.pieces)]
        lines.append(")")
        return "\n".join(lines)


def dedup_chunks(chunks):
    """같은 청크가 여러 번 들어온 것을 하나로 줄인다. 순서는 유지한다.

    강의에서 dict 로 중복을 없앴던 것과 같은 방식이다.
    """
    seen = {}
    for chunk in chunks:
        key = chunk.metadata.get("chunk_id") or chunk.page_content[:120]
        if key not in seen:
            seen[key] = chunk
    return list(seen.values())
