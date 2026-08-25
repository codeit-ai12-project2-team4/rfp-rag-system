"""청크를 다듬는 부품 — 검색 **후**, 생성 **전**에 끼운다.

검색은 넓게 뽑고(k=20~30), 여기서 좁힌다(k=3~5). 넓게 뽑아야 정답이
후보에 들어오고, 좁혀야 프롬프트가 짧아지고 LLM 이 헷갈리지 않는다.

Rerank    질문과 청크를 쌍으로 다시 채점해 순서를 고친다
Compress  청크 안에서 질문과 상관없는 문장을 지운다
TopK      그냥 앞에서 n개만
Widen     찾은 청크의 앞뒤 청크까지 끌어온다
"""

import numpy as np

from pieces.base import dedup_chunks


class Rerank:
    """질문과 청크를 한 쌍씩 다시 채점해 순서를 고친다.

    임베딩 검색은 질문 벡터와 청크 벡터를 **따로** 만들어 비교한다. 빠르지만
    거칠다. 리랭커(크로스 인코더)는 질문과 청크를 **같이** 모델에 넣어서 본다.
    느리지만 정확하다. 그래서 후보를 좁힐 때만 쓴다.

    강의 L05 의 CrossEncoder 와 같은 일을 한다. 다만 모델을 노트북 안에 올리지
    않고 TEI 서버에 맡긴다 — 커널을 재시작해도 모델이 안 죽고, GPU 메모리를
    노트북이 붙들고 있지 않는다.
    """

    def __init__(self, reranker, k=5):
        self.reranker = reranker
        self.k = k

    def __call__(self, state):
        if not state.chunks:
            state.note("청크가 없어 건너뜀")
            return state

        before = len(state.chunks)
        texts = [chunk.page_content for chunk in state.chunks]
        scores = self.reranker.score(state.question, texts)

        ranked = sorted(
            zip(scores, state.chunks, strict=False),
            key=lambda pair: pair[0],
            reverse=True,
        )
        picked = []
        for score, chunk in ranked[: self.k]:
            chunk.metadata["score"] = float(score)
            picked.append(chunk)

        state.chunks = picked
        state.note(f"다시 채점 {before} → {len(picked)}개")
        return state

    def __repr__(self):
        return f"Rerank(k={self.k})"


class Compress:
    """청크 안에서 질문과 관련 없는 문장을 지운다.

    청크 하나가 1,000자여도 정작 답은 한 문장인 경우가 많다. 나머지는
    프롬프트만 길게 만들고 LLM 을 헷갈리게 한다.

    문장마다 질문과의 코사인 유사도를 재서 threshold 아래를 버린다.
    강의 L05 의 compress_chunk 와 같은 방식이다.

    threshold 를 너무 올리면 정답 문장까지 날아간다. 0.3~0.5 사이에서
    노트북으로 직접 보면서 정하는 게 낫다.
    """

    def __init__(self, embedder, threshold=0.35, min_sentences=1):
        self.embedder = embedder
        self.threshold = threshold
        self.min_sentences = min_sentences

    def split_sentences(self, text):
        """줄바꿈과 마침표로 문장을 나눈다. RFP 는 개조식이라 줄바꿈이 더 중요하다."""
        parts = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            if len(line) < 80:
                parts.append(line)
            else:
                for piece in line.split(". "):
                    piece = piece.strip()
                    if piece:
                        parts.append(piece)
        return parts

    def __call__(self, state):
        if not state.chunks:
            return state

        question_vector = np.array(self.embedder.embed_query(state.question))
        kept_total = dropped_total = 0

        for chunk in state.chunks:
            sentences = self.split_sentences(chunk.page_content)
            if len(sentences) <= self.min_sentences:
                continue

            vectors = np.array(self.embedder.embed_documents(sentences))
            similarities = (
                vectors
                @ question_vector
                / (
                    np.linalg.norm(vectors, axis=1) * np.linalg.norm(question_vector)
                    + 1e-9
                )
            )

            kept = [
                s
                for s, sim in zip(sentences, similarities, strict=False)
                if sim >= self.threshold
            ]
            # 전부 잘리면 제일 비슷한 문장만이라도 남긴다
            if len(kept) < self.min_sentences:
                best = int(np.argmax(similarities))
                kept = [sentences[best]]

            kept_total += len(kept)
            dropped_total += len(sentences) - len(kept)
            chunk.page_content = "\n".join(kept)

        state.note(f"문장 {kept_total}개 남기고 {dropped_total}개 버림")
        return state

    def __repr__(self):
        return f"Compress(threshold={self.threshold})"


class TopK:
    """앞에서 n개만 남긴다. 제일 단순한 정제."""

    def __init__(self, k=5):
        self.k = k

    def __call__(self, state):
        before = len(state.chunks)
        state.chunks = state.chunks[: self.k]
        state.note(f"{before} → {len(state.chunks)}개")
        return state

    def __repr__(self):
        return f"TopK(k={self.k})"


class Widen:
    """찾은 청크의 앞뒤 청크까지 끌어온다.

    RFP 표는 라벨과 값이 다른 청크로 갈리는 일이 흔하다.

        …(청크 12)  요구사항 고유번호  SFR-002
        …(청크 13)  요구사항 명칭     하이브리드앱 환경 전환

    청크 12만 찾아오면 무슨 요구사항인지 모른다. 앞뒤를 같이 가져오면 붙는다.
    작은 청크로 자를수록 이 부품이 중요해진다.
    """

    def __init__(self, all_chunks, before=1, after=1, max_total=12):
        self.before = before
        self.after = after
        self.max_total = max_total
        # 문서별로 순서대로 정리해 둔다
        self.by_doc = {}
        for chunk in all_chunks:
            self.by_doc.setdefault(chunk.metadata.get("doc_id"), []).append(chunk)
        for chunks in self.by_doc.values():
            chunks.sort(key=lambda c: c.metadata.get("order", 0))

    def __call__(self, state):
        widened = []
        for chunk in state.chunks:
            siblings = self.by_doc.get(chunk.metadata.get("doc_id"), [])
            order = chunk.metadata.get("order", 0)
            index = next(
                (i for i, s in enumerate(siblings) if s.metadata.get("order") == order),
                None,
            )
            if index is None:
                widened.append(chunk)
                continue
            start = max(0, index - self.before)
            end = min(len(siblings), index + self.after + 1)
            widened.extend(siblings[start:end])

        before = len(state.chunks)
        state.chunks = dedup_chunks(widened)[: self.max_total]
        state.note(f"앞뒤로 넓힘 {before} → {len(state.chunks)}개")
        return state

    def __repr__(self):
        return f"Widen(앞{self.before} 뒤{self.after})"
