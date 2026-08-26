"""평가.

    import evaluate as ev

    pairs = ev.make_pairs_from_documents(documents)   # 질문 만들기
    ev.compare({"BM25": ...}, pairs)                  # 검색 비교 (공짜)
    ev.evaluate_answers(rag, pairs, judge_llm=llm)    # 충실성 (LLM 씀)

파일이 셋으로 갈려 있다.

    evalset.py      질문 세트 만들기·저장
    retrieval.py    검색 지표 — 적중률, MRR, compare
    generation.py   생성 지표 — 근거표시율, 물러섬, 충실성
"""

from evaluation.evalset import (
    FIELD_PATTERNS,
    load_evalset,
    make_pairs_from_documents,
    matches,
    normalize,
    save_evalset,
)
from evaluation.generation import (
    JUDGE_SYSTEM,
    cite_rate,
    evaluate_answers,
    judge_faithfulness,
    number_match,
    said_no_info,
)
from evaluation.retrieval import (
    compare,
    doc_hit_rate,
    hit_rate,
    mrr,
    score_all,
)

__all__ = [
    "FIELD_PATTERNS",
    "JUDGE_SYSTEM",
    "cite_rate",
    "compare",
    "doc_hit_rate",
    "evaluate_answers",
    "hit_rate",
    "judge_faithfulness",
    "load_evalset",
    "make_pairs_from_documents",
    "matches",
    "mrr",
    "normalize",
    "number_match",
    "said_no_info",
    "save_evalset",
    "score_all",
]
