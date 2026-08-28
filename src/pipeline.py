"""
Pipeline 실행 파일

검색(src/retriever.py)과 생성(src/generation.py)을 하나로 묶어, 질문 한 줄로
"검색 → 컨텍스트 구성 → 답변 생성"까지 한 번에 돌아가게 합니다.

    from src.pipeline import GenerationPipeline

    pipeline = GenerationPipeline(model_key="mini")
    result = pipeline("이 사업의 예산이 얼마야?")
    print(result.answer)

src/evaluation/generation.py의 evaluate_answers()가 기대하는 형태
(`pipeline(question)` 을 호출하면 `.answer` / `.context` 속성을 가진 결과가
나오는 것)에도 그대로 맞으므로, 평가 코드에서도 이 파일의 GenerationPipeline을
그대로 가져다 쓸 수 있습니다.

    from src.pipeline import GenerationPipeline
    from src.generation import AskableModel
    from src.evaluation import evaluate_answers, load_evalset

    pipeline = GenerationPipeline(model_key="mini")
    pairs = load_evalset("eval_qa")
    df = evaluate_answers(pipeline, pairs, judge_llm=AskableModel("nano"))
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

# `python src/pipeline.py` 로 직접 돌릴 때도, 다른 폴더에서
# `from src.pipeline import ...` 로 불러올 때도 똑같이 되도록 프로젝트 루트와
# src/ 를 sys.path 에 넣습니다. (retriever.py / chunking.py 등과 같은 패턴)
_ROOT = Path(__file__).resolve().parents[1]
for _folder in (_ROOT / "src", _ROOT):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

from generation import generate_answer
from retriever import retrieve_context


@dataclass
class PipelineResult:
    """evaluate_answers()가 기대하는 state 모양(.answer / .context)에 맞춘 결과 상자.

    pieces.base.State와 속성 이름(answer, context)을 맞춰서, evaluate_answers()가
    둘 중 어느 쪽이 왔는지 구분할 필요 없이 그대로 쓸 수 있게 했다.

    Attributes:
        question: 사용자 질문.
        context: 답변 생성에 실제로 쓰인 컨텍스트 문자열 (retrieve_context() 결과).
        answer: 생성된 답변 텍스트.
        result: generate_answer()의 원본 반환값. ok/usage/latency_sec 등을
            확인하고 싶을 때 여기서 꺼내 쓴다.
    """

    question: str
    context: str
    answer: str
    result: dict


class GenerationPipeline:
    """retrieve_context() + generate_answer() 를 묶어, 질문 하나로 답변까지 낸다.

    evaluate_answers(pipeline, pairs, ...) 는 `pipeline(question)` 을 호출해서
    .answer / .context 속성을 가진 결과를 기대한다. retriever.py 는 검색만,
    generate_answer() 는 생성만 담당하므로, 둘을 하나의 호출로 묶어주는 역할이
    필요해서 만들었다. main.py도 이 클래스를 통해 프로그램을 실행한다.

    Example:
        pipeline = GenerationPipeline(model_key="mini")
        result = pipeline("이 사업의 예산이 얼마야?")
        print(result.answer)
    """

    def __init__(self, model_key: str = "mini", **retrieve_kwargs):
        """
        Args:
            model_key: 답변 생성에 쓸 config.MODEL_CONFIGS 의 키.
            **retrieve_kwargs: retriever.retrieve_context() 에 그대로 넘길 인자
                (doc_ids, top_k, index, embed, rerank 등).
        """
        self.model_key = model_key
        self.retrieve_kwargs = retrieve_kwargs

    def __call__(self, question: str) -> PipelineResult:
        context = retrieve_context(question, **self.retrieve_kwargs)
        result = generate_answer(
            model_key=self.model_key, query=question, context=context
        )
        return PipelineResult(
            question=question,
            context=context,
            answer=result.get("answer") or "",
            result=result,
        )

    def __repr__(self) -> str:
        return f"GenerationPipeline(model_key={self.model_key!r})"
