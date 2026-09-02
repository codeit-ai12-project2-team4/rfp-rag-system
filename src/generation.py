"""
Generation 실행 파일

바깥에서 호출하는 실행 함수는 generate_answer() 딱 1개뿐입니다.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time

# 프로젝트 루트 (config를 찾기 위함) 와 src (models 를 찾기 위함)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import MAX_CONTEXT_CHARS, MODEL_CONFIGS, OPENAI_API_KEY, ModelConfig

# ---------------------------------------------------------------------------
# 기본값 상수 모음 (팀 회의로 값이 바뀌면 이 블록만 수정하면 됩니다)
# ---------------------------------------------------------------------------
DEFAULT_MAX_TOKENS_OPENAI = 5000
"""gpt-5 계열은 답변 전에 추론 토큰을 먼저 소모하므로, 여유 있게 잡는다.
1000이었을 때 추론에 예산을 다 써서 답변이 빈 문자열로 나오는 문제가 있었음(2026-08-28 회의 결정)."""

DEFAULT_MAX_TOKENS_HF = 512
"""HuggingFace 로컬 모델(시나리오 A)의 기본 생성 길이."""

DEFAULT_JUDGE_MAX_TOKENS = 2000
"""ask()의 기본값. YES/NO 같은 짧은 채점용이라 작게 잡는다."""

SYSTEM_PROMPT = """당신은 B2G 입찰 컨설팅 회사 입찰메이트의 RFP 분석 어시스턴트입니다.
주어진 컨텍스트(RFP 문서 조각)만을 근거로 답변하세요.
컨텍스트에 없는 내용은 "문서에서 확인되지 않습니다"라고 답하세요.
불필요한 미사여구 없이 핵심만 정리해서 답변하세요.
답변에서 특정 사실을 인용할 때는 그 근거가 된 컨텍스트 번호를 문장 끝에
[1], [2]와 같은 형식으로 표기하세요."""


def _result(
    ok: bool,
    model_key: str,
    cfg: ModelConfig | None = None,
    answer: str | None = None,
    usage: dict | None = None,
    latency_sec: float | None = None,
    error: str | None = None,
) -> dict:
    """generate_answer()가 항상 동일한 형태로 반환하도록 결과 딕셔너리를 만든다.

    Args:
        ok: 호출 성공 여부.
        model_key: 사용자가 요청한 model_key (config에 등록된 키).
        cfg: 해당 model_key의 ModelConfig. 조회 자체가 실패한 경우 None일 수 있음.
        answer: 모델이 생성한 답변 텍스트.
        usage: {"input_tokens": int, "output_tokens": int} 형태의 토큰 사용량.
        latency_sec: API 호출부터 응답까지 걸린 시간(초).
        error: 실패 시 에러 메시지.

    Returns:
        고정된 키 구성을 가진 결과 딕셔너리.
    """
    return {
        "ok": ok,
        "model_key": model_key,
        "provider": cfg.provider if cfg else None,
        "model": cfg.model if cfg else None,
        "answer": answer,
        "usage": usage,
        "latency_sec": latency_sec,
        "error": error,
    }


def _build_messages(
    query: str, context: str, history: list[dict] | None = None
) -> list[dict]:
    """system 프롬프트 + 대화 히스토리 + 이번 턴(컨텍스트+질문)을 메시지 배열로 조립한다.

    Args:
        query: 사용자 질문.
        context: RFP 문서에서 가져온 컨텍스트 텍스트. MAX_CONTEXT_CHARS를 넘으면 잘라낸다.
        history: 이전 대화 턴들. [{"role": "user"/"assistant", "content": str}, ...] 형태.

    Returns:
        OpenAI Chat Completions / HuggingFace chat template에 그대로 넣을 수 있는 메시지 리스트.
    """
    trimmed_context = context[:MAX_CONTEXT_CHARS]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append(
        {"role": "user", "content": f"[컨텍스트]\n{trimmed_context}\n\n[질문]\n{query}"}
    )
    return messages


def _run_openai(
    cfg: ModelConfig, messages: list[dict], max_tokens: int | None = None
) -> tuple[str, dict | None]:
    """OpenAI Chat Completions API로 실제 호출을 수행한다 (시나리오 B).

    reasoning_effort/verbosity 등 이 모델이 지원하지 않는 파라미터가 있으면
    해당 파라미터만 제거하고 자동으로 재시도한다.

    Args:
        cfg: 사용할 모델의 ModelConfig.
        messages: _build_messages()로 만든 메시지 배열.
        max_tokens: 생성할 최대 토큰 수. 생략하면 DEFAULT_MAX_TOKENS_OPENAI.
            judge_faithfulness()처럼 YES/NO 한 단어만 필요할 때는 짧게 넘긴다.

    Returns:
        (answer, usage) 튜플. usage는 {"input_tokens": int, "output_tokens": int} 또는 None.

    Raises:
        RuntimeError: 파라미터 제거를 반복해도 계속 실패하는 경우.
        openai.APIError 계열: 파라미터 문제가 아닌 API 오류(레이트리밋/인증/네트워크 등).
            이 예외는 여기서 잡지 않고 그대로 올려보내며, generate_answer()에서
            최종적으로 잡아 고정 스키마로 변환한다.
    """
    from openai import BadRequestError, OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    params = {
        "model": cfg.model,
        "messages": messages,
        "max_completion_tokens": max_tokens or DEFAULT_MAX_TOKENS_OPENAI,
        "reasoning_effort": cfg.reasoning_effort,
        "verbosity": cfg.verbosity,
    }

    max_attempts = len(params)
    for _ in range(max_attempts):
        try:
            response = client.chat.completions.create(**params)
            answer = response.choices[0].message.content
            usage = None
            if response.usage:
                usage = {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                }
            return answer, usage
        except BadRequestError as e:
            bad_param = getattr(e, "param", None) or (e.body or {}).get(
                "error", {}
            ).get("param")
            if bad_param and bad_param in params:
                print(
                    f"[generate_answer] {bad_param} 파라미터를 이 모델이 지원하지 않아 제외하고 재시도합니다."
                )
                del params[bad_param]
                continue
            raise

    raise RuntimeError("OpenAI API 호출에 반복적으로 실패했습니다.")


_hf_pipeline_cache: dict[str, object] = {}


def _get_hf_pipeline(cfg: ModelConfig):
    """HuggingFace 모델 파이프라인을 로드하고 캐싱한다 (시나리오 A).

    Args:
        cfg: 사용할 모델의 ModelConfig. dtype/device_map 필드가 정의돼 있어야 한다.

    Returns:
        transformers의 text-generation 파이프라인 객체.
    """
    if cfg.model not in _hf_pipeline_cache:
        import torch
        from transformers import pipeline

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        print(f"[generate_answer] HuggingFace 모델 {cfg.model} 로딩 중...")
        _hf_pipeline_cache[cfg.model] = pipeline(
            "text-generation",
            model=cfg.model,
            torch_dtype=dtype_map.get(getattr(cfg, "dtype", None), torch.bfloat16),
            device_map=getattr(cfg, "device_map", "auto"),
        )
    return _hf_pipeline_cache[cfg.model]


def _run_huggingface(
    cfg: ModelConfig, messages: list[dict], max_tokens: int | None = None
) -> tuple[str, dict | None]:
    """HuggingFace 로컬 모델로 실제 호출을 수행한다 (시나리오 A).

    Args:
        cfg: 사용할 모델의 ModelConfig.
        messages: _build_messages()로 만든 메시지 배열.
        max_tokens: 생성할 최대 토큰 수. 생략하면 cfg.max_new_tokens 또는
            DEFAULT_MAX_TOKENS_HF를 쓴다.

    Returns:
        (answer, usage) 튜플. 로컬 모델은 토큰 사용량 계측이 없으면 usage=None.
    """
    pipe = _get_hf_pipeline(cfg)
    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    outputs = pipe(
        prompt,
        max_new_tokens=max_tokens or getattr(cfg, "max_new_tokens", DEFAULT_MAX_TOKENS_HF),
        do_sample=False,
        **getattr(cfg, "extra", {}),
    )
    generated_text = outputs[0]["generated_text"]
    answer = generated_text[len(prompt) :].strip()
    return answer, None


SGLANG_URL = os.environ.get("SGLANG_URL", "http://localhost:8087")


def _run_sglang(
    cfg: ModelConfig, messages: list[dict], max_tokens: int | None = None
) -> tuple[str, dict | None]:
    """VM 의 SGLang 서버로 호출한다 (시나리오 A).

    SGLang 은 OpenAI 규격과 호환이라 base_url 만 바꾸면 _run_openai 와 같은
    코드다. 다른 점은 **부르기 전에 그 모델이 올라와 있는지 확인**하는 것뿐이다.
    L4 한 장에 여러 모델을 못 띄워서, 다른 모델이 올라와 있으면 갈아끼운다.
    처음 받는 8B 면 몇 분 걸린다 — 그동안 이 호출은 그냥 기다린다.

    Args:
        cfg: 사용할 모델의 ModelConfig. mem/args 필드가 서버 인자로 들어간다.
        messages: _build_messages()로 만든 메시지 배열.
        max_tokens: 생성할 최대 토큰 수. 생략하면 cfg.max_new_tokens.

    Returns:
        (answer, usage) 튜플.

    Raises:
        RuntimeError: 모델 교체가 제한 시간 안에 안 끝날 때.
    """
    from openai import OpenAI

    from models import sglang

    sglang.ensure(cfg.model, mem=cfg.mem or "0.45", args=cfg.args)
    client = OpenAI(base_url=f"{SGLANG_URL}/v1", api_key="local")
    response = client.chat.completions.create(
        model=cfg.model,
        messages=messages,
        max_tokens=max_tokens or cfg.max_new_tokens or DEFAULT_MAX_TOKENS_HF,
        # 샘플링을 전부 명시한다. **안 적으면 모델의 generation_config 가 이긴다.**
        # Qwen2.5 는 temperature 0.7 / top_p 0.8 / top_k 20 / repetition_penalty 1.05
        # 를 들고 있고, SGLang 이 그걸 기본값으로 쓴다("Using default chat sampling
        # params from model generation config" 로그). temperature 만 0 으로 눌러도
        # **repetition_penalty 는 그리디 디코딩에서도 로짓을 건드린다.**
        # RFP 답변은 같은 사업명·금액 단위를 반복해서 써야 하는데 거기에 벌점이 붙는다.
        # 모델마다 generation_config 가 달라서 비교도 못 하게 된다.
        temperature=0.0,
        top_p=1.0,
        extra_body={"top_k": -1, "repetition_penalty": 1.0},
    )
    usage = None
    if response.usage:
        usage = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }
    return (response.choices[0].message.content or "").strip(), usage


_PROVIDER_RUNNERS = {
    "openai": _run_openai,
    "sglang": _run_sglang,
    # 지금 MODEL_CONFIGS 에는 huggingface 항목이 없다. 평가 스크립트에서
    # 프로세스 안에 직접 올려 재고 싶을 때를 위해 남겨 둔다.
    "huggingface": _run_huggingface,
}


def generate_answer(
    model_key: str,
    query: str,
    context: str,
    history: list[dict] | None = None,
) -> dict:
    """질문 + 컨텍스트로 답변을 생성한다. 어떤 모델/시나리오를 쓸지는 model_key로만 결정한다.

    예외를 던지지 않는다: model_key가 잘못됐거나 API 호출이 실패해도 항상 동일한
    형태의 딕셔너리({"ok": False, "error": ...})를 반환한다. 여러 건을 반복 호출하는
    배치/평가 스크립트에서 한 건의 실패가 전체를 중단시키지 않도록 하기 위함이다.

    Args:
        model_key: config.MODEL_CONFIGS에 등록된 키 (예: "mini", "nano").
        query: 사용자 질문.
        context: RFP 문서에서 가져온 컨텍스트 텍스트.
        history: 이전 대화 턴들 (선택). [{"role": ..., "content": ...}, ...] 형태.

    Returns:
        {
            "ok": bool,
            "model_key": str,
            "provider": str | None,
            "model": str | None,
            "answer": str | None,
            "usage": {"input_tokens": int, "output_tokens": int} | None,
            "latency_sec": float | None,
            "error": str | None,
        }
    """
    if model_key not in MODEL_CONFIGS:
        return _result(
            ok=False,
            model_key=model_key,
            error=f"알 수 없는 model_key입니다: {model_key}. 사용 가능한 값: {list(MODEL_CONFIGS.keys())}",
        )

    cfg = MODEL_CONFIGS[model_key]
    runner = _PROVIDER_RUNNERS.get(cfg.provider)
    if runner is None:
        return _result(
            ok=False,
            model_key=model_key,
            cfg=cfg,
            error=f"지원하지 않는 provider입니다: {cfg.provider}",
        )

    messages = _build_messages(query, context, history)
    print(
        f"[generate_answer] model_key={model_key} -> provider={cfg.provider}, model={cfg.model}"
    )

    started = time.time()
    try:
        answer, usage = runner(cfg, messages)
    except Exception as e:  # noqa: BLE001 - 어떤 이유로 실패하든 고정 스키마로 반환
        return _result(
            ok=False,
            model_key=model_key,
            cfg=cfg,
            latency_sec=time.time() - started,
            error=str(e),
        )

    return _result(
        ok=True,
        model_key=model_key,
        cfg=cfg,
        answer=answer,
        usage=usage,
        latency_sec=time.time() - started,
    )


def run_comparison(
    model_keys: list[str],
    query: str,
    context: str,
    history: list[dict] | None = None,
) -> list[dict]:
    """여러 model_key(시나리오 A/B 혼합 가능)를 동일 입력으로 순차 실행해 결과를 모은다.

    Args:
        model_keys: 비교할 model_key 목록 (예: ["mini", "nano"]).
        query: 사용자 질문.
        context: RFP 문서에서 가져온 컨텍스트 텍스트.
        history: 이전 대화 턴들 (선택).

    Returns:
        generate_answer()의 반환값을 model_keys 순서대로 담은 리스트.
    """
    return [
        generate_answer(model_key=key, query=query, context=context, history=history)
        for key in model_keys
    ]


# --- 평가(evaluation) 연동용 어댑터 ---------------------------------------
#
# src/evaluation/generation.py의 judge_faithfulness()는
# judge_llm.ask(system, user, max_tokens) 메서드를 가진 객체를 전제로 만들어져
# 있다. generate_answer()는 이 형태와 다르므로(딕셔너리 반환, model_key 방식),
# 아래 ask()/AskableModel이 그 차이를 메운다.
#
# evaluate_answers()가 기대하는 다른 하나 — pipeline(question)을 호출하면
# .answer / .context 속성을 가진 결과가 나오는 것 — 은 검색(retriever.py)까지
# 엮어야 하므로 이 파일이 아니라 src/pipeline.py의 GenerationPipeline이 맡는다.


def ask(
    model_key: str, system: str, user: str, max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS
) -> str:
    """RFP 질의응답 전용 프롬프트 없이, 시스템/사용자 메시지를 그대로 LLM에 넘긴다.

    generate_answer()는 시스템 프롬프트와 "[컨텍스트]/[질문]" 형식을 강제로 씌우기
    때문에, judge_faithfulness()처럼 임의의 시스템 프롬프트로 채점만 시키는 범용
    LLM 호출에는 쓸 수 없다. 이 함수는 그 틀 없이 순수하게 system/user만 넘긴다.

    generate_answer()와 마찬가지로 예외를 던지지 않는다: 실패하면 빈 문자열을
    반환한다.

    Args:
        model_key: config.MODEL_CONFIGS에 등록된 키 (예: "nano").
        system: 시스템 프롬프트.
        user: 사용자 메시지.
        max_tokens: 생성할 최대 토큰 수. 기본값은 DEFAULT_JUDGE_MAX_TOKENS(짧은 채점용).

    Returns:
        모델이 생성한 텍스트. model_key가 잘못됐거나 호출이 실패하면 빈 문자열.
    """
    if model_key not in MODEL_CONFIGS:
        return ""
    cfg = MODEL_CONFIGS[model_key]
    runner = _PROVIDER_RUNNERS.get(cfg.provider)
    if runner is None:
        return ""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        answer, _usage = runner(cfg, messages, max_tokens=max_tokens)
    except Exception:  # noqa: BLE001 - 채점 실패로 평가 전체가 죽으면 안 됨
        return ""
    return answer or ""


class AskableModel:
    """judge_faithfulness()가 기대하는 `.ask(system, user, max_tokens)` 형태로
    generate_answer() 쪽 모델을 감싼 얇은 어댑터.

    Example:
        from evaluation import evaluate_answers
        from generation import AskableModel
        from pipeline import GenerationPipeline

        judge = AskableModel("nano")  # 채점은 싼 모델로 충분
        evaluate_answers(GenerationPipeline("mini"), pairs, judge_llm=judge)
    """

    def __init__(self, model_key: str = "nano"):
        self.model_key = model_key
        self.name = model_key  # models/llm.py 쪽 LLM 객체들과 인터페이스를 맞춤

    def ask(self, system: str, user: str, max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS) -> str:
        return ask(self.model_key, system, user, max_tokens=max_tokens)

    def __repr__(self) -> str:
        return f"AskableModel({self.model_key!r})"