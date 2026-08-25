"""
Generation 실행 파일

바깥에서 호출하는 실행 함수는 generate_answer() 딱 1개뿐입니다.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MODEL_CONFIGS, ModelConfig, OPENAI_API_KEY, MAX_CONTEXT_CHARS


SYSTEM_PROMPT = """당신은 B2G 입찰 컨설팅 회사 입찰메이트의 RFP 분석 어시스턴트입니다.
주어진 컨텍스트(RFP 문서 조각)만을 근거로 답변하세요.
컨텍스트에 없는 내용은 문서에서 확인되지 않습니다 라고 답하세요.
불필요한 미사여구 없이 핵심만 정리해서 답변하세요."""


def _run_openai(cfg: ModelConfig, messages: list[dict]) -> str:
    from openai import OpenAI, BadRequestError

    client = OpenAI(api_key=OPENAI_API_KEY)

    params = {
        "model": cfg.model,
        "messages": messages,
        "max_completion_tokens": 1000,
        "reasoning_effort": cfg.reasoning_effort,
        "verbosity": cfg.verbosity,
    }

    for _ in range(len(params)):
        try:
            response = client.chat.completions.create(**params)
            return response.choices[0].message.content
        except BadRequestError as e:
            bad_param = getattr(e, "param", None) or (e.body or {}).get("error", {}).get("param")
            if bad_param and bad_param in params:
                print(f"[generate_answer] {bad_param} 파라미터를 이 모델이 지원하지 않아 제외하고 재시도합니다.")
                del params[bad_param]
                continue
            raise

    raise RuntimeError("OpenAI API 호출에 반복적으로 실패했습니다.")


_hf_pipeline_cache: dict[str, object] = {}


def _get_hf_pipeline(cfg: ModelConfig):
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
            torch_dtype=dtype_map.get(cfg.dtype, torch.bfloat16),
            device_map=cfg.device_map,
        )
    return _hf_pipeline_cache[cfg.model]


def _run_huggingface(cfg: ModelConfig, messages: list[dict]) -> str:
    pipe = _get_hf_pipeline(cfg)

    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    outputs = pipe(
        prompt,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=False,
        **cfg.extra,
    )
    generated_text = outputs[0]["generated_text"]
    return generated_text[len(prompt):].strip()


_PROVIDER_RUNNERS = {
    "openai": _run_openai,
    "huggingface": _run_huggingface,
}


def generate_answer(
    model_key: str,
    query: str,
    context: str,
    history: Optional[list[dict]] = None,
) -> str:
    if model_key not in MODEL_CONFIGS:
        raise ValueError(
            f"알 수 없는 model_key입니다: {model_key}. "
            f"사용 가능한 값: {list(MODEL_CONFIGS.keys())}"
        )

    cfg = MODEL_CONFIGS[model_key]

    if cfg.provider not in _PROVIDER_RUNNERS:
        raise ValueError(f"지원하지 않는 provider입니다: {cfg.provider}")

    trimmed_context = context[:MAX_CONTEXT_CHARS]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append(
        {"role": "user", "content": f"[컨텍스트]\n{trimmed_context}\n\n[질문]\n{query}"}
    )

    print(
        f"[generate_answer] model_key={model_key} -> "
        f"provider={cfg.provider}, model={cfg.model}"
    )

    runner = _PROVIDER_RUNNERS[cfg.provider]
    return runner(cfg, messages)