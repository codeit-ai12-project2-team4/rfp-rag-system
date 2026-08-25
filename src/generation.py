"""
Generation 실행 파일

이 파일에는 실행 함수 generate_answer() 딱 1개만 있습니다.
어떤 모델(mini/nano)을 쓸지는 config/model_config.py에서 결정되고,
이 함수는 그 설정을 그대로 읽어와서 실행만 합니다.

사용 예시:
    from src.generation import generate_answer
    answer = generate_answer(query="이 RFP의 예산은 얼마야?", context="...검색된 문서 내용...")
"""

import sys
import os
from openai import OpenAI

# config 폴더를 import 경로에 추가 (프로젝트 루트 기준)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.model_config import get_active_model_name, get_active_params, ACTIVE_MODEL

client = OpenAI()  # 환경변수 OPENAI_API_KEY 사용


SYSTEM_PROMPT = """당신은 B2G 입찰 컨설팅 회사 '입찰메이트'의 RFP 분석 어시스턴트입니다.
주어진 컨텍스트(RFP 문서 조각)만을 근거로 답변하세요.
컨텍스트에 없는 내용은 "문서에서 확인되지 않습니다"라고 답하세요.
불필요한 미사여구 없이 핵심만 정리해서 답변하세요."""


def generate_answer(query: str, context: str, history: list = None) -> str:
    """
    RFP 컨텍스트를 바탕으로 사용자 질문에 답변하는 유일한 실행 함수.

    모델 선택은 config/model_config.py의 ACTIVE_MODEL 값을 그대로 따릅니다.
    이 함수 자체는 어떤 모델을 쓰는지 신경 쓰지 않고, config에서 읽어온
    이름과 파라미터로 API만 호출합니다.

    Args:
        query: 사용자 질문
        context: Retrieval 단계에서 찾아온 관련 문서 조각(들)
        history: [{"role": "user"/"assistant", "content": "..."}] 형태의 이전 대화 (선택)

    Returns:
        모델이 생성한 답변 문자열
    """
    model_name = get_active_model_name()   # 예: "gpt-5-mini" 또는 "gpt-5-nano"
    params = get_active_params()           # 예: {"temperature": 0.2, "max_tokens": 800}

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history:
        messages.extend(history)

    messages.append({
        "role": "user",
        "content": f"[컨텍스트]\n{context}\n\n[질문]\n{query}"
    })

    print(f"[generate_answer] 현재 사용 모델: {ACTIVE_MODEL} -> {model_name}")

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=params["max_tokens"]
    )

    return response.choices[0].message.content
