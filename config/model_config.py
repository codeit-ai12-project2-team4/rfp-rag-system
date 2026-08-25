"""
모델 관련 설정 파일

이 파일에서 값만 바꾸면 src/generation.py의 실행 함수가
자동으로 해당 모델/옵션을 사용하게 됩니다.
팀원들은 코드를 건드릴 필요 없이 이 파일만 수정하면 됩니다.
"""

# ============================================
# 1. 사용할 모델 선택
#    "mini" 또는 "nano" 중 하나로 설정
# ============================================
ACTIVE_MODEL = "nano"  # <-- 여기만 바꾸면 됩니다: "mini" | "nano"


# ============================================
# 2. 별칭(alias) -> 실제 API 모델명 매핑
#    새 모델이 추가되면 여기에만 추가하면 됩니다.
# ============================================
MODEL_MAP = {
    "mini": "gpt-5-mini",
    "nano": "gpt-5-nano",
}


# ============================================
# 3. 모델별 기본 생성 옵션
#    필요하면 모델마다 다르게 설정 가능
#    (예: nano는 분류/추출용이라 짧고 정확하게,
#         mini는 최종 답변용이라 조금 더 여유 있게)
# ============================================
GENERATION_PARAMS = {
    "mini": {
        "temperature": 0.2,
        "max_tokens": 800,
    },
    "nano": {
        "temperature": 0.0,
        "max_tokens": 200,
    },
}


# ============================================
# 4. 최종적으로 실행 함수가 참조할 값
#    (이 아래는 수정할 필요 없음)
# ============================================
def get_active_model_name() -> str:
    """ACTIVE_MODEL 별칭을 실제 API 모델명으로 변환해서 반환"""
    if ACTIVE_MODEL not in MODEL_MAP:
        raise ValueError(
            f"알 수 없는 모델 별칭입니다: '{ACTIVE_MODEL}'. "
            f"MODEL_MAP에 정의된 값 중 하나를 사용하세요: {list(MODEL_MAP.keys())}"
        )
    return MODEL_MAP[ACTIVE_MODEL]


def get_active_params() -> dict:
    """ACTIVE_MODEL에 해당하는 생성 옵션(temperature, max_tokens 등)을 반환"""
    return GENERATION_PARAMS.get(ACTIVE_MODEL, {"temperature": 0.2, "max_tokens": 500})
