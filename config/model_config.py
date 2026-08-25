"""모델별 설정만 모아두는 곳.

이 파일에 새 모델을 추가/변경하는 것만으로 generation.py 등 실행 코드는
전혀 건드릴 필요가 없어야 합니다 — 그게 이 파일을 따로 뺀 이유!!.

프로젝트 가이드 상 시나리오 A와 시나리오 B를
둘 다 구현해서 비교해야 하므로, provider 필드로 두 시나리오를 구분 합니다.
실행 코드는 provider만 보고 분기하고, 시나리오별로 파일을 따로 만들지 않습니다.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """모델 하나에 대한 실행 설정.

    Attributes:
        provider: "openai"(시나리오 B) 또는 "huggingface"(시나리오 A).
        model: 실제 모델 식별자. 예: "gpt-5-mini", "meta-llama/Meta-Llama-3-8B-Instruct".
        reasoning_effort: openai 전용 옵션. minimal | low | medium | high.
            사고 시간/품질 트레이드오프를 조절한다.
        verbosity: openai 전용 옵션. low | medium | high. 응답 길이/상세도를 조절한다.
    """

    provider: str
    model: str
    reasoning_effort: str = "medium"
    verbosity: str = "medium"


# 실제로 사용하는 모델 키는 여기 딕셔너리 키("mini", "nano" 등)로 통일!!
# 코드/로그/평가 결과 어디서든 같은 이름을 쓰기 (에러 방지).
MODEL_CONFIGS: dict[str, ModelConfig] = {
    # --- 시나리오 B ---
    "mini": ModelConfig(
        provider="openai",
        model="gpt-5-mini",
        reasoning_effort="medium",
        verbosity="medium",
    ),
    "nano": ModelConfig(
        provider="openai", model="gpt-5-nano", reasoning_effort="low", verbosity="low"
    ),
    # --- 시나리오 A: (나중에 추가 예정, 자리만 미리 잡아뒀어여) ---
}
