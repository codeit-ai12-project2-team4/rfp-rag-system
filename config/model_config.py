"""모델별 설정값을 정의하는 모듈.

새 모델을 추가할 때는 이 파일의 MODEL_CONFIGS에 항목을 추가하기만 하면 되고,
실행 코드(src/generation.py)는 건드릴 필요가 없습니다.
"""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """모델 하나에 대한 설정.

    Attributes:
        provider: "openai" 또는 "huggingface". generate_answer()가 이 값으로
            어떤 실행 함수를 쓸지 분기한다.
        model: 실제 호출할 모델명 (예: "gpt-5-mini", 또는 HuggingFace repo id).
        reasoning_effort: OpenAI 모델용. "minimal" | "low" | "medium" | "high".
            provider가 "huggingface"이면 사용되지 않는다.
        verbosity: OpenAI 모델용. "low" | "medium" | "high".
            provider가 "huggingface"이면 사용되지 않는다.
        dtype: HuggingFace 모델용. "bfloat16" | "float16" | "float32".
            provider가 "openai"이면 사용되지 않는다.
        device_map: HuggingFace 모델용 device_map 값 (예: "auto").
            provider가 "openai"이면 사용되지 않는다.
        max_new_tokens: HuggingFace 모델용 생성 토큰 상한.
            provider가 "openai"이면 사용되지 않는다.
        extra: HuggingFace pipeline() 호출 시 추가로 넘길 키워드 인자.
    """

    provider: str
    model: str
    reasoning_effort: str | None = None
    verbosity: str | None = None
    dtype: str | None = None
    device_map: str | None = None
    max_new_tokens: int | None = None
    extra: dict = field(default_factory=dict)


MODEL_CONFIGS = {
    "mini": ModelConfig(
        provider="openai",
        model="gpt-5-mini",
        reasoning_effort="medium",
        verbosity="medium",
    ),
    "nano": ModelConfig(
        provider="openai",
        model="gpt-5-nano",
        reasoning_effort="low",
        verbosity="low",
    ),
    # 시나리오 A 모델을 정하면 아래처럼 주석을 풀고 채우면 됩니댱.
    # 예시:
    # "llama-gcp": ModelConfig(
    #     provider="huggingface",
    #     model=" ",
    #     dtype="bfloat16",
    #     device_map="auto",
    #     max_new_tokens=512,
    # ),
}
