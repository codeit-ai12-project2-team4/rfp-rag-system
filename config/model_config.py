"""모델별 설정값을 정의하는 모듈.

새 모델을 추가할 때는 이 파일의 MODEL_CONFIGS에 항목을 추가하기만 하면 되고,
실행 코드(src/generation.py)는 건드릴 필요가 없습니다.

provider 는 셋이다.

    openai      OpenAI API. GPU 를 안 쓴다. 팀 한도 $20.
    sglang      VM 의 SGLang 컨테이너. **한 번에 한 모델만** 올라간다.
    huggingface transformers 로 프로세스 안에 직접. 지금은 안 쓰지만 평가
                스크립트가 부를 수 있어 남겨 둔다.

**mem 값은 실측해서 조정하는 값이다.** L4 24GB 에 TEI 셋이 약 4GB 상주하므로
생성용은 20GB(=0.83)뿐이다. OOM 이 나면 내리고, KV 캐시가 모자라 느리면 올린다.
"""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """모델 하나에 대한 설정.

    Attributes:
        provider: "openai" | "sglang" | "huggingface". generate_answer()가 이 값으로
            어떤 실행 함수를 쓸지 분기한다.
        model: 실제 호출할 모델명 (예: "gpt-5-mini", 또는 HuggingFace repo id).
        reasoning_effort: OpenAI 모델용. "minimal" | "low" | "medium" | "high".
        verbosity: OpenAI 모델용. "low" | "medium" | "high".
        dtype: HuggingFace 모델용. "bfloat16" | "float16" | "float32".
        device_map: HuggingFace 모델용 device_map 값 (예: "auto").
        max_new_tokens: 생성 토큰 상한. sglang / huggingface 에서 쓴다.
        mem: sglang 용 `--mem-fraction-static`. GPU **전체** 대비 비율.
        args: sglang 서버에 그 모델에만 붙일 추가 인자 문자열.
        extra: HuggingFace pipeline() 호출 시 추가로 넘길 키워드 인자.
    """

    provider: str
    model: str
    reasoning_effort: str | None = None
    verbosity: str | None = None
    dtype: str | None = None
    device_map: str | None = None
    max_new_tokens: int | None = None
    mem: str | None = None
    args: str = ""
    extra: dict = field(default_factory=dict)


MODEL_CONFIGS = {
    # --- 외부 API. GPU 를 안 쓰므로 교체 대기가 없다 ---------------------
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
    # --- VM 안. 고르면 그 모델로 컨테이너가 갈아끼워진다 ------------------
    # 가중치 크기(fp16)와 mem 값. 남는 20GB 안에서 KV 캐시까지 잡아야 한다.
    # kakaocorp/kanana-nano-2.1b-instruct 는 뺐다 (2026-09-02).
    # SGLang 이미지의 transformers 가 LlamaConfig 를 검증할 때
    #     ValueError: The hidden size (1792) is not a multiple of
    #                 the number of attention heads (24)
    # 로 launch 단계에서 죽는다. 이 모델은 head_dim=128 을 config 에 명시해서
    # hidden_size / num_heads 와 일부러 다르게 잡은 건데(1792 vs 24*128=3072),
    # 검증기가 head_dim 을 안 본다. 우리 설정 문제가 아니라 이미지 쪽 문제다.
    # **되살리려면 이미지를 바꿔서 실제로 떠야 확인된다.** 목록에만 넣으면
    # 사용자가 고르고 15분을 기다린 뒤 에러를 본다.
    "exaone": ModelConfig(  # 2.4B / 약 4.8GB
        provider="sglang",
        model="LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
        max_new_tokens=512,
        mem="0.38",
    ),
    "qwen": ModelConfig(  # 3.09B / 약 6.2GB
        provider="sglang",
        model="Qwen/Qwen2.5-3B-Instruct",
        max_new_tokens=512,
        mem="0.45",
    ),
    "kanana8b": ModelConfig(  # 8B / 약 16GB. 혼자만 올라간다
        provider="sglang",
        model="kakaocorp/kanana-1.5-8b-instruct-2505",
        max_new_tokens=512,
        mem="0.78",
    ),
    "luxia8b": ModelConfig(  # 8B / 약 16GB
        # 주의: Luxia 는 **베이스 모델**이라 채팅 템플릿이 없다. Llama-3 템플릿을
        # 씌워서 쓴다. 답변이 이상하면 이 모델부터 의심할 것.
        provider="sglang",
        model="saltlux/Ko-Llama3-Luxia-8B",
        max_new_tokens=512,
        mem="0.78",
        args="--chat-template llama-3-instruct",
    ),
}
