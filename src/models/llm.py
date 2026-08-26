"""답변 생성 — LLM.

    llm = load_llm("openai")   # OpenAI API (기본값)
    llm = load_llm("vllm")     # 도커로 띄운 vLLM
    llm = load_llm("hf")       # 노트북 안에 모델을 올림 (강의 방식)
    llm = load_llm("echo")     # 가짜. 프롬프트에 뭐가 들어갔는지만 볼 때

어느 쪽이든 ask(system, user) 하나만 있으면 된다. 강의의 ask() 와 같은 자리다.
"""

import os

from resources import free_disk_gb

VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8087/v1")
LOCAL_LLM_MODEL = os.environ.get(
    "LOCAL_LLM_MODEL", "Qwen/Qwen3-4B-Instruct-2507"
)  # "Qwen/Qwen2.5-7B-Instruct")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini")


# 모델별 대략 내려받는 용량 (fp16 기준). 디스크가 부족한 VM 이 많아서 미리 경고한다.
MODEL_SIZE_GB = {
    "Qwen/Qwen2.5-7B-Instruct": 15,
    "Qwen/Qwen2.5-3B-Instruct": 6,
    "Qwen/Qwen2.5-1.5B-Instruct": 3,
    "Qwen/Qwen3-4B-Instruct-2507": 8,
    "unsloth/Llama-3.2-3B-Instruct": 6,
    "K-intelligence/Midm-2.0-Mini-Instruct": 5,
}


class HFLLM:
    """노트북 안에 모델을 올려서 쓴다. 강의에서 하던 방식 그대로.

    **디스크를 많이 먹는다.** 7B 를 fp16 으로 받으면 15GB 다. 거기에 torch 와
    CUDA 라이브러리가 6~8GB. VM 디스크가 50GB 면 이것만으로 절반이 찬다.

    디스크가 빠듯하면 순서대로 검토할 것:
      1. load_llm("openai")     디스크 0. 팀 한도 $20 안에서
      2. 작은 모델              3B 는 6GB, 1.5B 는 3GB
      3. AWQ 4bit               7B 가 약 5GB 로 줄어든다
      4. 디스크를 늘린다        과제 지침상 200GB 까지 허용된다

    L4 24GB 면 7~9B 를 float16 으로 **올릴 수는** 있다. GPU 메모리와 디스크는
    다른 문제라는 걸 헷갈리지 말 것 — 여기서 막히는 건 대개 디스크다.
    """

    name = "hf"

    def __init__(self, model=None, device="cuda", dtype=None, skip_disk_check=False):
        model_name = model or LOCAL_LLM_MODEL

        # 무거운 import 전에 디스크부터 본다. 다 받고 나서 터지면 시간만 버린다.
        need = MODEL_SIZE_GB.get(model_name)
        free = free_disk_gb()
        if not skip_disk_check and need and free < need * 1.3:
            raise RuntimeError(
                f"디스크가 부족합니다.\n"
                f"  {model_name} 은 약 {need}GB 를 받습니다 (여유 공간은 그 1.3배는 있어야 함)\n"
                f"  지금 남은 공간: {free:.1f}GB\n\n"
                f"선택지:\n"
                f"  1. load_llm('openai')                 디스크 0\n"
                f"  2. load_llm('hf', model='Qwen/Qwen2.5-1.5B-Instruct')   약 3GB\n"
                f"  3. 캐시 정리: rm -rf ~/.cache/huggingface/hub/models--*\n"
                f"  4. 디스크 늘리기 (과제 지침상 200GB 까지 허용)\n\n"
                f"그래도 강행하려면 skip_disk_check=True"
            )
        if need:
            print(f"내려받기 시작: {model_name} (약 {need}GB, 남은 공간 {free:.1f}GB)")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        if torch.mps.is_available():
            device = "mps"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype or torch.float16, quantization_config=bnb_config
        ).to(device)
        self.device = device
        print(f"모델 올림: {model_name} ({device})")

    def ask(self, system, user, max_tokens=800):
        text = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        output = self.model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=False
        )
        generated = output[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


class OpenAILLM:
    """OpenAI API 또는 그와 호환되는 서버(vLLM, TGI)에 물어본다.

    vLLM 을 도커로 띄우면 OpenAI 와 똑같은 형식으로 말을 받으므로
    base_url 만 바꾸면 같은 코드로 쓸 수 있다.
    """

    def __init__(
        self, model=None, base_url=None, api_key=None, temperature=0.0, name="openai"
    ):
        from openai import OpenAI

        self.name = name
        self.model = model or OPENAI_MODEL
        self.temperature = temperature
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or os.environ.get("OPENAI_API_KEY") or "unused",
        )

    def ask(self, system, user, max_tokens=800):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            max_completion_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()


class EchoLLM:
    """가짜 LLM. 프롬프트에 뭐가 들어갔는지만 보고 싶을 때.

    돈도 GPU 도 안 든다. 부품을 새로 만들고 배관을 확인할 때 쓴다.
    """

    name = "echo"

    def __init__(self, reply=None):
        self.reply = reply
        self.calls = []

    def ask(self, system, user, max_tokens=800):
        self.calls.append({"system": system, "user": user})
        if self.reply is not None:
            return self.reply
        return "\n".join([
            "(가짜 LLM 입니다. 실제 답변이 아닙니다.)",
            f"발췌 {user.count('---') + 1}덩어리를 받았습니다.",
            f"프롬프트 길이 {len(system) + len(user)}자",
        ])


def load_llm(kind="openai", model=None, **kwargs):
    """답변 생성 모델을 붙인다.

    | 종류 | 무엇 | 디스크 | 돈 |
    |---|---|---|---|
    | `openai` | OpenAI API | 0 | 팀 한도 $20 |
    | `echo`   | 가짜. 프롬프트 확인용 | 0 | 0 |
    | `vllm`   | 도커로 띄운 vLLM | 이미지 10GB + 모델 | 0 |
    | `hf`     | 노트북 안에 transformers 로 (강의 방식) | 모델 크기만큼 | 0 |

    **기본값이 openai 인 이유** — `hf` 는 7B fp16 이면 15GB 를 내려받는다.
    50GB VM 에서 그냥 부르면 디스크가 찬다. 로컬 모델은 필요할 때
    명시적으로 고르게 했다.
    """
    if kind == "hf":
        return HFLLM(model=model, **kwargs)
    if kind == "vllm":
        return OpenAILLM(
            model=model or LOCAL_LLM_MODEL,
            base_url=kwargs.pop("base_url", VLLM_URL),
            api_key="local",
            name="vllm",
            **kwargs,
        )
    if kind == "openai":
        return OpenAILLM(model=model or OPENAI_MODEL, name="openai", **kwargs)
    if kind == "echo":
        return EchoLLM(**kwargs)
    raise ValueError(f"모르는 LLM 종류: {kind!r} (hf / vllm / openai / echo 중 하나)")
