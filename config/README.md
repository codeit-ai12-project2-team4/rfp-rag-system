## config 디렉토리 사용

- settings.py
  - 내용: 프로젝트 전반의 고정 세팅값을 정의
  - 메서드 사용법
    - load_env(): `.env`파일의 내용을 훑어서 `os.environ`에 입력함. ※ os.environ 은 python 런타임이 읽어오는 환경 변수를 담고 있음
    - make_dirs(): 프로젝트 초기 세팅시 필요한 디렉토리를 만들어줌

- model_config.py
  - 내용: 모델별 설정(`MODEL_CONFIGS`)과 `ModelConfig` 자료형

---

## Provider — 모델을 어디서 부르는지

`config/settings.py` 에 있습니다. **`StrEnum` 이라 문자열과 그대로 비교됩니다.**
그래서 지금 코드를 안 고쳐도 됩니다.

```python
from config import Provider

Provider.OPENAI == "openai"          # True
Provider.HF     == "huggingface"     # True
f"{Provider.OPENAI}"                 # "openai"
{"openai": run}.get(Provider.OPENAI) # run    ← dict 키로도 그대로
```

문자열을 넣으면 대소문자와 앞뒤 공백을 봐줍니다.

```python
Provider("  OpenAI ")   # Provider.OPENAI
Provider("HF")          # Provider.HF
Provider("groq")        # ValueError: 'groq' is not a valid Provider
Provider.list_values()  # ['openai', 'huggingface']
```

### model_config.py 에서

타입만 바꾸면 됩니다. 값은 그대로 두세요.

```python
from .settings import Provider

@dataclass
class ModelConfig:
    provider: Provider      # 전에는 str
    model: str
    ...

MODEL_CONFIGS = {
    "mini": ModelConfig(
        provider=Provider.OPENAI,        # "openai" 라고 써도 똑같이 동작합니다
        model="gpt-5-mini",
        ...
    ),
}
```

이렇게 하면 `provider="openaii"` 같은 오타를 등록하는 순간 잡힙니다.

### generation.py 에서

**안 고쳐도 그대로 돕니다.** `_PROVIDER_RUNNERS.get(cfg.provider)` 가 Provider 를
받아도 문자열 키를 찾습니다. 굳이 맞추고 싶으면 키만 바꾸세요.

```python
_PROVIDER_RUNNERS = {
    Provider.OPENAI: _run_openai,
    Provider.HF: _run_huggingface,
}
```

에러 메시지에 선택지를 넣을 수 있습니다.

```python
error=f"지원하지 않는 provider 입니다: {cfg.provider} "
      f"(가능한 값: {Provider.list_values()})"
```

### 새 프로바이더를 넣을 때

`settings.py` 의 `Provider` 에 한 줄 추가하고, `generation.py` 에
`_run_XXX` 를 만들어 `_PROVIDER_RUNNERS` 에 등록하면 끝입니다.

**지금은 둘뿐입니다.** vLLM 은 OpenAI 규격과 호환이라 `Provider.OPENAI` 에
`base_url` 만 바꿔 붙입니다. Anthropic·Google·Groq 등은 키도 계획도 없어서
넣지 않았습니다 — 선언만 있고 동작 안 하는 값은 두지 않는 편이 낫습니다.
