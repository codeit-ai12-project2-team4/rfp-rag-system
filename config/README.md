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

---

## prompts/ — 프롬프트는 코드가 아니라 내용이다

```
config/prompts/system.md    답변 생성용 시스템 프롬프트
```

`src/generation.py` 의 `system_prompt()` 가 **호출할 때마다** 읽습니다.
파일만 고치면 서버 재시작 없이 다음 질문부터 적용됩니다.

```python
from generation import system_prompt
system_prompt()   # config/prompts/system.md 의 내용
```

### 쓸 때 지킬 것

- **코드에 프롬프트를 박지 마세요.** 박으면 문구 한 줄 고치는 데 배포가 필요합니다.
- 한 파일에 한 용도. 다른 프롬프트가 생기면 `config/prompts/<용도>.md` 를 새로
  만들고 그걸 읽는 함수를 하나 더 답니다.
- `{}` 를 쓰지 마세요. 지금은 `.format()` 을 안 쓰지만, 나중에 누가 붙이면
  RFP 본문의 중괄호에서 터집니다. 변수는 메시지로 넘깁니다.
- **고칠 때마다 재보세요.** 프롬프트 한 줄이 근거표시율을 0에서 100으로 바꿉니다.
  실제로 인용 지시가 없어서 `[1] [2]` 가 구조적으로 안 붙고 있었습니다.

### 현재 내용이 하는 일

| 줄 | 왜 있나 |
|---|---|
| 컨텍스트만 근거로 | 환각 억제. RAG 의 존재 이유다 |
| 없으면 "확인되지 않습니다" | 모른다고 말하게 한다. 평가 세트의 `answerable: false` 가 이걸 잰다 |
| 미사여구 없이 | 입찰 담당자는 요약을 원하지 인사말을 원하지 않는다 |
| `[1] [2]` 인용 | 근거표시율. 이게 없으면 출처 화면을 못 만든다 |

관리자 화면은 이 파일 하나로 부족해질 때 만듭니다. 그때도 저장 대상은
여전히 이 문자열입니다.
