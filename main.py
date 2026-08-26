"""
실행 예시 파일

python main.py 로 실행하면, config/model_config.py에 등록된 모델 목록을
보여주고 사용자가 직접 번호로 선택할 수 있습니다.
"""

from config import MODEL_CONFIGS
from src.generation import generate_answer


def choose_model() -> str:
    """등록된 모델 목록을 보여주고 사용자에게 번호로 하나를 선택하게 한다.

    잘못된 입력이면 첫 번째 모델을 기본값으로 사용한다.

    Returns:
        선택된 model_key.
    """
    keys = list(MODEL_CONFIGS.keys())
    print("=" * 50)
    print("사용할 모델을 선택하세요")
    print("=" * 50)
    for i, key in enumerate(keys, start=1):
        cfg = MODEL_CONFIGS[key]
        print(f"  [{i}] {key}  (provider={cfg.provider}, model={cfg.model})")
    print("=" * 50)

    choice = input(f"번호 입력 (1-{len(keys)}): ").strip()
    try:
        index = int(choice) - 1
        if index < 0:
            raise ValueError
        return keys[index]
    except (ValueError, IndexError):
        default_key = keys[0]
        print(f"잘못된 입력이라 기본값 '{default_key}'을(를) 사용합니다.")
        return default_key


if __name__ == "__main__":
    model_key = choose_model()

    sample_context = (
        "본 사업은 국민연금공단이 발주한 이러닝시스템 고도화 사업으로, "
        "예산은 3억원이며 사업 기간은 6개월이다."
    )
    sample_query = "이 사업의 예산이 얼마야?"

    result = generate_answer(
        model_key=model_key,
        query=sample_query,
        context=sample_context,
    )

    if not result["ok"]:
        print("\n[오류]")
        print(result["error"])
    else:
        print("\n[답변]")
        print(result["answer"])
        if result["usage"]:
            print(
                f"\n(토큰 사용량: {result['usage']}, 소요 시간: {result['latency_sec']:.2f}초)"
            )
