"""
실행 파일

python main.py 로 실행하면, config/model_config.py에 등록된 모델 목록을
보여주고 사용자가 직접 번호로 선택할 수 있습니다.
"""

from config import MODEL_CONFIGS
from src.pipeline import GenerationPipeline


def choose_model() -> str:
    """등록된 모델 목록을 보여주고 사용자에게 번호로 하나를 선택하게 합니다.

    잘못된 입력이면 첫 번째 모델을 기본값으로 사용합니다.

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

    sample_query = "이 사업의 예산이 얼마야?"

    pipeline = GenerationPipeline(model_key=model_key)
    pipeline_result = pipeline(sample_query)
    result = pipeline_result.result

    print("\n[검색된 컨텍스트]")
    preview = pipeline_result.context[:300]
    print(preview + ("..." if len(pipeline_result.context) > 300 else ""))

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
