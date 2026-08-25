"""
실행 예시 파일

이 파일은 건드릴 필요 없이, config/model_config.py의 ACTIVE_MODEL 값만
"mini" <-> "nano"로 바꾸고 이 파일을 실행하면 자동으로 다른 모델이 호출됩니다.
"""

from dotenv import load_dotenv
load_dotenv()   # .env 파일을 읽어서 환경변수로 등록 (반드시 다른 import보다 먼저!)

from src.generation import generate_answer

if __name__ == "__main__":
    # 실제로는 Retrieval 단계에서 이 context가 자동으로 채워집니다.
    sample_context = "본 사업은 국민연금공단이 발주한 이러닝시스템 고도화 사업으로, 예산은 3억원이며 사업 기간은 6개월이다."
    sample_query = "이 사업의 예산이 얼마야?"

    answer = generate_answer(query=sample_query, context=sample_context)
    print("\n[답변]")
    print(answer)
    