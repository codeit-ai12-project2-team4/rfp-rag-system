---

### 1. `generator.py` (Golden Q&A 생성기)

전처리된 RFP 본문(JSONL)을 청크 단위로 나누고, 경량 LLM(Qwen2.5-3B-Instruct)을 활용해 질문-정답-근거(Evidence) 세트를 대량 추출합니다.

* **주요 기능:**
  * **JSON 포맷 제어:** `question`, `answer`, `evidence_text` 구조의 정형 JSON 자동 추출
  * **청크 분할 처리:** 긴 RFP 문서에서 상위 N개 핵심 청크를 추출해 균일한 Q&A 세트 생성
* **핵심 클래스/함수:**
  * `QAEvalGenerator`: LLM 로드 및 Q&A 배치 생성 클래스
  * `load_documents(path)`: JSONL 문서 로더
  * `save_jsonl(items, path)`: JSONL 데이터 저장 유틸리티

---

### 2. `sampler.py` (품질 검사 및 벤치마크 샘플러)

대량 생성된 Q&A 풀에서 LLM 환각 및 생성 결함을 자동으로 걸러내고, 도메인 특화 4대 유형별로 균형 잡힌 벤치마크 셋을 추출합니다. (LLM 로드 불필요)

* **주요 기능:**
  * **4대 결함 자동 필터링 (`inspect_quality`):**
    1. 동일 8자 이상 반복 구절 (생성 루프 폭주 감지)
    2. 지시어/참고만 적힌 빈 근거 본문
    3. 근거 원문과 정답 키워드 불일치 (명백한 환각 차단)
    4. 전처리 특수문자 반복 등 파싱 결함
  * **RFP 4대 평가 유형 층화 추출 (총 80문항):**
    * **배점 (15문항):** 정량/정성 평가표, 배점 한도, 점수 수치 추출 평가
    * **요구사항 (15문항):** SFR 등 고유 식별자 및 기능 명칭 매칭 평가
    * **의역 (40문항):** 사용자 구어체 질의(예: "돈이 얼마나 드나?") 검색 민감도 평가
    * **없음 (10문항):** 문서에 없는 내용에 대한 환각 방어 및 답변 거부 평가
* **핵심 함수:**
  * `sample_benchmark_dataset()`: 결함 제거 및 80문항 벤치마크셋 추출 (`eval_set_80.jsonl`, `defect_items.jsonl` 반환)
  * `sample_balanced_items()`: 문서별 단순 균등 분배 샘플링 (100문항/30문항 빠른 분할용)
  * `inspect_quality()`: 개별 Q&A 아이템 무결성 검증

---

### 📊 평가셋 활용 가이드

* **`eval_set_100.jsonl` / `eval_set_80.jsonl` (평상시):** 청킹 방식, 임베딩 모델, Hybrid 검색 및 Re-ranker 최적화를 위한 검색(Retrieval) 벤치마크용.
* **`eval_set_30.jsonl` (마무리 단계):** LLM-as-a-judge(Ragas)를 활용한 최종 답변 생성 품질 및 Faithfulness 채점용.