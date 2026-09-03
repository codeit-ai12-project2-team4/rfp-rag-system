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
---

## 채점만 로컬에서 (다른 팀원용)

**검색을 다시 돌릴 필요가 없다.** `contexts_*.jsonl` 한 줄이 곧
`generate_answer()` 한 번이라, 그 파일만 받으면 GPU도 TEI도 코퍼스도 필요 없다.
원본 RFP 는 NDA 라 어차피 못 넘긴다.

### 필요한 것
- 파이썬 3.11 이상
- `contexts_*.jsonl` (Joel 이 준다)
- 본인 `OPENAI_API_KEY`

### 준비

```bash
git clone -b feature/retrieval <저장소> rfp-rag-system
cd rfp-rag-system

python3 -m venv .venv && source .venv/bin/activate
# **전부 깔 필요 없다.** 채점 경로는 이 셋만 쓴다 (torch·faiss·lancedb 안 쓴다)
pip install openai requests langchain-core

printf 'OPENAI_API_KEY=sk-...\n' > .env      # .env 는 gitignore 다. 직접 만든다

mkdir -p outputs/eval_results
cp ~/받은/contexts_*.jsonl outputs/eval_results/
```

`uv sync` 로 전부 깔아도 되지만 torch 까지 받아서 3GB 쯤 된다. 채점만 할 거면 위가 낫다.

### 돌리기

```bash
CTX=outputs/eval_results/contexts_eval_qa_both.jsonl   # 받은 파일 이름으로

# 1) 연습 — 5문항만. 여기서 답이 이상하면 아래를 돌리지 말 것 (돈이 나간다)
python scripts/retrieval/answer.py $CTX --model mini --limit 5 \
    --out outputs/eval_results/answers_test.jsonl
python scripts/retrieval/score_answers.py outputs/eval_results/answers_test.jsonl

# 2) 본 실행. **중단해도 된다** — 이미 만든 문항은 건너뛰고 이어 쓴다
python scripts/retrieval/answer.py $CTX --model mini

# 3) 채점. --judge 를 빼면 LLM 없이 공짜 지표 넷만 나온다
python scripts/retrieval/score_answers.py \
    outputs/eval_results/answers_eval_qa_both.jsonl \
    --judge --model nano --json outputs/eval_results/metrics.json
```

`--out` 을 생략하면 `contexts_` 를 `answers_` 로 바꾼 이름으로 저장한다.

### 비용
문항당 대략 답변(mini) $0.005 + 채점(nano) $0.002.
191문항 한 바퀴에 $1.34 다. **팀 예산이 $20 이니 본 실행 전에 꼭 5문항으로 연습할 것.**
`--model nano` 로 답변까지 내리면 $0.76 이다.

### 지표 읽기

    인용표시율   답변에 [n] 을 달았나
    인용정확도   그 [n] 이 발췌 번호 범위 안인가
    숫자근거율   답변의 숫자가 발췌 안에 있는가
    충실성       발췌에 있는 내용만 말했나 (--judge 필요)
    물러섬       근거가 없을 때 모른다고 했나

**인용정확도가 0.000 이면 성능이 아니라 고장이다.** 발췌가 비었다는 뜻이다.
`contexts_*.jsonl` 의 `context` 필드가 빈 문자열인지 먼저 보라.

    python -c "import json;rows=[json.loads(l) for l in open('$CTX')];\
print('빈 발췌', sum(1 for r in rows if not r['context']), '/', len(rows))"

`Groundedness` 는 안 잰다 — 충실성과 사실상 같은 것을 묻는데 LLM 호출만 두 배다.

---

## 검색부터 전부 돌리려면

브랜치만으로는 안 된다. 아래가 더 필요하고, 앞의 둘은 NDA 라 직접 받아야 한다.

    data/raw/                    원본 RFP (커밋 안 됨)
    data/metadata/data_list.csv  공고 메타데이터 (커밋 안 됨)
    TEI 도커 2개                 임베더 :8085 · 리랭커 :8086
    .env                         DOCS/CHUNKS/EMBED/STORE… (docker/README.md 참고)

```bash
uv sync                                              # torch 포함, 3GB 쯤
docker compose -f docker/docker-compose.yml up -d embed rerank
uv run python scripts/retrieval/check_setup.py       # 뭐가 빠졌는지 먼저 본다
uv run python scripts/retrieval/prepare.py --build   # 전처리 → 청크 → 색인
uv run python src/retriever.py --export --evalset eval_qa_both --generation \
    --out outputs/eval_results/contexts_eval_qa_both.jsonl
```

첫 실행은 BM25 색인을 짓느라 2분 30초가 더 걸린다.
