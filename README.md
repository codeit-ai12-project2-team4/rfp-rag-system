# rfp-rag-system

RFP 문서 기반 RAG 상담 시스템 구축 프로젝트

## 구조

```
config/       설정과 모델 카탈로그      → config/README.md
src/          파이프라인               → src/README.md
eda/          문서 탐색·통계
main.py       모델 골라서 한 번 돌려보는 예시
```

`data/`, `outputs/`, `notebooks/`, `scripts/`, `docker/` 는 push 하지 않는다.
원본 RFP 가 NDA 대상이고, 나머지는 각자의 실험 공간이라 repo 를 불린다.
**clone 만으로는 데이터도 인덱스도 없다.** 아래 순서로 만들거나 담당자에게 받는다.

## 실행

```bash
python src/preprocessing/run.py                → data/processed/documents.jsonl
python src/chunking.py --docs cleaned_documents --how recursive --size 1200
python src/vectorstore.py --chunks cleaned_documents__recursive_1200_200
python src/retriever.py "이 사업의 예산이 얼마야?"
```

## retrieval → generation

```python
from src.retriever import retrieve_context
from src.generation import generate_answer

context = retrieve_context("이 사업의 예산이 얼마야?")
result = generate_answer(model_key="mini", query="이 사업의 예산이 얼마야?", context=context)
```

TEI 서버와 인덱스 없이 생성만 확인하려면 검색 결과를 파일로 받는다.
만드는 쪽은 `python src/retriever.py --export`, 쓰는 쪽은 jsonl 한 줄이 곧 호출 한 번이다.
형식과 주의점은 [src/README.md](src/README.md) 에 있다.

## 검색 설정 (2026-08-26 실측, 133문항)

`cleaned_documents → recursive/1200/200 → 목차 줄 제거 → 머리말 → Dense(k=30) → Rerank(k=8)`

| 질문 유형 | 적중률 |
|---|---|
| 요구사항 | 0.975 |
| 배점 | 0.950 |
| 의역 | 0.675 |

머리말과 리랭커가 승패를 가르고, BM25 를 섞으면 오히려 떨어진다.
근거는 `src/retriever.py` 맨 위 주석에 있다.

## 환경

python 3.11 이상 (`config.Provider` 가 `StrEnum` 을 쓴다).

```bash
pip install -r requirements.txt
cp .env.example .env      # 그 다음 키를 채운다
```
