# src — 파일 역할과 실행 방법

## generation 파트가 볼 것은 두 개뿐이다

```python
from src.retriever import retrieve_context      # 질문 → 발췌 문자열
from src.generation import generate_answer      # 질문 + 발췌 → 답변

context = retrieve_context("이 사업의 예산이 얼마야?")
result = generate_answer(model_key="mini", query="이 사업의 예산이 얼마야?", context=context)
```

프로젝트 루트에서 실행하면 된다. `retriever.py` 가 경로를 알아서 잡는다.

**단, `retrieve_context()` 는 TEI 서버(8085 · 8086)와 FAISS 인덱스가 있어야 돈다.**
그게 없는 상태로 생성 쪽만 확인하려면 검색 결과를 파일로 받는다 — 아래 "인계 파일".

| 헷갈리기 쉬운 것 | 답 |
|---|---|
| `pieces/search.py` 와 `retriever.py` | `search.py` 는 부품, `retriever.py` 는 그걸 조립해 놓은 창구. 밖에서는 `retriever.py` 만 쓴다 |
| `evaluation/generation.py` | 생성 **지표** (근거표시율·물러섬·충실성). 생성 로직이 아니다 |
| `pieces/` 안에 생성 부품이 없는 이유 | 답을 만드는 곳은 `src/generation.py` 하나다. 프롬프트가 두 벌이 되지 않게 `pieces/` 는 검색까지만 한다 |

## 인계 파일 — 검색 없이 생성만 돌릴 때

검색 담당이 아래를 돌려 `contexts_eval_qa.jsonl` 을 만들어 전달한다.
(`outputs/` 는 gitignore 대상이라 repo 에 없다. 파일로 받으면 된다.)

```bash
python src/retriever.py --export
```

한 줄이 곧 `generate_answer()` 한 번이다.

```python
import json

for row in map(json.loads, open("contexts_eval_qa.jsonl", encoding="utf-8")):
    result = generate_answer(model_key="mini", query=row["question"], context=row["context"])
```

```json
{
  "qid": "요구사항-001",
  "question": "「…벤처확인종합관리시스템…」의 SFR-001 요구사항 명칭은 무엇인가?",
  "type": "요구사항",
  "answerable": true,
  "doc_ids": ["20240330003-0"],
  "keywords": ["사용자 인증 기능 구현"],
  "context": "[1] 사업명 · 발주기관\n…",
  "sources": [{"n": 1, "doc_id": "…", "title": "…", "chunk_id": "…::0000"}],
  "chunks": 4, "chars": 4920
}
```

`answerable: false` 인 질문은 발췌에 답이 없는 것이 정답이다. 물러서야 맞다.
`keywords` 는 채점 참고용이라 프롬프트에 넣지 않는다.

## 팀 표준 구조와의 대응

팀 규칙은 파일 하나씩이지만, 여기서는 **부품이 여러 개인 단계만 디렉토리**로 두었다.

| 팀 규칙 | 여기 | 왜 |
|---|---|---|
| `preprocessing.py` | `preprocessing/` | 형식마다 추출기가 다르다 (hwp · hwp_table · pdf) + 후처리 (clean · toc) |
| `chunking.py` | `chunking.py` | 그대로 |
| `embedding.py` | `models/embed.py` | 리랭커·LLM 로더와 한 묶음 (`models/`) |
| `vectorstore.py` | `vectorstore.py` | 그대로 |
| `retriever.py` | `retriever.py` | 검색 부품은 `pieces/` 에 있고 이 파일이 조립한다 |
| `generation.py` | `generation.py` | 그대로 |
| `evaluation.py` | `evaluation/` | 검색 지표와 생성 지표를 분리 (담당이 다르다) |

## 파일

```
retriever.py         질문 → 발췌. 밖으로 나가는 창구 (generation · UI)
generation.py        발췌 → 답변. OpenAI / HuggingFace
chunking.py          documents.jsonl → 청크. section / recursive / semantic
vectorstore.py       FAISS 인덱스 만들기·불러오기
resources.py         메모리·디스크 감시 (VM 이 멈추는 걸 막는다)

preprocessing/       원본 hwp · pdf → 본문 텍스트
  hwp.py               olefile 로 OLE 를 직접 읽어 문단만 (pyhwp 는 안 씀)
  hwp_table.py         표 구조까지 복원. RFP 는 글자의 60~80%가 표 안에 있다
  pdf.py               pdfplumber. 표 영역은 본문에서 빼고 따로 붙인다
  clean.py             머리말·꼬리말·깨진 필드 제거
  toc.py               목차 제거
  run.py               위를 묶어 documents.jsonl 생성 + CSV 메타 병합

models/              모델 붙이기. 부품 쪽 코드는 안 바뀐다
  embed.py             TEI(8085) / local / fake
  rerank.py            TEI(8086) / local / fake
  llm.py               openai / vllm(8087) / hf / echo
  health.py            check_servers() — 뭐가 떠 있는지 한눈에

pieces/              retriever.py 가 쓰는 검색 부품. 갈아끼우며 A/B 하려고 나눠 뒀다
  base.py              Pipeline, State
  search.py            Dense · BM25 · Hybrid · FilterBy
  refine.py            Rerank · TopK · Widen

evaluation/
  evalset.py           질문 세트 만들기·저장 (→ data/eval_qa.json)
  retrieval.py         적중률 · MRR · compare        ← 검색 담당
  generation.py        근거표시율 · 물러섬 · 충실성   ← 생성 담당
```

## import 규칙

밖에서 부를 때는 `src.` 를 붙인다.

```python
from src.retriever import retrieve_context, search_notices
from src.generation import generate_answer
```

`src/` **안에서는** 평평하게 쓴다. 설정만 루트의 `config` 에서 가져온다.

```python
from config import settings
from preprocessing import load_documents
from models import load_embedder
from pieces import Pipeline, Dense, Rerank    # 검색 부품. 밖에서는 쓸 일 없다
```

## 실행 — 산출물을 만드는 건 src 안의 파일이다

각 단계가 자기 산출물을 직접 만든다.

```bash
python src/preprocessing/run.py                     → data/processed/documents.jsonl
python src/chunking.py                              → outputs/chunks/chunks_*.jsonl
python src/vectorstore.py --chunks <청크이름>        → outputs/vectorstore/<이름>/
python src/retriever.py "질문"                       → 무엇이 뽑히는지 눈으로 확인
python src/retriever.py --export                    → outputs/eval_results/contexts_*.jsonl
```

이름이 이어진다. `chunking.py` 가 다음에 칠 명령을 찍어 주므로 그대로 붙이면 된다.

```
python src/chunking.py --docs cleaned_documents --how recursive --size 1200
  → outputs/chunks/chunks_cleaned_documents__recursive_1200_200.jsonl
  → 다음:  python src/vectorstore.py --chunks cleaned_documents__recursive_1200_200
  → outputs/vectorstore/cleaned_documents__recursive_1200_200__tei/
```

**전처리본 · 자르기 설정 · 임베딩이 이름 하나에 다 남는다.** A/B 를 여러 벌
돌려도 어느 조합인지 파일 이름만 보면 안다.

`data/` 와 `outputs/` 는 gitignore 대상이다 (원본 RFP 가 NDA). clone 만으로는
비어 있으니 위 순서대로 한 번 돌리거나, 검색 담당에게 인계 파일을 받는다.

## 주석은 구글 스타일

```python
def load_store(name, embedder):
    """저장해 둔 인덱스를 불러온다.

    Args:
        name: `build_store` 에 준 것과 같은 이름.
        embedder: 만들 때와 **같은** 임베딩 객체.

    Returns:
        FAISS 인덱스.

    Raises:
        FileNotFoundError: 그 이름으로 저장된 인덱스가 없을 때.
    """
```

`chunking.py`, `vectorstore.py`, `retriever.py`, `preprocessing/run.py` 의 `main()`
은 옮겼다. 나머지 파일은 아직 서술형이다.
