# src — 파일 역할과 실행 방법

## 팀 표준 구조와의 대응

팀 규칙은 파일 하나씩이지만, 여기서는 **부품이 여러 개인 단계만 디렉토리**로 두었다.
역할은 그대로다.

| 팀 규칙 | 여기 | 왜 |
|---|---|---|
| `preprocessing.py` | `preprocessing/` | 형식마다 추출기가 다르다 (hwp · hwp_table · pdf) + 후처리 (clean · toc) |
| `chunking.py` | `chunking.py` | 그대로 |
| `embedding.py` | `models/embed.py` | 리랭커·LLM 로더와 한 묶음 (`models/`) |
| `vectorstore.py` | `vectorstore.py` | 그대로 |
| `retriever.py` | `pieces/search.py` | Dense · BM25 · Hybrid · FilterBy 를 갈아끼운다 |
| `generation.py` | `pieces/generate.py` | Generate · MakeCard |
| `evaluation.py` | `evaluation/` | 검색 지표와 생성 지표를 분리 (담당이 다르다) |

## 파일

```
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

pieces/              검색·생성 부품. nn.Sequential 처럼 끼우고 뺀다
  base.py              Pipeline, State
  search.py            Dense · BM25 · Hybrid · FilterBy
  expand.py            AddKeywords · QueryRewrite · MultiQuery
  refine.py            Rerank · Compress · TopK · Widen
  generate.py          Generate · MakeCard

evaluation/
  evalset.py           질문 세트 만들기·저장 (→ data/eval_qa.json)
  retrieval.py         적중률 · MRR · compare        ← 검색 담당
  generation.py        근거표시율 · 물러섬 · 충실성   ← 생성 담당
```

## import 규칙

`src/` 를 sys.path 에 넣고 **평평하게** 쓴다. 설정만 루트의 `config` 에서 가져온다.

```python
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from config import settings
import chunking
from preprocessing import load_documents
from models import load_embedder
from pieces import Pipeline, Dense, BM25, Hybrid, Rerank, Generate
```

`from src.preprocessing import ...` 는 안 된다. 패키지 내부가 평평한 import 를 쓴다.

## 실행 — 산출물을 만드는 건 src 안의 파일이다

각 단계가 자기 산출물을 직접 만든다. `scripts/` 는 실험·점검용이지 파이프라인이
아니다.

```bash
python src/preprocessing/run.py                     → data/processed/documents.jsonl
python src/chunking.py                              → outputs/chunks/chunks_*.jsonl
python src/vectorstore.py --chunks <청크이름>        → outputs/vectorstore/<이름>/
```

이름이 이어진다. `chunking.py` 가 다음에 칠 명령을 찍어 주므로 그대로 붙이면 된다.

```
python src/chunking.py --docs cleaned_documents --how recursive --size 1500
  → outputs/chunks/chunks_cleaned_documents__recursive_1500_200.jsonl
  → 다음:  python src/vectorstore.py --chunks cleaned_documents__recursive_1500_200
  → outputs/vectorstore/cleaned_documents__recursive_1500_200__tei/
```

**전처리본 · 자르기 설정 · 임베딩이 이름 하나에 다 남는다.** A/B 를 여러 벌
돌려도 어느 조합인지 파일 이름만 보면 안다.

### scripts/ — 실험과 점검

```bash
python scripts/check_setup.py       import · 데이터 · 서버 점검
python scripts/eval_tables.py       표 추출 점검 (--dump 로 눈으로 대조)
python scripts/compare_chunking.py  청킹 설정 A/B (--docs 로 전처리본까지)
```

서버는 `docker/` 에서 띄운다 (TEI 임베딩 8085 · 리랭커 8086).

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

`chunking.py`, `vectorstore.py`, `preprocessing/run.py` 의 `main()` 은 옮겼다.
나머지 파일은 아직 서술형이다.
