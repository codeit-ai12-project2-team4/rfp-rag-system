# src — 파일 역할과 실행 방법

## generation 파트가 볼 것은 두 개뿐이다

```python
from src.retriever import retrieve_context  # 질문 → 발췌 문자열
from src.generation import generate_answer  # 질문 + 발췌 → 답변

context = retrieve_context("이 사업의 예산이 얼마야?")
result = generate_answer(
    model_key="mini", query="이 사업의 예산이 얼마야?", context=context
)
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
    result = generate_answer(
        model_key="mini", query=row["question"], context=row["context"]
    )
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
lance_store.py       FAISS 대신 LanceDB. vectorstore.py 와 같은 자리, STORE=lance 환경변수로 전환
resources.py         메모리·디스크 감시 (VM 이 멈추는 걸 막는다)
crawl.py             나라장터 입찰공고 수집 → data/metadata/data_list.csv, data/raw/
evalrun.py           E2E 평가를 백그라운드로 돌리고 진행 상황을 outputs/eval_runs/<작업번호>.json 에 기록
api.py               웹 UI 가 부르는 HTTP 경계. 자세한 내용은 아래 "api.py" 절 참고

preprocessing/       원본 hwp · pdf → 본문 텍스트 (rfp/ 서브패키지 등 세부 구성은 아래 "preprocessing/ 상세" 절 참고)
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
  sglang.py            SGLang 생성 서버 컨테이너 교체 (ensure() — 한 번에 한 모델만)
  health.py            check_servers() — 뭐가 떠 있는지 한눈에

pieces/              retriever.py 가 쓰는 검색 부품. 갈아끼우며 A/B 하려고 나눠 뒀다
  base.py              Pipeline, State
  search.py            Dense · BM25 · Hybrid · FilterBy
  refine.py            Rerank · TopK · Widen
  expand.py            검색 전 질문 다듬기 — QueryRewrite · MultiQuery

evaluation/
  evalset.py           질문 세트 만들기·저장 (→ data/eval_qa.json)
  retrieval.py         적중률 · MRR · compare        ← 검색 담당
  generation.py        근거표시율 · 물러섬 · 충실성   ← 생성 담당
```

## preprocessing/ 상세 — 두 갈래가 같이 산다

이 폴더에는 **성격이 다른 두 묶음**이 있다. 이름이 비슷해서 헷갈리기 쉬운데,
하나도 죽지 않았다.

```
preprocessing/
├── clean.py  hwp.py  hwp_table.py  pdf.py  toc.py  run.py  fields.py   ← 검색 파트
├── pipeline.py                                              ← 전처리팀 원본. 현재 비활성
└── rfp/                                                     ← pipeline.py 를 나눈 것
    ├── common.py  hwp.py  pdf.py  extract.py  hwpx.py
    ├── clean.py   meta.py chunk.py build.py
    └── __init__.py
```

| 묶음 | 쓰는 곳 | 역할 |
|---|---|---|
| 바깥 `hwp/pdf/clean/toc/run` | `chunking.py` → `retriever.py` → API | 전처리본을 **읽어** Document 로 만든다 |
| `rfp/` | `prepare.py`, `ingest.py` | 원본 hwp/pdf 에서 전처리본·청크를 **만든다** |
| `pipeline.py` | 없음 | 위의 원본. 아래 참고 |

**바깥 것들이 죽었다고 오해하지 말 것.** `chunking._row_to_document()` 가
`preprocessing.run` 의 `from_langchain` / `tidy_doc_id` 를 부르고, 그게 doc_id
규칙을 만든다. 검색·평가가 전부 여기에 달려 있다.

---

### 왜 나눴나 (2026-09-03)

전처리팀에서 `pipeline.py` 작업 권한을 넘겨받았다. 한 파일에 **2,430줄**이었다.

나누기 전에 세 가지가 막혀 있었다.

#### 1. import 하면 죽었다

파일 끝, 모듈 최상위(함수 밖)에 전처리팀의 작업용 코드가 남아 있었다.

```python
df = pd.read_csv("")                                 # ← import 즉시 예외
DOC_PATH = Path(r"C:\Users\asd\Desktop\...")         # 남의 윈도우 경로
DOC_PATH.parent.mkdir(parents=True, exist_ok=True)   # 폴더까지 만든다
write_jsonl(df, DOC_PATH, ...)
```

`from preprocessing.pipeline import run_pipeline` 자체가 불가능했다.
자동화에 붙이려면 이게 첫 관문이었다.

#### 2. 비트 OR

```python
chunk_size: int = int(retrieval_settings.SIZE) | 1500      # or 가 아니다
```

`1500 | 1500` 은 우연히 1500 이라 지금은 맞는다. 하지만 `SIZE=1200` 으로
바꾸면 `1200 | 1500 = 1532` 가 **오류 없이** 들어간다. config 에 이미 기본값이
있으므로 폴백을 지우고 상수만 쓴다. - 이건 파이썬 문법 오해로 생긴 해프닝

#### 3. 직접 실행하면 청크가 안 나왔다

`__main__` 이 `enable_chunk_output` 을 안 켜서 전처리본만 나왔다.
청크가 없으면 색인을 못 만든다.

---

### 어떻게 나눴나

**파일 안에 이미 있던 섹션 표지(`# 0.` ~ `# 18.`)를 그대로 경계로 썼다.**
전처리팀이 나눠 둔 선이라 새로 판단할 게 없었다.

| 새 파일 | 원본 섹션 | 줄 수 |
|---|---|---|
| `common.py` | 0–6 설정·예외·결과 타입·품질검증 | 316 |
| `hwp.py` | 7–10 HWP 추출, OLE 저수준, 표 구조 복원 | 715 |
| `pdf.py` | 11 PDF 본문/표 분리 | 141 |
| `extract.py` | 12, 15 확장자 디스패처 + 문서 1건 처리 | 99 |
| `clean.py` | 13 텍스트 정제 | 388 |
| `meta.py` | 14, 16, 17 메타 추출·CSV 병합·스키마 변환 | 369 |
| `chunk.py` | 18 중 청킹 | 260 |
| `build.py` | 18 중 `run_pipeline` · `write_jsonl` · 리포트 | 278 |

**본문은 한 줄도 안 고쳤다.** 옮기고, 모듈 사이 import 를 붙이고,
위 세 가지만 손봤다.

의존 방향은 한쪽으로만 흐른다(순환 없음, 확인함).

```
build → chunk → clean → common
  ├──→ extract → hwp → common
  │              pdf → hwp
  └──→ meta ──────────→ common
```

`_CHUNK_SEPARATORS` 하나만 자리를 옮겼다. 원래 `run_pipeline` 쪽에 있어서
`build ↔ chunk` 순환이 났는데, 청킹 상수이므로 `chunk.py` 로 보냈다.

#### 왜 `rfp/` 하위 패키지인가

`hwp.py` · `pdf.py` · `clean.py` 가 바깥에 **이미 있다.** 같은 이름으로 풀어놓으면
덮어쓰거나, 안 덮어도 어느 쪽을 import 했는지 매번 헷갈린다.
하위 패키지로 두면 `preprocessing.rfp.hwp` 와 `preprocessing.hwp` 가 눈으로
구분된다.

#### 진입점

```python
from preprocessing.rfp import run_pipeline
```

`pipeline.py` 를 직접 import 하는 곳은 이제 없다.
`prepare.py` 와 `ingest.py` 를 `preprocessing.rfp` 로 돌렸다.

---

### `pipeline.py` 는 왜 남겨 뒀나

**지웠다가 되돌린 게 아니라, 원본을 그대로 둔 것이다.** `git` 상태와 같다.

- 전처리팀이 아직 이 파일을 기준으로 이야기한다. 대조할 원본이 필요하다
- 나눈 결과가 원본과 같은지 의심될 때 여기와 비교한다
- 파일 끝 스크래치 코드 때문에 **import 하면 안 된다.** 읽기 전용으로만 본다

전처리팀과 합의가 끝나면 지운다. 그때까지는 이 절이 "왜 두 벌인가" 의 답이다.

---

### `rfp/hwpx.py` 는 원본에 없던 파일이다 (2026-09-03)

크롤러가 hwpx 를 받아오는데 `SUPPORTED_EXTENSIONS = {".hwp", ".pdf"}` 라서
**조용히 버려지고 있었다.** 파일은 `data/raw` 에 쌓이고 CSV 에 행도 생기는데
문서가 안 된다 — 오류도 안 난다.

hwpx 는 차세대 나라장터가 내려주는 형식이라 앞으로 비중이 는다.
**hwp 와 완전히 다른 파일이다** — hwp 는 OLE 복합문서(바이너리),
hwpx 는 OWPML(zip 안에 XML). `hwp.py` 의 저수준 파서가 한 줄도 안 통한다.

- 표 렌더러(`_is_keyvalue_table` · `_render_keyvalue` · `_render_matrix`)는
  `hwp.py` 것을 그대로 쓴다. 그래야 두 형식이 같은 모양의 본문을 내놓고
  `clean.py` 와 청킹이 어느 쪽인지 몰라도 된다
- XML 이라 셀 주소를 복원할 필요가 없어서 **표가 오히려 hwp 보다 잘 나온다**
- `python src/preprocessing/rfp/hwpx.py` 로 자체 검사

자체 검사가 실제로 버그를 하나 잡았다: `iter()` 가 표 안까지 훑어서
**셀 내용이 표로 한 번, 본문으로 또 한 번** 나왔다. 청크가 두 배로 부풀고
검색이 같은 말을 두 번 센다. 표 안 `<hp:t>` 를 미리 표시해 걸러낸다.

**아직 실제 hwpx 로 표유실률을 안 쟀다.** 병합 셀(colSpan/rowSpan)을 안 펴서
병합이 많은 표는 열이 밀릴 수 있다. 새 공고를 받은 뒤
`scripts/retrieval/eval_tables.py` 로 재고 필요하면 편다.

`crawl.py` 의 `DOC_EXTS` 도 여기 맞춰 좁혔다 — doc/docx 는 파이프라인이 못
읽으니 받아봐야 쌓이기만 한다. **두 목록은 항상 같이 고친다.**

### 안 쓰게 된 것

| | 상태 |
|---|---|
| `pipeline.py` | 참조용으로만. import 하는 코드 없음 |
| `rfp_preprocessing_pipeline.py` | **파일이 이미 없다.** `ingest.py` 가 이걸 import 하고 있어서 같이 고쳤다 |

바깥 `hwp/hwp_table/pdf/clean/toc/run` 은 **전부 살아 있다.** 지우면 API 가 죽는다.

---

### 필드 이름표는 `fields.py` 하나다 (2026-09-03)

같은 아홉 개 필드를 설명하는 표가 네 군데 있었고 이미 갈라져 있었다.

| 있던 곳 | 개수 | 상태 |
|---|---|---|
| `run.py COLUMNS` | 11 | CSV(공백 있는 이름) → 영문 |
| `run.py _LANGCHAIN_META` | 9 | **`공고차수`·`입찰참여시작일` 누락** |
| `rfp/common.ORIGINAL_METADATA_COLUMNS` | 12 | **`공개기관` 은 CSV 에 없는 유령** |
| `rfp/meta.build_doc_schema_record` | 10 | 하드코딩. 아무도 안 부른다 |

`입찰참여시작일`(bid_open_at) 은 CSV 에도 있고 청크 메타에도 남는데
`from_langchain` 이 안 옮겨서 **검색단에서만 존재하지 않는 필드**였다.
`공개기관` 은 전처리를 돌릴 때마다 "없는 컬럼(건너뜀)" 을 찍고 있었다.

네 곳 모두 `preprocessing/fields.py` 의 `FIELDS` 에서 파생한다.
확인은 `python scripts/retrieval/check_fields.py` — 표가 하나인지, 열한 개가
전처리 → 청크 → 검색단으로 다 건너오는지 본다.

### 추출 결과 행의 키는 영문이다

`extract.process_document` 가 내놓는 행의 키는 `extractor`, `clean_text`,
`table_parse_success` … 전부 영문인데 파일명만 `파일명` 이었다. `filename` 으로
맞췄다. **CSV 쪽 컬럼(`meta_df`)은 한글 그대로다** — 그건 CSV 파일의 이름이고,
병합은 `_병합키` 로 붙으므로 양쪽 이름이 달라도 상관없다.

같이 고친 것: `build.py` 의 예외 분기가 `filename` 을 쓰고 정상 분기가 `파일명`
이었다. 추출이 실패한 문서만 병합 키가 NaN 이 되고 `source` 까지 사라져,
**메타데이터 없는 문서로 조용히 둔갑**했다.

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
from pieces import Pipeline, Dense, Rerank  # 검색 부품. 밖에서는 쓸 일 없다
```

## 실행 — 산출물을 만드는 건 src 안의 파일이다

각 단계가 자기 산출물을 직접 만든다.

```bash
python src/preprocessing/run.py                     → data/processed/documents.jsonl
python src/chunking.py                              → outputs/chunks/chunks_*.jsonl (평범한 것 + 머리말 붙은 것)
python src/vectorstore.py --chunks <청크이름>        → outputs/vectorstore/<이름>/
python src/retriever.py "질문"                       → 무엇이 뽑히는지 눈으로 확인
python src/retriever.py --export                    → outputs/eval_results/contexts_*.jsonl
```

이름이 이어진다. `chunking.py` 가 다음에 칠 명령을 찍어 주므로 그대로 붙이면 된다.

```
python src/chunking.py --docs cleaned_documents --how recursive --size 1200
  → chunks_cleaned_documents__recursive_1200_200.jsonl            BM25 가 쓴다
  → chunks_cleaned_documents__recursive_1200_200__header.jsonl    임베딩 인덱스가 쓴다
  → 다음:  python src/vectorstore.py --chunks cleaned_documents__recursive_1200_200
          python src/vectorstore.py --chunks cleaned_documents__recursive_1200_200__header
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

## api.py — 웹 UI 가 부르는 HTTP 경계

프론트(Next.js)는 별도 repo 입니다. 파이썬 소스를 볼 이유가 없고 배포 주기도
다릅니다. 둘을 잇는 건 이 파일뿐이라 **여기만 안 바뀌면 양쪽이 따로 움직입니다.**

```bash
uvicorn src.api:app --reload --port 8088
open http://localhost:8088/docs      # 스펙 원본. 따로 문서 쓰지 않는다
```

| 엔드포인트 | 무엇 |
|---|---|
| `POST /search` | 자연어로 공고 찾기 (1단계). 리랭커 안 씀 |
| `POST /ask` | 고른 공고 안에서 질문 (2단계). 답변 + `sources` |
| `GET /models` | 드롭다운 채우기 |

`/ask` 는 답변과 **출처를 같이** 돌려줍니다. 근거 없는 답은 입찰 담당자가
안 씁니다. `answer` 의 `[1] [2]` 가 `sources[n]` 과 대응합니다.

```json
{ "ok": true, "answer": "배정예산은 87,000,000원입니다 [1]",
  "sources": [{ "n": 1, "doc_id": "…", "title": "…", "agency": "…" }] }
```

`generate_answer()` 가 예외를 안 던지므로 실패해도 200 입니다. **`ok` 를 보세요.**

새 엔드포인트를 붙일 때 — 여기서 검색·생성 로직을 쓰지 마세요. `retriever.py`
와 `generation.py` 의 함수를 부르기만 합니다. 이 파일은 껍데기여야 평가
스크립트와 UI 가 같은 코드를 씁니다.

**시나리오 A/B 는 여기 없습니다.** 인프라 선택이지 고객 선택이 아니고,
외부 API 를 고르면 NDA 문서가 밖으로 나갑니다. 배포 시점에 환경변수로 정합니다.
