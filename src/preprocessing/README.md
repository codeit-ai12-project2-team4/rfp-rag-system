# preprocessing — 두 갈래가 같이 산다

이 폴더에는 **성격이 다른 두 묶음**이 있다. 이름이 비슷해서 헷갈리기 쉬운데,
하나도 죽지 않았다.

```
preprocessing/
├── clean.py  hwp.py  hwp_table.py  pdf.py  toc.py  run.py   ← 검색 파트
├── pipeline.py                                              ← 전처리팀 원본. 현재 비활성
└── rfp/                                                     ← pipeline.py 를 나눈 것
    ├── common.py  hwp.py  pdf.py  extract.py
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

## 왜 나눴나 (2026-09-03)

전처리팀에서 `pipeline.py` 작업 권한을 넘겨받았다. 한 파일에 **2,430줄**이었다.

나누기 전에 세 가지가 막혀 있었다.

### 1. import 하면 죽었다

파일 끝, 모듈 최상위(함수 밖)에 전처리팀의 작업용 코드가 남아 있었다.

```python
df = pd.read_csv("")                                 # ← import 즉시 예외
DOC_PATH = Path(r"C:\Users\asd\Desktop\...")         # 남의 윈도우 경로
DOC_PATH.parent.mkdir(parents=True, exist_ok=True)   # 폴더까지 만든다
write_jsonl(df, DOC_PATH, ...)
```

`from preprocessing.pipeline import run_pipeline` 자체가 불가능했다.
자동화에 붙이려면 이게 첫 관문이었다.

### 2. 비트 OR

```python
chunk_size: int = int(retrieval_settings.SIZE) | 1500      # or 가 아니다
```

`1500 | 1500` 은 우연히 1500 이라 지금은 맞는다. 하지만 `SIZE=1200` 으로
바꾸면 `1200 | 1500 = 1532` 가 **오류 없이** 들어간다. config 에 이미 기본값이
있으므로 폴백을 지우고 상수만 쓴다. - 이건 파이썬 문법 오해로 생긴 해프닝

### 3. 직접 실행하면 청크가 안 나왔다

`__main__` 이 `enable_chunk_output` 을 안 켜서 전처리본만 나왔다.
청크가 없으면 색인을 못 만든다.

---

## 어떻게 나눴나

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

### 왜 `rfp/` 하위 패키지인가

`hwp.py` · `pdf.py` · `clean.py` 가 바깥에 **이미 있다.** 같은 이름으로 풀어놓으면
덮어쓰거나, 안 덮어도 어느 쪽을 import 했는지 매번 헷갈린다.
하위 패키지로 두면 `preprocessing.rfp.hwp` 와 `preprocessing.hwp` 가 눈으로
구분된다.

### 진입점

```python
from preprocessing.rfp import run_pipeline
```

`pipeline.py` 를 직접 import 하는 곳은 이제 없다.
`prepare.py` 와 `ingest.py` 를 `preprocessing.rfp` 로 돌렸다.

---

## `pipeline.py` 는 왜 남겨 뒀나

**지웠다가 되돌린 게 아니라, 원본을 그대로 둔 것이다.** `git` 상태와 같다.

- 전처리팀이 아직 이 파일을 기준으로 이야기한다. 대조할 원본이 필요하다
- 나눈 결과가 원본과 같은지 의심될 때 여기와 비교한다
- 파일 끝 스크래치 코드 때문에 **import 하면 안 된다.** 읽기 전용으로만 본다

전처리팀과 합의가 끝나면 지운다. 그때까지는 이 README 가 "왜 두 벌인가" 의 답이다.

---

## 안 쓰게 된 것

| | 상태 |
|---|---|
| `pipeline.py` | 참조용으로만. import 하는 코드 없음 |
| `rfp_preprocessing_pipeline.py` | **파일이 이미 없다.** `ingest.py` 가 이걸 import 하고 있어서 같이 고쳤다 |

바깥 `hwp/hwp_table/pdf/clean/toc/run` 은 **전부 살아 있다.** 지우면 API 가 죽는다.
