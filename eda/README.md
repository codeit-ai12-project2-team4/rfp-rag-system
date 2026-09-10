# EDA 

원본 RFP 문서(HWP/PDF)와 메타데이터 CSV를 본격적인 전처리 파이프라인에 넣기 전에, 문서 구조·품질·메타데이터 표현이 어떻게 생겼는지 먼저 파악하기 위한 탐색적 분석 코드 모음이다. 여기서 확인한 패턴이 `src/preprocessing/`의 정제·추출 로직 설계 근거가 된다.

---

## 구성

| 파일 | 역할 |
|---|---|
| `eda_code/extract.py` | HWP/PDF 텍스트 추출 (hwp5txt → LibreOffice → raw 파서 순서로 폴백) |
| `eda_code/patterns.py` | 구조/메타데이터 탐지에 쓰는 정규식을 한 곳에 모아 관리 |
| `eda_code/metadata.py` | `patterns.py`의 정규식으로 메타데이터(사업기간, 참가자격 등) 추출 |
| `eda_code/structure.py` | 헤더 체계, 섹션 분리, 표 추정, □ 항목·제목 후보 추출 등 구조 분석 |
| `eda_code/pipeline.py` | 위 모듈들을 엮어 실제 EDA를 실행하는 함수들(`run`, `run_eda`, `run_a2`, `print_a2_result`) |
| `eda_code/check_fail_doc.py` | 특정 문서에 대해 두 추출 방식(raw 파서 vs LibreOffice) 결과를 비교·검수하는 스크립트 |
| `eda_code/examples/run_a2.py` | `run_a2` + `print_a2_result`를 실제 데이터 디렉터리에 대해 실행하는 예시 |
| `eda_code/examples/run_csv_eda.py` | 메타데이터 CSV(`data_list.csv`)에 대한 EDA(결측치·중복·컬럼 분포 등) |
| `eda_code/__init__.py` | 외부에서 자주 쓰는 함수만 노출 (`extract_text`, `extract_metadata`, `run`, `run_a2`, `run_eda`, `run_csv_eda`) |
| `eda_results/` | 위 코드를 돌려서 나온 실행 결과(출력 로그를 `.py`로 남긴 것) |

---

## 사용법

`__init__.py`가 자주 쓰는 함수를 이미 노출해두었기 때문에 `eda.eda_code`에서 바로 가져다 쓰면 된다.

**1) 문서 하나에서 메타데이터만 뽑기**

```python
from eda.eda_code import extract_text, extract_metadata
from pathlib import Path

text = extract_text(Path("data/raw/공고문.hwp"))
meta = extract_metadata(text)   # {"사업기간": ..., "참가자격": ..., "보안특약": ..., "하자보수": ...}
```

**2) 디렉터리 전체를 메타데이터 표로 뽑기**

```python
from eda.eda_code import run
from pathlib import Path

df = run(Path("data/raw"), sample_size=50)   # 파일명/확장자/글자수 + 메타데이터 컬럼을 담은 DataFrame
```

**3) 구조/메타데이터 분포 EDA (헤더 체계, 섹션 길이, 표 추정치 등)**

```python
from eda.eda_code import run_eda
from pathlib import Path

stats = run_eda(Path("data/raw"), sample_size=50)
# stats["header"], stats["metadata"], stats["section_lengths"], stats["tables"]
```

**4) 문서별 상세 EDA (추출 성공률, 텍스트 품질, 구조 마커, □ 항목, 제목 후보, 샘플)**

`eda_code/examples/run_a2.py`가 그대로 실행 예시다.

```python
from eda.eda_code.pipeline import run_a2, print_a2_result
from pathlib import Path

df_a2, texts, square_lines, heading_candidates = run_a2(Path("data/raw"))
print_a2_result(df_a2, texts, square_lines, heading_candidates)
```

**5) 메타데이터 CSV EDA**

```python
from eda.eda_code import run_csv_eda
from pathlib import Path

df = run_csv_eda(Path("data/data_list.csv"))
```

**6) 특정 문서 파싱 결과 수동 검수**

`check_fail_doc.py`는 라이브러리로 감싸져 있지 않고, 파일 맨 아래 `check_two_documents(...)` 호출부의 `filenames`/`data_dir`를 확인하려는 문서로 바꾼 뒤 스크립트를 직접 실행하는 형태다.

> **주의**: `extract.py`의 `DATA_DIR`, `check_fail_doc.py`의 `DATA_DIR`, `examples/run_csv_eda.py`의 `CSV_PATH`는 EDA 당시 담당자의 로컬 경로로 하드코딩돼 있다. 그대로 돌리지 말고 함수에 실제 `Path`를 인자로 넘기거나, 상수 값을 직접 바꿔서 사용해야 한다.

---

## 사용한 기법

**다단계 폴백 추출** — HWP는 `hwp5txt` → LibreOffice(`soffice --headless` 변환) → OLE/HWPTAG 바이너리 직접 파싱(raw parser) 순서로 시도해서, 앞 방식이 실패해도 다음 방식으로 넘어가 최대한 텍스트를 살린다. raw parser는 HWP 내부 레코드(`HWPTAG_PARA_TEXT`)를 직접 순회하며 제어 문자(부가 데이터 동반/미동반 구분)를 걸러내는 저수준 파싱이다. PDF는 `pdftotext`를 그대로 사용한다.

**정규식 레지스트리 분리** — 구조·메타데이터 탐지에 쓰는 모든 정규식을 `patterns.py` 한 곳에 모아뒀다. 같은 개념(예: "사업기간")이라도 실제 문서마다 "사업 기간/수행기간/용역기간/과업기간"처럼 표현이 갈리는 걸 `METADATA_VARIANTS`로 전부 등록해서, 표현 다양성 자체를 EDA 결과로 집계할 수 있게 했다.

**구조 마커 빈도 분석** — 로마숫자(Ⅰ.)/숫자점(1.)/숫자괄호(1))/한글점(가.)/□■○◦ 등 RFP 문서가 실제로 어떤 번호 체계를 섞어 쓰는지 줄 단위로 세어(`analyze_headers`, `analyze_structure`) 전처리 단계에서 어떤 마커를 헤더/섹션 경계로 인정해야 할지 결정하는 근거로 삼았다.

**섹션·항목 단위 분리** — 헤더로 인식되는 줄이 나올 때마다 새 섹션을 시작하는 방식(`split_sections`)으로 문서를 쪼개 섹션 길이 분포를 보고, □ 항목(`□ (사업기간)` 같은 형태)은 원본/정제 라벨을 함께 추출(`extract_square_items`)해 표 유사 구조가 얼마나 섞여 있는지 가늠했다(`analyze_table_like`).

**텍스트 품질 진단** — 깨진 문자, NULL 문자, 제어 문자, 5칸 이상 긴 공백, 4줄 이상 과도한 줄바꿈 등을 문서별로 집계(`analyze_text_quality`)해서, 이후 `src/preprocessing/rfp/clean.py`의 정제 정규식(모지바케 제거, 긴 공백/줄바꿈 정리 등)을 어떤 기준으로 만들지 판단하는 데 썼다.

**교차 검증(diff 기반 수동 검수)** — 같은 문서를 raw parser와 LibreOffice 두 방식으로 각각 파싱한 뒤 `difflib.SequenceMatcher`로 유사도를 재서, 두 결과가 90% 미만으로 갈리면 "수동 검수 필요"로 표시했다(`check_fail_doc.py`). 자동 추출 결과를 맹신하지 않고 방식 간 불일치를 신호로 활용하는 방식이다.

**일반 테이블형 EDA** — 원본 문서 EDA와는 별개로, 메타데이터 CSV(`data_list.csv`)에 대해서는 결측치·중복·컬럼별 고유값 분포·텍스트 길이 통계·확장자 분포를 pandas로 표준적으로 분석했다(`examples/run_csv_eda.py`).

---

## 확인한 내용

`eda_results/`에 남아 있는 실제 실행 결과 기준. 표본은 원본 RFP 문서 100건(hwp 96 · pdf 4)과 메타데이터 CSV 100행.

**문서 추출 (A-2, `eda_original_doc.py`)**

- 텍스트 추출 성공률 100/100(100%).
- 문서당 글자 수: 평균 26,342자, 최소 7,374자, 최대 194,043자(고려대학교 PDF) — 문서 간 분량 편차가 매우 크다(표준편차 23,401).
- PDF 문서는 hwp와 달리 제어문자가 다수 검출됐다(고려대학교 문서 297건). hwp는 거의 0건이라, PDF 추출 경로에 별도 정제가 필요하다는 근거가 됐다.
- 구조 마커 빈도: 숫자_점(6,646) > 한글_점(3,282) > ○(2,698) > □(2,623) > ※(2,233) > 숫자_괄호(1,992) > ◦(978) > 한글_괄호(457) > 괄호숫자(425) > 로마숫자(282) > ■(27) > ●(1). 로마숫자·■·●는 희소해서 헤더 판정의 주력으로 쓰기 어렵다는 걸 확인했다.
- □ 항목 상위권이 "법률 및 고시"(65건 문서), "서비스 접근 및 전달 분야"·"인터페이스 및 통합 분야"·"플랫폼 및 기반구조 분야"·"요소기술 분야"(각 64건), "보안 분야"(62건)로, 대부분 문서가 같은 표준 카테고리 세트를 git  있었다.

**특수문자/노이즈 검증 (`eda_Special symbol.py`)**

- `<표>`/`<그림>` 같은 HWP 태그 잔재가 문서에 따라 수백 건씩(대표 문서 508건) 남아있는 걸 확인 — `clean.py`의 태그 잔재 제거 정규식 도입 근거가 됐다.
- 플레이스홀더(○○○/ㅇㅇㅇ)가 공동수급협정서 등 표준 계약서식에서 실제 값 대신 다수 사용됨을 확인(대표 문서 46건).
- 페이지번호 정규식이 진짜 페이지 번호("- 38 -")뿐 아니라 표 안 숫자 셀("100", "3" 등)까지 잡는 과매칭 사례를 발견 — 정규식을 더 다듬어야 한다는 근거가 됐다.
- 문서 간 반복되는 boilerplate 후보 49건을 실측으로 뽑아냈다(예: "□ 플랫폼 및 기반구조 분야"가 60개 문서에서 반복 등장). 이 목록이 전처리의 boilerplate 제거 로직 기준이 됐다.

**메타데이터 CSV (`eda_csv_file.py`, 100행 × 12열)**

- 완전 중복 행 0건, 파일명 100% 고유 — 같은 파일이 중복 등록되진 않았다.
- 결측치 비율: 공고번호 18%, 공고차수 18%, 사업금액 1%, 입찰참여시작일 26%, 입찰참여마감일 8%. 그 외 컬럼(사업명·발주기관·공개일자·사업요약·파일명 등)은 결측 없음.
- 사업금액은 0원부터 약 112.7억 원까지 편차가 크게 분포.
- 파일형식은 hwp 96건, pdf 4건 — 원본 문서 100건 표본과 일치.

---

## 참고

`eda_results/`에는 위 함수들을 실제로 돌린 출력 로그가 파일로 남아 있다(`eda_original_doc.py`, `eda_csv_file.py`, `eda_Special symbol.py`). 재사용 가능한 코드가 아니라 실행 결과 기록이므로, 로직을 다시 쓰고 싶다면 `eda_code/` 쪽을 참고하는 게 맞다.
