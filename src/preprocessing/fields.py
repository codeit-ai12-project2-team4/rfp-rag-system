"""공고 메타데이터 필드 이름표. **여기 하나만 고친다.**

같은 아홉 개 필드를 설명하는 표가 네 군데 흩어져 있었고, 이미 갈라져 있었다.

    run.py COLUMNS                   CSV(공백 있는 이름) → 영문   11개
    run.py _LANGCHAIN_META           파이프라인 메타 → 영문        9개  ← 두 개 빠짐
    rfp/common.ORIGINAL_METADATA_COLUMNS  CSV 에서 살릴 컬럼      12개  ← 하나 유령
    rfp/meta.build_doc_schema_record      하드코딩                10개  ← 안 쓰임

갈라진 결과:

- `입찰참여시작일`(bid_open_at) 이 파이프라인 경로에서 **조용히 사라진다.**
  CSV 에도 있고 청크 메타에도 남는데, `from_langchain` 이 안 옮겨서
  검색단에서는 존재하지 않는 필드가 된다.
- `공개기관` 은 CSV 에 아예 없다. 크롤러 HEADER 에도 없다. 전처리를 돌릴
  때마다 "없는 컬럼(건너뜀)" 이 찍힌다 — 아무도 안 읽는 경고가 하나 더 늘었다.
- `공고차수`(notice_seq) 는 doc_id 를 만들 때만 곁눈질로 읽고 메타에는 안 남는다.

CSV 컬럼 이름에 공백이 섞여 있는 건(`공고 번호` vs `공고번호`) 원본 파일 사정이다.
읽는 쪽에서 공백을 지워 맞추므로 여기서는 **공백 없는 이름 하나만** 쓴다.
"""

# (CSV 컬럼, 코드에서 쓰는 이름). 순서는 CSV 순서다.
FIELDS = [
    ("공고번호", "notice_no"),
    ("공고차수", "notice_seq"),
    ("사업명", "title"),
    ("사업금액", "budget"),
    ("발주기관", "agency"),
    ("공개일자", "published_at"),
    ("입찰참여시작일", "bid_open_at"),
    ("입찰참여마감일", "bid_close_at"),
    ("사업요약", "summary"),
    ("파일형식", "file_type"),
    ("파일명", "file_name"),
]

#: CSV 에서 살릴 컬럼 (한글, 공백 없음).
CSV_COLUMNS = [korean for korean, _ in FIELDS]

#: 한글 → 영문.
TO_ENGLISH = dict(FIELDS)


def normalize_columns(columns):
    """CSV 컬럼 이름에서 공백을 지워 정규 이름으로. `{실제이름: 정규이름}`.

    `공고 번호` 와 `공고번호` 가 같은 것이라고 알려 주는 게 전부다.

    Args:
        columns: 읽어 온 DataFrame 의 컬럼 이름들.

    Returns:
        dict: 실제 컬럼 이름 → 공백 없는 이름. 정규 이름에 없는 컬럼은 뺀다.
    """
    known = set(CSV_COLUMNS)
    return {
        col: col.replace(" ", "")
        for col in columns
        if col.replace(" ", "") in known
    }
