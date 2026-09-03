"""확장자별 디스패처와 문서 1건 처리.

원본: `preprocessing/pipeline.py` (전처리팀). 분할 경위는 `src/preprocessing/README.md`.
"""

from pathlib import Path

from preprocessing.rfp.clean import clean_text_verbose, strip_table_markup
from preprocessing.rfp.common import FIELD_ALIASES, ExtractionResult
from preprocessing.rfp.hwp import extract_hwp_document
from preprocessing.rfp.meta import extract_metadata
from preprocessing.rfp.pdf import extract_pdf_document

# ============================================================
# 12. 확장자 통합 디스패처
# ============================================================


def extract_document(path: Path) -> ExtractionResult:

    suffix = path.suffix.lower()

    if suffix == ".hwp":
        return extract_hwp_document(path)

    if suffix == ".pdf":
        return extract_pdf_document(path)

    return ExtractionResult(
        text="",
        extractor=None,
        table_parse_success=False,
        tables_total=0,
        tables_failed=0,
        error_reason=f"지원하지 않는 확장자: {suffix}",
        fallback_used=False,
        attempted_errors={},
    )


# ============================================================
# 15. 문서 1건 처리
# ============================================================


def process_document(path: Path) -> dict:

    extraction = extract_document(path)

    base = {
        "파일명": path.name,
        "extractor": extraction.extractor,
        "table_parse_success": extraction.table_parse_success,
        "tables_total": extraction.tables_total,
        "tables_failed": extraction.tables_failed,
        "tables_partial": extraction.tables_partial,  # [수정 1]
        "fallback_used": extraction.fallback_used,
        "error_reason": extraction.error_reason,
        # hwp_raw가 채택되지 못한 이유 (성공한 문서라도 조용한 폴백을 추적하기 위함)
        "hwp_raw_skipped_reason": extraction.attempted_errors.get("hwp_raw"),
        "clean_text": "",
        "clean_text_for_generation": "",
        "_warnings": [],
    }
    base.update({k: None for k in FIELD_ALIASES})

    if not extraction.text:
        return base

    clean, warnings_list = clean_text_verbose(extraction.text)

    # [표현 이원화] clean은 _render_table이 표 형태별로(단순 격자는
    # Markdown, 병합 셀은 row-block, key-value는 그대로) 이미 구조화해
    # 놓은 상태 - 이걸 그대로 생성(LLM)용으로 쓴다. 검색/임베딩용은 여기서
    # 마크업만 걷어낸 평문을 별도로 만든다. 표 성공/부분/실패 판정과 경고
    # 집계(위 tables_failed/tables_partial)는 fill_ratio 기준 그대로 유지.
    clean_for_generation = clean
    clean_for_embedding = strip_table_markup(clean)

    if extraction.tables_failed > 0:
        warnings_list.append(
            f"표 {extraction.tables_total}개 중 {extraction.tables_failed}개 복원 실패"
        )

    # [수정 1] 부분 복원된 표도 경고로 남겨 검수 시 눈에 띄게 한다.
    if extraction.tables_partial > 0:
        warnings_list.append(
            f"표 {extraction.tables_total}개 중 {extraction.tables_partial}개 부분 복원"
        )

    base["clean_text"] = clean_for_embedding
    base["clean_text_for_generation"] = clean_for_generation
    base["_warnings"] = warnings_list
    base.update(extract_metadata(extraction.text))

    return base


