"""PDF 추출. 본문과 표를 나눠 뽑는다.

원본: `preprocessing/pipeline.py` (전처리팀). 분할 경위는 `src/preprocessing/README.md`.
"""

from pathlib import Path

import pdfplumber

# from config import settings as path_settings
from preprocessing.rfp.common import ExtractionResult, HwpParseError
from preprocessing.rfp.hwp import (
    _is_keyvalue_table,
    _render_keyvalue,
    _render_matrix,
)

# ============================================================
# 11. PDF 추출 - 본문/표 분리
# ============================================================


def _bbox_inside(word_bbox: tuple, table_bboxes: list) -> bool:
    x0, top, x1, bottom = word_bbox
    for bx0, btop, bx1, bbottom in table_bboxes:
        if x0 >= bx0 and x1 <= bx1 and top >= btop and bottom <= bbottom:
            return True
    return False


# [수정 5 - PDF 표 추출 강화] find_tables()를 선(lines)/텍스트(text) 두
# 전략으로 각각 시도해, 셀이 더 많이 채워진 쪽을 채택한다. 페이지 단위로
# 전략 하나만 고르고 두 결과를 합치지 않으므로, 같은 표가 두 번 추출되는
# 중복 문제는 생기지 않는다.
_PDF_TABLE_STRATEGIES = [
    {"vertical_strategy": "lines", "horizontal_strategy": "lines"},
    {"vertical_strategy": "text", "horizontal_strategy": "text"},
]


def _find_best_pdf_tables(page) -> list:
    """두 표 탐지 전략 중, 채워진 셀 수가 더 많은 결과를 반환한다."""

    best_tables, best_filled = [], -1

    for settings in _PDF_TABLE_STRATEGIES:
        try:
            tables = page.find_tables(table_settings=settings)
        except Exception:  # noqa: BLE001, S112
            # [버그 수정] pdfplumber/pdfminer가 던지는 파싱 오류도
            # RuntimeError가 아닐 수 있다. 한 전략이 실패해도 다른 전략은
            # 계속 시도해야 하므로 넓게 잡는다.
            continue

        filled = 0
        for table in tables:
            grid = table.extract() or []
            filled += sum(1 for row in grid for cell in row if cell)

        if filled > best_filled:
            best_filled, best_tables = filled, tables

    return best_tables


def extract_pdf_document(path: Path) -> ExtractionResult:
    """
    pdfplumber.find_tables()로 표 영역을 먼저 찾아 본문에서 제외하고,
    표는 별도로 렌더링해 문서 끝에 붙인다. (같은 표 중복 방지)
    """

    try:
        body_parts = []
        table_blocks = []
        tables_total = 0
        tables_failed = 0

        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                found_tables = _find_best_pdf_tables(page)  # [수정 5]
                table_bboxes = [t.bbox for t in found_tables]

                words = page.extract_words()
                body_words = [
                    w
                    for w in words
                    if not _bbox_inside(
                        (w["x0"], w["top"], w["x1"], w["bottom"]),
                        table_bboxes,
                    )
                ]
                body_parts.append(" ".join(w["text"] for w in body_words))

                for t in found_tables:
                    tables_total += 1
                    grid = t.extract()

                    if not grid or not grid[0]:
                        tables_failed += 1
                        table_blocks.append("[표 복원 실패] 빈 표")
                        continue

                    grid = [[cell or "" for cell in row] for row in grid]

                    if len(grid[0]) == 2 and _is_keyvalue_table(grid):
                        table_blocks.append(_render_keyvalue(grid))
                    else:
                        table_blocks.append(_render_matrix(grid))

        text = "\n\n".join(body_parts) + "\n\n" + "\n\n".join(table_blocks)

        if not text.strip():
            raise HwpParseError("pdfplumber 결과가 비어 있음")

        return ExtractionResult(
            text=text,
            extractor="pdfplumber",
            table_parse_success=(tables_failed == 0),
            tables_total=tables_total,
            tables_failed=tables_failed,
            error_reason=None,
            fallback_used=False,
            attempted_errors={},
        )

    except Exception as e:  # noqa: BLE001
        # [버그 수정] pdfplumber.open() 등에서 나는 실제 예외(OSError,
        # PDFSyntaxError 등)는 RuntimeError가 아니어서 기존 except로는
        # 못 잡고 그대로 새어나갔다.
        return ExtractionResult(
            text="",
            extractor=None,
            table_parse_success=False,
            tables_total=0,
            tables_failed=0,
            error_reason=str(e),
            fallback_used=False,
            attempted_errors={"pdfplumber": str(e)},
        )
