"""PDF 텍스트 추출.

RFP PDF는 대부분 한글에서 내보낸 텍스트 PDF라 pdfplumber로 충분하다.
다만 표가 많아서 `extract_text()`만 쓰면 셀이 뭉개진다. 그래서
표 영역은 따로 뽑아 마크다운 파이프 형태로 붙이고, 나머지 텍스트와 합친다.

텍스트가 거의 안 나오면 스캔 PDF로 보고 OCR 후보로 표시한다(OCR은 선택 과제).
"""

from __future__ import annotations

from pathlib import Path

import pdfplumber

# 페이지당 이 글자 수보다 적으면 스캔본으로 의심
SCANNED_CHAR_THRESHOLD = 40


def _table_to_text(table: list[list[str | None]]) -> str:
    rows = []
    for row in table:
        cells = [(c or "").replace("\n", " ").strip() for c in row]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _outside(boxes):
    """글자 하나가 표 밖에 있는지 판단하는 함수를 만든다."""

    def keep(obj):
        x = (obj.get("x0", 0) + obj.get("x1", 0)) / 2
        y = (obj.get("top", 0) + obj.get("bottom", 0)) / 2
        return not any(
            x0 <= x <= x1 and top <= y <= bottom for x0, top, x1, bottom in boxes
        )

    return keep


def extract_pdf_text(path: str | Path, with_tables: bool = True) -> str:
    """본문 + 표. **표는 한 번만 들어간다.**

    예전에는 page.extract_text() 로 페이지 전체를 뽑고 거기에 표를 한 번 더
    붙였다. 그러면 같은 표가 두 벌 들어간다 — 하나는 열이 뭉개진 판, 하나는
    제대로 된 판. 인덱스가 부풀고, 검색이 뭉개진 쪽을 물어오면 답이 틀린다.

    지금은 표가 차지한 영역을 본문에서 빼고 뽑은 뒤, 표를 따로 붙인다.
    """
    path = Path(path)
    pages: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            tables = page.find_tables() if with_tables else []

            if tables:
                body = (
                    page.filter(_outside([t.bbox for t in tables])).extract_text() or ""
                )
            else:
                body = page.extract_text() or ""

            parts = [body]
            for table in tables:
                rendered = _table_to_text(table.extract())
                if rendered:
                    parts.append(rendered)
            pages.append("\n".join(p for p in parts if p.strip()))
    return "\n".join(pages)


def looks_scanned(path: str | Path) -> bool:
    with pdfplumber.open(str(path)) as pdf:
        n = len(pdf.pages)
        if n == 0:
            return True
        sample = pdf.pages[: min(5, n)]
        chars = sum(len((p.extract_text() or "").strip()) for p in sample)
        return chars / len(sample) < SCANNED_CHAR_THRESHOLD
