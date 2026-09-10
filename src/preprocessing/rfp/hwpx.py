"""HWPX 추출. 차세대 나라장터가 내려주는 형식이다.

**HWP 와 완전히 다른 파일이다.** hwp 는 OLE 복합문서(바이너리)고 hwpx 는
OWPML — zip 안에 XML 이다. 그래서 `hwp.py` 의 저수준 파서가 한 줄도 안 통한다.

    ~/rfp-rag-system$ python -m preprocessing.rfp.hwpx   자체 검사

**표 렌더러는 `hwp.py` 것을 그대로 쓴다.** 그래야 hwp 와 hwpx 가 같은 모양의
본문을 내놓고, `clean.py` 와 청킹이 어느 쪽인지 몰라도 된다.
XML 이라 셀 주소를 복원할 필요가 없어서 표가 오히려 hwp 보다 잘 나온다.

원본: 이 파일만 `pipeline.py` 에 없던 것이다. 크롤러가 hwpx 를 받아오는데
파이프라인이 `.hwp`/`.pdf` 만 읽어서 **조용히 버려지고 있었다** (2026-09-03).
"""

import zipfile
from pathlib import Path
from xml.etree import ElementTree

from preprocessing.rfp.common import ExtractionResult, TableParseResult, _check_hangul
from preprocessing.rfp.hwp import _is_keyvalue_table, _render_keyvalue, _render_matrix

# OWPML 문단 네임스페이스. 한글 버전에 따라 연도가 다를 수 있어 접두어로 찾는다.
_NS = "http://www.hancom.co.kr/hwpml/2011/paragraph"


def _tag(elem):
    """`{ns}tbl` → `tbl`. 네임스페이스 연도가 달라도 태그로만 판정한다."""
    return elem.tag.rsplit("}", 1)[-1]


def _cell_text(cell):
    """셀 하나의 글자. 셀 안 문단이 여럿이면 공백으로 잇는다."""
    parts = [t.text or "" for t in cell.iter() if _tag(t) == "t"]
    return " ".join("".join(parts).split())


def _table_grid(tbl):
    """`<hp:tbl>` 을 `[[셀, 셀], ...]` 로. 못 읽으면 None.

    **병합 셀은 colSpan/rowSpan 을 안 편다.** hwp 쪽 `_render_row_block` 이
    그 경우를 다루는데, 거기 맞추려면 표 전체를 다시 격자로 펴야 한다.
    ponytail: 지금은 읽힌 셀만 순서대로 넣는다. 병합이 많은 표에서 열이
    밀릴 수 있다 — 실제 파일로 표유실률을 재고 나서 필요하면 편다.
    """
    rows = []
    for tr in tbl.iter():
        if _tag(tr) != "tr":
            continue
        cells = [_cell_text(tc) for tc in tr if _tag(tc) == "tc"]
        if cells:
            rows.append(cells)
    if not rows or not rows[0]:
        return None
    width = max(len(r) for r in rows)
    return [r + [""] * (width - len(r)) for r in rows]


def _render(grid):
    """hwp 와 **같은** 모양으로 렌더링한다. 2열이면 키:값, 아니면 마크다운 표."""
    if len(grid[0]) == 2 and _is_keyvalue_table(grid):
        return _render_keyvalue(grid)
    return _render_matrix(grid)


def extract_hwpx_document(path: Path) -> ExtractionResult:
    """hwpx 한 건에서 본문과 표를 뽑는다.

    Args:
        path: `.hwpx` 파일.

    Returns:
        ExtractionResult. hwp/pdf 쪽과 같은 형태라 상위 디스패처가 구분하지 않는다.
    """
    blocks, total, failed = [], 0, 0
    try:
        with zipfile.ZipFile(path) as zf:
            # section0, section1 ... 순서가 곧 문서 순서다. 이름순으로 읽는다.
            names = sorted(
                n for n in zf.namelist()
                if n.startswith("Contents/section") and n.endswith(".xml")
            )
            if not names:
                return ExtractionResult(
                    text="", extractor=None, table_parse_success=False,
                    tables_total=0, tables_failed=0,
                    error_reason="Contents/sectionN.xml 이 없습니다 (hwpx 가 아닐 수 있음)",
                    fallback_used=False, attempted_errors={},
                )
            for name in names:
                root = ElementTree.fromstring(zf.read(name))
                # **표 안의 글자를 먼저 표시해 둔다.** `iter()` 는 표 안까지
                # 훑기 때문에, 안 걸러내면 셀 내용이 표로 한 번 본문으로 또
                # 한 번 나온다. 청크가 두 배로 부풀고 검색이 같은 말을 두 번 센다.
                #
                # ponytail: 표 안에 표가 있으면 안쪽 표는 바깥 셀 글자로 한 번,
                # 제 표로 또 한 번 나온다. 실제 파일에서 보이면 그때 재귀로 막는다.
                inside = {
                    id(t)
                    for tbl in root.iter() if _tag(tbl) == "tbl"
                    for t in tbl.iter() if _tag(t) == "t"
                }
                for node in root.iter():
                    kind = _tag(node)
                    if kind == "tbl":
                        total += 1
                        grid = _table_grid(node)
                        if grid:
                            blocks.append(_render(grid))
                        else:
                            failed += 1
                            blocks.append("[표 복원 실패: 셀을 못 읽었습니다]")
                    elif kind == "t" and node.text and id(node) not in inside:
                        blocks.append(node.text)
    except (zipfile.BadZipFile, ElementTree.ParseError, KeyError, OSError) as error:
        return ExtractionResult(
            text="", extractor=None, table_parse_success=False,
            tables_total=total, tables_failed=total,
            error_reason=f"{type(error).__name__}: {error}",
            fallback_used=False, attempted_errors={"hwpx": str(error)},
        )

    text = "\n".join(b for b in blocks if b and b.strip())
    # hwp/pdf 쪽과 같은 기준을 쓴다. 한글 비율이 낮으면 뭔가 잘못 뽑힌 것이다.
    ok = _check_hangul(text)
    return ExtractionResult(
        text=text if ok else "",
        extractor="hwpx" if ok else None,
        table_parse_success=failed == 0,
        tables_total=total,
        tables_failed=failed,
        error_reason=None if ok else "한글 비율이 낮습니다 (추출 실패로 봅니다)",
        fallback_used=False,
        attempted_errors={},
    )


def _selftest():
    """샘플 hwpx 를 만들어서 본문·표가 나오는지 본다. 실제 파일은 못 대신한다."""
    import io

    ns = 'xmlns:hp="http://www.hancom.co.kr/hwpml/2011/paragraph"'
    section = (
        f'<?xml version="1.0" encoding="UTF-8"?><hp:sec {ns}>'
        "<hp:p><hp:run><hp:t>제안요청서</hp:t></hp:run></hp:p>"
        "<hp:p><hp:run><hp:tbl>"
        "<hp:tr><hp:tc><hp:subList><hp:p><hp:run><hp:t>구분</hp:t>"
        "</hp:run></hp:p></hp:subList></hp:tc>"
        "<hp:tc><hp:subList><hp:p><hp:run><hp:t>내용</hp:t>"
        "</hp:run></hp:p></hp:subList></hp:tc></hp:tr>"
        "<hp:tr><hp:tc><hp:subList><hp:p><hp:run><hp:t>배정예산</hp:t>"
        "</hp:run></hp:p></hp:subList></hp:tc>"
        "<hp:tc><hp:subList><hp:p><hp:run><hp:t>130,000,000원</hp:t>"
        "</hp:run></hp:p></hp:subList></hp:tc></hp:tr>"
        "</hp:tbl></hp:run></hp:p>"
        "<hp:p><hp:run><hp:t>과업 내용은 다음과 같다</hp:t></hp:run></hp:p>"
        "</hp:sec>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("mimetype", "application/hwp+zip")
        z.writestr("Contents/section0.xml", section)
    sample = Path("/tmp/_selftest.hwpx")
    sample.write_bytes(buf.getvalue())

    got = extract_hwpx_document(sample)
    assert got.extractor == "hwpx", (got.extractor, got.error_reason)
    assert got.tables_total == 1 and got.tables_failed == 0, got
    assert "제안요청서" in got.text and "과업 내용" in got.text
    assert "배정예산" in got.text and "130,000,000원" in got.text
    # 표 안 글자가 본문으로 또 나오면 안 된다. 이 검사가 실제로 버그를 잡았다.
    assert got.text.count("배정예산") == 1, f"표 내용이 중복됐다\n{got.text}"
    assert got.text.count("130,000,000원") == 1, got.text
    print(got.text)

    broken = Path("/tmp/_broken.hwpx")
    broken.write_bytes(b"not a zip")
    bad = extract_hwpx_document(broken)
    assert bad.extractor is None and bad.error_reason, bad
    print("\nselftest 통과 — 본문·표 추출, 깨진 파일은 사유와 함께 실패")


if __name__ == "__main__":
    _selftest()
