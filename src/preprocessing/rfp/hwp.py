"""HWP 추출. OLE 레코드를 직접 읽어 표 구조까지 복원한다.

원본: `preprocessing/pipeline.py` (전처리팀). 분할 경위는 `src/preprocessing/README.md`.
"""

import re
import struct
import subprocess
import tempfile
import zlib
from pathlib import Path

import olefile

from preprocessing.rfp.common import (
    _HWP_CONTROL_SKIP_ONLY,
    _HWP_CONTROL_WITH_EXTRA,
    _SIBLING_CONTAINER_TAGS,
    HWP_EXTRACTION_METHODS,
    HWPTAG_CTRL_HEADER,
    HWPTAG_LIST_HEADER,
    HWPTAG_PARA_HEADER,
    HWPTAG_PARA_TEXT,
    HWPTAG_TABLE,
    KEYVALUE_MAX_KEY_LENGTH,
    TABLE_CTRL_ID,
    TABLE_FULL_SUCCESS_RATIO,
    TABLE_PARTIAL_MIN_RATIO,
    ExtractionResult,
    HwpParseError,
    TableParseResult,
    _check_hangul,
    _describe_hwp_tag,
)

# ============================================================
# 7. HWP 추출 - hwp5txt / LibreOffice (텍스트만, 표 구조 없음)
# ============================================================


def extract_text_hwp5txt(path: Path) -> str:

    result = subprocess.run(
        ["hwp5txt", str(path)], capture_output=True, text=True, timeout=60, check=False
    )

    if result.returncode != 0:
        raise HwpParseError(f"hwp5txt 실패: {result.stderr[:300]}")

    if not result.stdout.strip():
        raise HwpParseError("hwp5txt 결과가 비어 있음")

    return result.stdout


def extract_text_libreoffice(path: Path) -> str:

    with tempfile.TemporaryDirectory() as tmp_dir:
        result = subprocess.run(
            [
                "soffice",
                "--headless",
                "--convert-to",
                "txt:Text",
                "--outdir",
                tmp_dir,
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

        if result.returncode != 0:
            raise HwpParseError(f"soffice 변환 실패: {result.stderr[:300]}")

        produced = list(Path(tmp_dir).glob("*.txt"))

        if not produced:
            raise HwpParseError("soffice 변환 결과 파일 없음")

        text = produced[0].read_text(encoding="utf-8", errors="replace")

        if not text.strip():
            raise HwpParseError("soffice 변환 결과가 비어 있음")

        return text


# ============================================================
# 8. HWP 추출 - OLE 레코드 저수준 유틸
# ============================================================


def _hwp_is_compressed(ole: olefile.OleFileIO) -> bool:
    header = ole.openstream("FileHeader").read()
    return bool(struct.unpack("<I", header[36:40])[0] & 0x01)


def _hwp_iter_records(data: bytes):
    """(tag_id, level, payload)를 순서대로 yield한다."""

    i, n = 0, len(data)

    while i + 4 <= n:
        header = struct.unpack("<I", data[i : i + 4])[0]
        tag_id = header & 0x3FF
        level = (header >> 10) & 0x3FF
        size = (header >> 20) & 0xFFF
        i += 4

        if size == 0xFFF:
            size = struct.unpack("<I", data[i : i + 4])[0]
            i += 4

        payload = data[i : i + size]
        i += size

        yield tag_id, level, payload


def _hwp_decode_para_text(payload: bytes) -> str:

    chars = []
    i, n = 0, len(payload)

    while i + 2 <= n:
        code = struct.unpack("<H", payload[i : i + 2])[0]
        i += 2

        if code in _HWP_CONTROL_SKIP_ONLY:
            continue

        if code in _HWP_CONTROL_WITH_EXTRA:
            # [수정 1] 탭(9)은 뒤따르는 14바이트가 리더문자/폭 등의
            # 바이너리 파라미터일 뿐 텍스트가 아니므로 건너뛴다.
            # 사람이 읽는 텍스트에서는 탭 문자 하나로만 남긴다.
            if code == 9:
                chars.append("\t")
            i += 14
            continue

        if code in (10, 13):
            chars.append("\n")
            continue

        if 0xD800 <= code <= 0xDBFF:
            # 상위 서로게이트: 다음 2바이트가 하위 서로게이트면 하나의 문자로 결합
            if i + 2 <= n:
                low = struct.unpack("<H", payload[i : i + 2])[0]
                if 0xDC00 <= low <= 0xDFFF:
                    i += 2
                    codepoint = 0x10000 + (code - 0xD800) * 0x400 + (low - 0xDC00)
                    chars.append(chr(codepoint))
                    continue
            # 짝이 없는 상위 서로게이트 - 손상된 데이터로 보고 폐기
            continue

        if 0xDC00 <= code <= 0xDFFF:
            # 짝 없이 단독으로 나온 하위 서로게이트 - 폐기
            continue

        chars.append(chr(code))

    return "".join(chars)


def _hwp_parse_table_dims(payload: bytes) -> tuple[int, int]:
    rows = struct.unpack_from("<H", payload, 4)[0]
    cols = struct.unpack_from("<H", payload, 6)[0]
    return rows, cols


def _hwp_parse_list_header_addr(payload: bytes) -> dict:

    if len(payload) < 16:
        return {"col": 0, "row": 0, "colspan": 1, "rowspan": 1}

    return {
        "col": struct.unpack_from("<H", payload, 8)[0],
        "row": struct.unpack_from("<H", payload, 10)[0],
        "colspan": struct.unpack_from("<H", payload, 12)[0],
        "rowspan": struct.unpack_from("<H", payload, 14)[0],
    }


# ============================================================
# 9. HWP 추출 - 표 구조 복원 (핵심)
# ============================================================


def extract_text_hwp_raw_structured(path: Path) -> tuple[str, TableParseResult]:
    """
    OLE 레코드를 레벨 기준 스택으로 순회하며 표를 복원한다.

    레코드 계층:
        PARA_HEADER
          PARA_TEXT           본문 글자
          CTRL_HEADER         payload[:4][::-1] == b"tbl " 이면 표 시작
            TABLE             행수/열수
            LIST_HEADER       칸 하나의 시작 (칸 주소 포함)
              PARA_HEADER     칸 안 문단
                PARA_TEXT     칸 안 글자

    표가 일부 실패해도(격자를 못 채우면) 문서 전체를 실패로 보지 않고
    '-'로 채워 표시한 채 나머지는 그대로 사용한다.

    [수정 2 - 핵심 버그]
    HWP5 포맷에서 LIST_HEADER(칸/리스트 시작)와 그 안의 "첫 번째 이후"
    문단들의 PARA_HEADER는 같은 level 값을 공유한다(문단마다 새 레벨이
    아니라, 같은 리스트의 형제 항목이기 때문). 그런데 기존 pop 조건
    `stack[-1]["level"] >= level`은 "같음(==)"도 pop 대상으로 취급해서,
    칸이 열리자마자 그 칸의 첫 PARA_HEADER가 도착하는 순간 칸 프레임이
    내용물이 채워지기도 전에 즉시 닫혀버렸다. 그 결과 칸의 실제 텍스트는
    이미 부모(표) 프레임으로 새어나가 버리고, cells[addr]["text"]는
    거의 항상 빈 문자열로 남았다. (실측: 이 문서 기준 표 렌더링 라인
    1148개 중 1141개, 즉 99.4%가 완전히 빈 값으로 나왔음 - 표
    "성공"으로 집계된 178개 표도 사실상 내용이 전부 유실된 상태였다.)

    고친 조건:
      (a) 더 얕은 레벨(level < frame.level)이면 무조건 닫는다.
      (b) 같은 레벨(level == frame.level)이어도, 들어오는 레코드가
          "새 형제 컨테이너의 시작"을 뜻하는 CTRL_HEADER/LIST_HEADER일
          때만 닫는다. PARA_HEADER(같은 리스트의 다음 문단)는 같은
          레벨이어도 열어둔 채로 둔다.

    이 수정은 표 셀 안에 인라인 개체(텍스트박스 등)가 들어있어 더 깊은
    level에서 발생하는 자체 LIST_HEADER(표와 무관한 문단 리스트)를
    표의 칸으로 잘못 인식해 주소가 덮어써지는 문제(이번에 실패로
    집계됐던 유일한 표, table_id=72)도 함께 해결한다 - 그 LIST_HEADER는
    level이 셀 프레임의 level보다 깊으므로 이제는 셀이 아닌 별도의
    "list" 프레임으로 올바르게 격리된다.
    """

    with olefile.OleFileIO(str(path)) as ole:
        compressed = _hwp_is_compressed(ole)

        sections = sorted(
            (e for e in ole.listdir() if len(e) == 2 and e[0] == "BodyText"),
            key=lambda e: int(re.sub(r"\D", "", e[1]) or 0),
        )

        if not sections:
            raise HwpParseError("BodyText 스트림을 찾을 수 없음")

        root = {"type": "root", "level": -1, "buffer": [""]}
        stack = [root]

        tables_total = 0
        tables_failed = 0
        tables_partial = 0  # [수정 1] 부분 복원된 표 개수 추적

        for entry in sections:
            raw = ole.openstream(entry).read()
            data = zlib.decompressobj(-15).decompress(raw) if compressed else raw

            for tag_id, level, payload in _hwp_iter_records(data):
                # [수정 2] 레벨 기준으로 형제/부모 경계 정리
                while len(stack) > 1 and (
                    stack[-1]["level"] > level
                    or (
                        stack[-1]["level"] == level
                        and tag_id in _SIBLING_CONTAINER_TAGS
                    )
                ):
                    finished = stack.pop()
                    if finished["type"] == "table_pending":
                        # [진단용] TABLE 레코드를 못 받은 채 닫히는 경우,
                        # 어떤 태그/레벨이 이 프레임을 닫았는지 기록한다.
                        # pop 조건 자체는 그대로이고, 닫히는 순간의 정보만
                        # 남긴다.
                        finished["_closed_by"] = (
                            f"{_describe_hwp_tag(tag_id)}(level={level})"
                        )
                    tables_total, tables_failed, tables_partial = _finalize_frame(
                        finished,
                        stack[-1],
                        tables_total,
                        tables_failed,
                        tables_partial,
                    )

                if tag_id == HWPTAG_CTRL_HEADER:
                    if payload[:4][::-1] == TABLE_CTRL_ID:
                        stack.append({
                            "type": "table_pending",
                            "level": level,
                            "rows": None,
                            "cols": None,
                            "cells": {},
                            "buffer": [""],  # 표 레벨에 낀 비정상 텍스트 대비
                        })
                    # 표가 아닌 컨트롤(그림 등)은 텍스트에 영향 없으므로 무시

                elif tag_id == HWPTAG_TABLE and stack[-1]["type"] == "table_pending":
                    rows, cols = _hwp_parse_table_dims(payload)
                    stack[-1].update(type="table", rows=rows, cols=cols)

                elif tag_id == HWPTAG_LIST_HEADER:
                    addr = _hwp_parse_list_header_addr(payload)

                    if stack[-1]["type"] == "table":
                        stack.append({
                            "type": "cell",
                            "level": level,
                            "addr": (addr["row"], addr["col"]),
                            "rowspan": max(1, addr["rowspan"]),
                            "colspan": max(1, addr["colspan"]),
                            "buffer": [""],
                        })
                    else:
                        # 표 밖의 리스트(머리말/꼬리말/각주/텍스트박스 등)
                        # - 일반 텍스트로 취급
                        stack.append({
                            "type": "list",
                            "level": level,
                            "buffer": [""],
                        })

                elif tag_id == HWPTAG_PARA_HEADER:
                    stack[-1]["buffer"].append("")

                elif tag_id == HWPTAG_PARA_TEXT:
                    piece = _hwp_decode_para_text(payload)
                    if not stack[-1]["buffer"]:
                        stack[-1]["buffer"].append("")
                    stack[-1]["buffer"][-1] += piece

        while len(stack) > 1:
            finished = stack.pop()
            if finished["type"] == "table_pending":
                # [진단용] 여기서 닫히는 건 새 태그 때문이 아니라 문서/
                # 섹션이 그냥 끝나버린 경우다.
                finished["_closed_by"] = "문서(섹션) 끝(EOF)"
            tables_total, tables_failed, tables_partial = _finalize_frame(
                finished,
                stack[-1],
                tables_total,
                tables_failed,
                tables_partial,
            )

        text = "\n\n".join(p for p in root["buffer"] if p.strip())

        # 방어: 어떤 경로로든 단독 서로게이트가 남아있으면 UTF-8 인코딩이
        # 깨지므로 여기서 한 번 더 제거한다.
        text = re.sub(r"[\ud800-\udfff]", "", text)

        if not text.strip():
            raise HwpParseError("raw 파서 결과가 비어 있음")

        result = TableParseResult(
            # [수정 1] 완전 실패(<70% 채움) 표가 없으면 success=True로 유지.
            # 부분 복원은 더 이상 실패로 집계하지 않고 tables_partial에 따로 남긴다.
            success=(tables_total == 0 or tables_failed == 0),
            tables_total=tables_total,
            tables_failed=tables_failed,
            tables_partial=tables_partial,
        )

        return text, result


# [수정 4 - 셀 내부 구조 유지] ○/◦/□/■/※/•/▶/▷, (1)(2)식 번호, 가./나./다.
# 같은 한글 순번, ①~⑳ 원문자로 시작하는 줄을 Markdown 리스트("- ...")로
# 바꾼다. 구조가 살아있어야 청킹 후에도 항목이 뭉개지지 않는다.
# 주의: [가-힣]\. 은 "답." 처럼 한 글자 + 마침표로 끝나는 일반 문장도
# 리스트 항목으로 오인할 수 있다 (회귀 가능성 - 응답 하단 요약 참고).
_CELL_STRUCTURE_MARKER = re.compile(
    r"^(?:[○◦□■※•▶▷]|\([0-9]+\)|[0-9]+\.|[가-힣]\.|[①-⑳])\s*"
)


def _join_cell_paragraphs(paragraphs: list) -> str:
    """셀 안 문단 목록을, 구조 마커로 시작하는 줄은 Markdown 리스트로 바꿔 합친다."""

    lines = []

    for paragraph in paragraphs:
        stripped = paragraph.strip()

        if not stripped:
            continue

        match = _CELL_STRUCTURE_MARKER.match(stripped)

        if match:
            content = stripped[match.end() :].strip()
            lines.append(f"- {content}" if content else "-")
        else:
            lines.append(stripped)

    return "\n".join(lines)


def _finalize_frame(
    frame: dict,
    parent: dict,
    tables_total: int,
    tables_failed: int,
    tables_partial: int = 0,
):
    """스택에서 닫히는 frame을 부모 컨텍스트에 반영한다."""

    if frame["type"] == "cell":
        # [수정 4] 단순 "\n".join(...) 대신 구조 마커를 인식하는 join 사용
        text = _join_cell_paragraphs(frame["buffer"])

        if parent["type"] == "table":
            parent["cells"][frame["addr"]] = {
                "text": text,
                "colspan": frame["colspan"],
                "rowspan": frame["rowspan"],
            }
        else:
            parent["buffer"].append(text)

    elif frame["type"] in ("table", "table_pending"):
        tables_total += 1
        # [수정 1] bool 대신 "success"/"partial"/"failed" 3단계 상태를 받는다.
        rendered, status = _render_table(frame)

        if status == "failed":
            tables_failed += 1
        elif status == "partial":
            tables_partial += 1

        parent["buffer"].append(rendered)

        # 표 레벨에 직접 낀 비정상 텍스트(칸이 아닌 문단)는 버리지 않고 뒤에 붙인다
        stray_text = "\n".join(p for p in frame.get("buffer", []) if p.strip())
        if stray_text:
            parent["buffer"].append(stray_text)

    elif frame["type"] == "list":
        text = "\n".join(p for p in frame["buffer"] if p.strip())
        parent["buffer"].append(text)

    return tables_total, tables_failed, tables_partial


def _render_table(frame: dict) -> tuple[str, str]:
    """
    표 frame을 문자열로 렌더링한다.
    2열 표(첫 열이 짧으면)는 key=value, 그 외는 Markdown 표 형태로.

    [수정 1 - 표 성공 판정 완화]
    셀 하나만 비어도 통째로 실패 처리하던 기존 방식은 실제로는 대부분
    채워진 표까지 실패로 깎아내렸다. 채워진 비율(fill_ratio) 기준으로
    success(>=95%) / partial(70~95%) / failed(<70%) 3단계로 나누고,
    실패여도 채워진 내용은 '-'로만 빈 칸을 메워 최대한 유지한다.
    반환 타입이 bool -> str(status)로 바뀌어 호출부(_finalize_frame)도
    함께 수정했다.
    """

    rows, cols = frame.get("rows"), frame.get("cols")

    if not rows or not cols:
        # [진단용] 실패 사유를 구분해서 남긴다 - 재실행 없이 기존
        # cleaned_texts/*.txt를 grep만 해도 "표 레코드 누락"으로 인한
        # 실패가 몇 건인지 바로 셀 수 있다. 무엇 때문에 table_pending이
        # TABLE 레코드를 못 받고 닫혔는지(_closed_by)도 함께 남긴다.
        closed_by = frame.get("_closed_by", "알 수 없음")
        return f"[표 복원 실패: TABLE 레코드 누락 (닫힌 원인: {closed_by})]", "failed"

    grid = [[None] * cols for _ in range(rows)]

    try:
        for (row0, col0), cell in frame["cells"].items():
            for r in range(row0, min(row0 + cell["rowspan"], rows)):
                for c in range(col0, min(col0 + cell["colspan"], cols)):
                    grid[r][c] = cell["text"]
    except Exception as e:  # noqa: BLE001
        # [버그 수정] 원본 코드는 RuntimeError만 잡았는데, 여기서 실제로
        # 날 수 있는 오류(IndexError 등)는 RuntimeError가 아니라서 사실상
        # 한 번도 안 잡혔다. 사용자가 실제 100건 실행에서 동일 패턴의
        # 버그(NotOleFileError 미포착)를 발견해 함께 넓혀 잡는다.
        # [진단용] 실제 예외 종류/메시지를 태그에 남긴다.
        return f"[표 복원 실패: 셀 주소 처리 오류 - {type(e).__name__}: {e}]", "failed"

    total_cells = rows * cols
    filled_cells = sum(1 for row in grid for v in row if v is not None)
    fill_ratio = filled_cells / total_cells if total_cells else 0.0

    # [수정 3 - 병합셀 렌더링 개선] fill_ratio/키-값 판정용 grid는 기존처럼
    # 병합 범위 전체에 텍스트를 복제한 상태를 그대로 쓴다(주소 처리 로직은
    # 건드리지 않음). 화면에 보여줄 display_grid만 별도로 만들어, 병합된
    # 칸 중 origin이 아닌 칸은 빈 문자열로 비운다 - 같은 텍스트가 셀마다
    # 반복 출력되는 것을 막기 위함(진짜 병합 표현은 Markdown 표가 지원하지
    # 않으므로, 빈 칸으로 이어짐을 표시하는 절충안).
    display_grid = [row[:] for row in grid]
    for (row0, col0), cell in frame["cells"].items():
        if cell["rowspan"] <= 1 and cell["colspan"] <= 1:
            continue
        for r in range(row0, min(row0 + cell["rowspan"], rows)):
            for c in range(col0, min(col0 + cell["colspan"], cols)):
                if (r, c) != (row0, col0):
                    display_grid[r][c] = ""

    filled_display = [
        [v if v is not None else "-" for v in row] for row in display_grid
    ]

    # [표현 이원화] 표 형태에 따라 생성(LLM)용 렌더링 방식을 분기한다.
    # - key-value(2열, 짧은 키): 이미 "키 값" 한 줄이라 그대로도 안전
    # - 병합 셀 있음: Markdown 표는 병합을 표현 못 해서(빈 칸으로만 처리)
    #   헤더-값 결합이 애매해짐 -> 행 단위로 "헤더: 값"을 명시하는
    #   row-block 형태로 렌더링
    # - 그 외(단순 격자): 기존처럼 Markdown 표
    # 이 함수가 반환하는 텍스트는 "생성용" 원본이고, 검색/임베딩용 평문은
    # 이후 strip_table_markup()이 이 결과에서 마크업만 제거해 만든다
    # (렌더링을 두 번 하지 않고 한 번의 결과에서 파생시켜 정합성을 보장).
    has_merged_cells = any(
        cell["rowspan"] > 1 or cell["colspan"] > 1 for cell in frame["cells"].values()
    )

    if cols == 2 and _is_keyvalue_table(grid):
        rendered = _render_keyvalue(filled_display)
    elif has_merged_cells:
        # [버그 수정] display_grid는 Markdown 표용으로 병합된 칸 중
        # origin이 아닌 칸을 빈 문자열로 비워둔다(중복 출력 방지 목적).
        # row-block은 "그 행 하나만 봐도 의미가 통해야" 하므로 정반대로,
        # 병합 값이 spanned된 모든 행에 그대로 반복돼야 한다. 그래서
        # 병합을 죽이지 않은 grid(셀 population 단계에서 이미 rowspan/
        # colspan 범위 전체에 텍스트가 복제돼 있음)를 그대로 쓴다.
        filled_full = [[v if v is not None else "-" for v in row] for row in grid]
        rendered = _render_row_block(filled_full)
    else:
        rendered = _render_matrix(filled_display)

    if fill_ratio >= TABLE_FULL_SUCCESS_RATIO:
        return rendered, "success"

    if fill_ratio >= TABLE_PARTIAL_MIN_RATIO:
        return f"[표 부분 복원: 채움비율 {fill_ratio:.0%}]\n" + rendered, "partial"

    # [진단용] 실제 fill_ratio 수치를 태그에 남긴다. 0%대(=거의 다 비어
    # 있는 표, table_pending 프레임 자체가 잘못 열렸을 가능성)인지
    # 60%대(=진짜 아깝게 임계값 미달)인지 구분해서 볼 수 있다.
    return f"[표 복원 실패: 채움비율 {fill_ratio:.0%}]\n" + rendered, "failed"


def _is_keyvalue_table(grid: list) -> bool:
    keys = [row[0] for row in grid if row[0]]
    if not keys:
        return False
    avg_len = sum(len(k) for k in keys) / len(keys)
    return avg_len <= KEYVALUE_MAX_KEY_LENGTH


def _render_keyvalue(grid: list) -> str:
    # [표 마커 완전 제거] "키 = 값"의 "="도 표에서 왔다는 걸 드러내는
    # 마커로 보고 없앤다. 실험 결과 셀 구분자까지 없앤 순수 공백 구분
    # 텍스트가 성능이 더 좋다고 판단되어, 그냥 공백으로 이어붙인다.
    return "\n".join(f"{row[0].strip()} {row[1].strip()}" for row in grid)


def _render_row_block(grid: list) -> str:
    """병합 셀이 있는 표를 '[행 N] 헤더1: 값1 헤더2: 값2 ...' 형태로
    렌더링한다.

    [표현 이원화] Markdown 표는 rowspan/colspan을 표현할 방법이 없어서,
    병합된 칸을 빈 칸으로 남기면(_render_table의 display_grid 처리) 그
    행에서 어떤 값이 어떤 헤더에 속하는지가 다시 애매해진다. 병합 셀이
    있는 표만 이 형태로 바꿔서, 행마다 헤더-값 쌍을 명시적으로 풀어쓴다.
    한 행을 한 줄로 유지해 다른 행과 섞이지 않게 한다.

    시간복잡도: O(행 수 * 열 수)
    """

    header = [cell.strip() for cell in grid[0]]
    lines = []

    for row_idx, row in enumerate(grid[1:], start=1):
        pairs = [
            f"{header[c]}: {row[c].strip()}"
            for c in range(len(header))
            if row[c].strip()
        ]
        if pairs:
            lines.append(f"[행 {row_idx}] " + " ".join(pairs))

    return "\n".join(lines)


def _render_matrix(grid: list) -> str:
    """
    [수정 2 - Markdown 표 출력] 기존 "헤더=값 · 헤더=값" 형태는 사람이
    읽기엔 괜찮지만 임베딩 시 표 구조가 뭉개진다. 청크 검색 품질을 위해
    표준 Markdown 표(|헤더|...|, |---|...|)로 바꾼다. 셀 안에 이미 줄바꿈이
    있으면(예: 수정 4의 Markdown 리스트) Markdown 표 한 줄이 깨지므로
    <br>로 치환한다. 파이프 문자(|)도 이스케이프한다.
    """

    def _escape_cell(value) -> str:
        text = str(value).strip()
        text = text.replace("|", "\\|")
        text = text.replace("\n", "<br>")
        return text

    header = [_escape_cell(cell) for cell in grid[0]]
    lines = [
        "|" + "|".join(header) + "|",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]

    for row in grid[1:]:
        cells = [_escape_cell(row[c]) for c in range(len(header))]
        lines.append("|" + "|".join(cells) + "|")

    return "\n".join(lines)


# ============================================================
# 10. HWP 통합 디스패처
# ============================================================


def extract_hwp_document(path: Path) -> ExtractionResult:
    """
    HWP_EXTRACTION_METHODS 순서대로 시도한다.
    hwp_raw는 한글 비율 검증만 통과하면 채택한다 - 표 일부가 실패해도
    표를 아예 포기하는 hwp5txt/libreoffice보다 낫다고 판단하기 때문.
    (표 전부/일부 성공 여부는 table_parse_success로 별도 기록)

    성공한 문서라도, 먼저 시도했다가 실패한 방법이 있으면 그 에러를
    attempted_errors에 남긴다. hwp_raw가 조용히 실패하고 hwp5txt로
    넘어가는 경우(표 손실)를 추적하기 위함.
    """

    errors = {}

    for i, method in enumerate(HWP_EXTRACTION_METHODS):
        fallback_used = i > 0

        try:
            if method == "hwp_raw":
                text, table_result = extract_text_hwp_raw_structured(path)

                if not _check_hangul(text):
                    raise HwpParseError("한글 비율 미달")

                return ExtractionResult(
                    text=text,
                    extractor="hwp_raw",
                    table_parse_success=table_result.success,
                    tables_total=table_result.tables_total,
                    tables_failed=table_result.tables_failed,
                    error_reason=None,
                    fallback_used=fallback_used,
                    attempted_errors=dict(errors),
                    tables_partial=table_result.tables_partial,  # [수정 1]
                )

            elif method == "hwp5txt":
                text = extract_text_hwp5txt(path)

                if not _check_hangul(text):
                    raise HwpParseError("한글 비율 미달")

                return ExtractionResult(
                    text=text,
                    extractor="hwp5txt",
                    table_parse_success=False,
                    tables_total=0,
                    tables_failed=0,
                    error_reason=None,
                    fallback_used=fallback_used,
                    attempted_errors=dict(errors),
                )

            elif method == "libreoffice":
                text = extract_text_libreoffice(path)

                if not _check_hangul(text):
                    raise HwpParseError("한글 비율 미달")

                return ExtractionResult(
                    text=text,
                    extractor="libreoffice",
                    table_parse_success=False,
                    tables_total=0,
                    tables_failed=0,
                    error_reason=None,
                    fallback_used=fallback_used,
                    attempted_errors=dict(errors),
                )

            else:
                raise ValueError(f"알 수 없는 추출 방법: {method}")

        except Exception as e:  # noqa: BLE001
            # [버그 수정] HwpParseError(Exception 상속)와 olefile의
            # NotOleFileError(OSError 상속) 모두 RuntimeError가 아니라서
            # 기존 "except RuntimeError"로는 못 잡았다. 그러면 hwp_raw가
            # OLE2가 아닌 파일(HWPX 오분류, 손상 파일 등)에서 죽었을 때
            # hwp5txt/libreoffice 폴백을 타지 못하고 예외가 extract_hwp_document
            # 밖으로 그대로 새어나가 파이프라인 전체가 멈췄다(사용자가 100건
            # 실행 중 NotOleFileError로 직접 재현). 폴백 체인이 실제로
            # 작동하려면 여기서 넓게 잡아야 한다.
            errors[method] = str(e)

    return ExtractionResult(
        text="",
        extractor=None,
        table_parse_success=False,
        tables_total=0,
        tables_failed=0,
        error_reason="; ".join(f"{k}={v}" for k, v in errors.items()),
        fallback_used=True,
        attempted_errors=dict(errors),
    )


