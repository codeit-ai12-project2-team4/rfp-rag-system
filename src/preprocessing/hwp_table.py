"""HWP 표 구조를 살려서 뽑는다. hwp.py 의 문단 전용 추출기와 나란히 둔다.

hwp.py 는 HWPTAG_PARA_TEXT(67) 만 골라 이어 붙인다. 표 안의 글자도 결국
문단 레코드라 딸려 나오지만, **행과 열 관계가 사라진다.**

    평가항목 / 배점 / 세부기준 / 기술능력평가 / 80 / …

여기서는 표 레코드를 같이 읽어 행을 복원한다.

    [평가항목별 배점] 평가항목=기술능력평가 · 배점=80 · 세부기준=…

RFP 는 글자의 60~80%가 표 안에 있어서 이 차이가 크다.

## 레코드가 어떻게 생겼나

    PARA_HEADER(0)
      PARA_TEXT(1)          본문 글자
      CTRL_HEADER(1)        payload[:4][::-1] == b"tbl " 이면 표
        TABLE(2)            offset 4: UINT16 행수, offset 6: UINT16 열수
        LIST_HEADER(2)      칸 하나의 시작
        PARA_HEADER(2)      칸 안 문단 (LIST_HEADER 와 형제다)
          PARA_TEXT(3)      칸 안 글자
    PARA_HEADER(0)          ← 여기서 표가 닫힌다

level 값은 실제 문서에서 확인한 것이다. 표 안의 표는 level 4에서 다시 시작한다.

## 칸의 위치는 어디에 있나

LIST_HEADER 페이로드를 UINT16 으로 늘어놓고 실제 문서로 확인한 결과다.

    바이트  0-3   INT32   문단 수
            4-7   UINT32  속성
            8-9   UINT16  열 주소   (0부터)
           10-11  UINT16  행 주소
           12-13  UINT16  열 병합 수
           14-15  UINT16  행 병합 수

검증 — 17행x4열 표의 첫 두 칸이 (열0,행0,열병합2) (열2,행0,열병합2) 로 나와
합이 4열이 되고, 12행x19열 표의 첫 행이 (0,0,1) (1,0,6) (7,0,12) 로 19열이 된다.

이 값을 읽으면 **병합이 있어도 격자를 복원할 수 있다.** 칸을 순서대로
열 수만큼 끊는 방식은 병합 하나에 전부 어긋난다.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import olefile

from preprocessing.hwp import HwpParseError, _decode_para_text

TAG_PARA_TEXT = 67
TAG_CTRL_HEADER = 71
TAG_LIST_HEADER = 72
TAG_TABLE = 77

# 본문에 한글이 이 비율보다 적으면 글자가 아닌 걸 읽은 것으로 본다
MIN_HANGUL_RATIO = 0.15


def read_records(path):
    """BodyText 섹션들을 순서대로 훑어 (tag_id, level, payload) 를 돌려준다.

    hwp.py 의 extract_hwp_text 와 같은 경로로 스트림에 접근한다.
    다른 건 level 을 같이 준다는 것뿐이다.

    **파일을 그냥 open() 해서 읽으면 안 된다.** HWP 5.0 은 OLE 복합 문서이고
    레코드는 BodyText/SectionN 스트림 안에 zlib 로 압축돼 있다.

    섹션이 바뀌면 level 이 0부터 다시 시작하므로 (None, -1, b"") 를 끼워 준다.
    """
    path = Path(path)
    if not olefile.isOleFile(str(path)):
        raise HwpParseError(f"OLE 파일이 아님(HWPX이거나 손상 가능): {path.name}")

    ole = olefile.OleFileIO(str(path))
    try:
        header = ole.openstream("FileHeader").read()
        if header[:32].rstrip(b"\x00") != b"HWP Document File":
            raise HwpParseError(f"HWP 5.x 시그니처 아님: {path.name}")
        flags = struct.unpack_from("<I", header, 36)[0]
        compressed = bool(flags & 0x01)
        if flags & 0x02:
            raise HwpParseError(f"암호화된 문서: {path.name}")

        sections = sorted(
            (d for d in ole.listdir() if d[0] == "BodyText"),
            key=lambda d: int(d[1].replace("Section", "")),
        )
        if not sections:
            raise HwpParseError(f"BodyText 섹션 없음: {path.name}")

        for entry in sections:
            raw = ole.openstream("/".join(entry)).read()
            if compressed:
                raw = zlib.decompress(raw, -15)

            yield None, -1, b""  # 섹션 경계

            i = 0
            while i + 4 <= len(raw):
                (head,) = struct.unpack_from("<I", raw, i)
                tag_id = head & 0x3FF  # 하위 10비트
                level = (head >> 10) & 0x3FF  # 중간 10비트
                size = (head >> 20) & 0xFFF  # 상위 12비트
                i += 4
                if size == 0xFFF:  # 확장 크기
                    (size,) = struct.unpack_from("<I", raw, i)
                    i += 4
                yield tag_id, level, raw[i : i + size]
                i += size
    finally:
        ole.close()


# --- 표를 글자로 펴기 -----------------------------------------------------


def parse_cell_header(payload):
    """LIST_HEADER 에서 (열, 행, 열병합, 행병합) 을 읽는다."""
    if len(payload) < 16:
        return None
    col, row, colspan, rowspan = struct.unpack_from("<4H", payload, 8)
    return {
        "col": col,
        "row": row,
        "colspan": max(1, colspan),
        "rowspan": max(1, rowspan),
    }


def build_grid(table):
    """칸들을 행×열 격자에 놓는다. 병합된 칸은 덮는 자리마다 값을 되풀이한다.

    되풀이하는 이유 — 머리글 하나가 6열을 덮고 있으면, 아래 데이터 행마다
    '머리글=값' 짝을 만들려면 그 6열 전부에 머리글이 있어야 한다.
    """
    rows, cols = table["rows"], table["cols"]
    if rows <= 0 or cols <= 0 or rows * cols > 20000:
        return None

    grid = [["" for _ in range(cols)] for _ in range(rows)]
    placed = 0
    for cell in table["cells"]:
        if cell["pos"] is None:
            return None
        text = "\n".join(cell["lines"]).strip()
        pos = cell["pos"]
        if pos["row"] >= rows or pos["col"] >= cols:
            return None
        for dr in range(pos["rowspan"]):
            for dc in range(pos["colspan"]):
                r, c = pos["row"] + dr, pos["col"] + dc
                if r < rows and c < cols:
                    grid[r][c] = text
        placed += 1
    return grid if placed else None


def render_table(table, caption="표"):
    """표 하나를 검색하기 좋은 문자열로.

    행렬형(헤더 행 + 데이터 행)은 **행마다 헤더를 붙인다.** 한 줄이 스스로
    완결돼야 청킹에서 잘려도 답이 살아남는다.

        [평가배점] 평가항목=기술능력평가 · 배점=80

    2열 키-값형(요구사항 상세, 사업 개요)은 원래 모양이 낫다.

        요구사항 고유번호: SFR-002

    병합이 있어 칸 수가 행×열과 안 맞으면 격자 복원을 포기한다.
    억지로 맞추면 열이 어긋나서 **틀린 값**을 만들기 때문이다.
    """
    flat = ["\n".join(c["lines"]).strip() for c in table["cells"]]
    if not any(flat):
        return ""

    grid = build_grid(table)
    if grid is None:  # 칸 위치를 못 읽었다 — 값만 늘어놓는다
        table["fallback"] = True
        body = "\n".join(f"- {c}" for c in flat if c)
        return f"[{caption} · {table['rows']}x{table['cols']} 격자복원실패]\n{body}"

    cols = table["cols"]

    # 1열 표는 레이아웃용 상자다. 열 관계가 없으므로 칸 글자를 그대로 내보낸다.
    if cols == 1:
        return "\n".join(dict.fromkeys(c for row in grid for c in row if c))

    if cols == 2 and all(len(r[0]) <= 20 for r in grid if r[0]):
        lines = [f"{k}: {v}" for k, v in grid if k or v]
        return "\n".join(dict.fromkeys(lines))

    header, *data = grid
    # (제목 행 재시도 패치는 삭제한다. 이득보다 손해가 컸다.)

    lines = []
    head_line = " | ".join(dict.fromkeys(h for h in header if h))
    if head_line:
        lines.append(f"[{caption}] {head_line}")

    for row in data:
        pairs = [f"{h}={v}" for h, v in zip(header, row) if v and h and h != v]
        if pairs:
            lines.append(f"[{caption}] " + " · ".join(pairs))

    # 짝을 하나도 못 만들었으면 격자를 그대로 내보낸다. 글자를 버리지 않는다.
    if not lines:
        return "\n".join(
            dict.fromkeys(" | ".join(x for x in row if x) for row in grid if any(row))
        )
    return "\n".join(dict.fromkeys(lines))


def _caption_of(parts):
    """표 바로 앞 줄을 캡션으로 쓴다. '<표 3> 평가항목별 배점' 같은 것.

    앞 표가 펴진 줄(`[…] 항목=값`)은 건너뛴다. 그걸 캡션으로 쓰면
    캡션이 표 내용으로 채워져서 청크가 지저분해진다.
    """
    for part in reversed(parts[-4:]):
        line = part.strip().splitlines()[-1] if part.strip() else ""
        if not line or line.startswith("[") or "=" in line or " | " in line:
            continue
        if 2 < len(line) < 60:
            return line
    return "표"


# --- 본문 + 표 ------------------------------------------------------------


def extract_hwp_tables(path, check_hangul=True):
    """본문은 그대로, 표는 구조를 살려서 하나의 문자열로."""
    text, _ = extract_with_report(path)
    if check_hangul:
        _check_hangul(text, path)
    return text


def extract_with_report(path, collect=None):
    """(본문, 통계) 를 돌려준다. 표를 얼마나 복원했는지 보고 싶을 때.

    collect 에 리스트를 주면 표마다 격자와 렌더 결과를 담아 준다.
    눈으로 대조할 때 쓴다 (scripts/eval_tables.py --dump).
    """
    parts: list[str] = []
    stack: list[dict] = []
    # slots/filled 는 격자를 얼마나 채웠는지 재는 데 쓴다 (scripts/eval_tables.py)
    report = {
        "tables": 0,
        "fallback": 0,
        "empty": 0,
        "cells": 0,
        "sections": 0,
        "slots": 0,
        "filled": 0,
        "blank_cells": 0,
    }

    def close_top():
        done = stack.pop()
        caption = _caption_of(parts)
        report["blank_cells"] += sum(
            1 for c in done["cells"] if not "\n".join(c["lines"]).strip()
        )
        rendered = render_table(done, caption=caption)
        grid = None
        if done.get("fallback"):
            report["fallback"] += 1
        else:
            grid = build_grid(done)
            if grid:
                report["slots"] += done["rows"] * done["cols"]
                report["filled"] += sum(1 for row in grid for cell in row if cell)
        if collect is not None:
            collect.append({
                "caption": caption,
                "rows": done["rows"],
                "cols": done["cols"],
                "cells": len(done["cells"]),
                "grid": grid,
                "rendered": rendered,
            })
        if not rendered:
            report["empty"] += 1
            return
        if stack and stack[-1]["cells"]:
            stack[-1]["cells"][-1]["lines"].append(rendered)  # 표 안의 표
        else:
            parts.append(rendered)

    for tag, level, payload in read_records(path):
        if tag is None:  # 섹션 경계 — 열린 표를 닫는다
            report["sections"] += 1
            while stack:
                close_top()
            continue

        while stack and level < stack[-1]["level"]:
            close_top()

        if tag == TAG_TABLE and len(payload) >= 8:
            rows, cols = struct.unpack_from("<HH", payload, 4)
            stack.append({"level": level, "rows": rows, "cols": cols, "cells": []})
            report["tables"] += 1

        elif tag == TAG_LIST_HEADER and stack and level == stack[-1]["level"]:
            stack[-1]["cells"].append({"pos": parse_cell_header(payload), "lines": []})
            report["cells"] += 1

        elif tag == TAG_PARA_TEXT:
            text = _decode_para_text(payload).strip()
            if not text:
                continue
            if stack and level > stack[-1]["level"] and stack[-1]["cells"]:
                stack[-1]["cells"][-1]["lines"].append(text)
            else:
                parts.append(text)

    while stack:
        close_top()

    return "\n".join(parts), report


def _check_hangul(text, path):
    """한글이 거의 없으면 글자가 아닌 걸 읽은 것이다.

    스트림 접근이나 압축 해제가 틀리면 이진 데이터를 UTF-16LE 로 읽게 되는데,
    무작위 바이트쌍은 절반쯤이 한자 영역에 떨어져서 '한자가 섞였다'로 보인다.
    조용히 넘어가면 안 되는 실패라 여기서 막는다.
    """
    sample = text[:5000]
    if not sample:
        return
    hangul = sum(1 for ch in sample if "가" <= ch <= "힣")
    if hangul / len(sample) < MIN_HANGUL_RATIO:
        raise HwpParseError(
            f"본문에 한글이 거의 없습니다 ({hangul}/{len(sample)}자) — {Path(path).name}\n"
            "스트림 접근이나 압축 해제를 확인하세요."
        )
