"""텍스트 정제. 노이즈 제거와 표 마크업 처리.

원본: `preprocessing/pipeline.py` (전처리팀). 분할 경위는 `src/preprocessing/README.md`.
"""

import re

import pandas as pd

from preprocessing.rfp.common import (
    PATTERN_CONTROL_CHAR,
    PATTERN_EMPTY_PAREN,
    PATTERN_HWP_TAG_RESIDUE,
    PATTERN_LONG_NEWLINE,
    PATTERN_LONG_SPACE,
    PATTERN_MOJIBAKE,
    PATTERN_PAGE_NUMBER_DASH,
    PATTERN_PLACEHOLDER,
    _is_parenthesized_mojibake_match,
    get_boilerplate_lines,
)

# ============================================================
# 13. 텍스트 정제
# ============================================================


def clean_text_verbose(
    text: str,
    remove_angle_brackets: bool = False,
    remove_bullet_dot: bool = False,
    boilerplate_lines: set | None = None,
) -> tuple[str, list[str]]:
    """
    Args:
        text: 정제할 원문.
        remove_angle_brackets: <,> 문자 제거 여부.
        remove_bullet_dot: • 문자 제거 여부.
        boilerplate_lines: 제거할 boilerplate 라인 집합. None이면 기본값 사용.

    Returns:
        tuple[str, list[str]]: (정제된 텍스트, 적용된 정제 작업 설명 목록).
    """

    text = re.sub(r"[\ud800-\udfff]", "", text)

    if boilerplate_lines is None:
        boilerplate_lines = get_boilerplate_lines()

    notes = []
    page_number_removed = 0
    boilerplate_removed = 0

    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        if PATTERN_PAGE_NUMBER_DASH.match(stripped):
            page_number_removed += 1
            continue

        normalized = re.sub(r"\s+", " ", stripped)

        if boilerplate_lines and normalized in boilerplate_lines:
            boilerplate_removed += 1
            continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    if page_number_removed:
        notes.append(f"페이지번호 라인 {page_number_removed}개 제거")

    if boilerplate_removed:
        notes.append(f"boilerplate 라인 {boilerplate_removed}개 제거")

    tag_count = len(PATTERN_HWP_TAG_RESIDUE.findall(text))
    text = PATTERN_HWP_TAG_RESIDUE.sub("", text)
    if tag_count:
        notes.append(f"<표>/<그림> 태그 잔재 {tag_count}개 제거")

    placeholder_count = len(PATTERN_PLACEHOLDER.findall(text))
    text = PATTERN_PLACEHOLDER.sub("", text)
    if placeholder_count:
        notes.append(f"플레이스홀더(ㅇㅇㅇ 등) {placeholder_count}개 제거")

    empty_paren_count = len(PATTERN_EMPTY_PAREN.findall(text))
    text = PATTERN_EMPTY_PAREN.sub("", text)
    if empty_paren_count:
        notes.append(f"빈 괄호 {empty_paren_count}개 제거")

    control_count = len(PATTERN_CONTROL_CHAR.findall(text))
    text = PATTERN_CONTROL_CHAR.sub("", text)
    if control_count:
        notes.append(f"제어문자 {control_count}개 제거")

    # [수정 6 - 인코딩 노이즈 제거] 괄호로 둘러싸인 런(정상 한자 병기)은
    # 보존하고, 그 외 런만 통째로 제거한다.
    mojibake_count = sum(
        1
        for mt in PATTERN_MOJIBAKE.finditer(text)
        if not _is_parenthesized_mojibake_match(mt)
    )
    text = PATTERN_MOJIBAKE.sub(
        lambda mt: mt.group(0) if _is_parenthesized_mojibake_match(mt) else "",
        text,
    )
    if mojibake_count:
        notes.append(f"mojibake 추정 문자 묶음 {mojibake_count}개 제거")

    if remove_angle_brackets:
        angle_count = len(re.findall(r"[<>]", text))
        text = re.sub(r"[<>]", "", text)
        if angle_count:
            notes.append(f"<,> 문자 {angle_count}개 제거")

    if remove_bullet_dot:
        bullet_count = text.count("•")
        text = text.replace("•", "")
        if bullet_count:
            notes.append(f"• 문자 {bullet_count}개 제거")

    text = PATTERN_LONG_SPACE.sub(" ", text)
    text = PATTERN_LONG_NEWLINE.sub("\n\n", text)

    return text.strip(), notes


def clean_text(
    text: str,
    remove_angle_brackets: bool = False,
    remove_bullet_dot: bool = False,
    boilerplate_lines: set | None = None,
) -> str:
    """
    구조 마커(숫자_점, 가., ○, □, ※, ◦, (1), 로마숫자, •)는 절대 건드리지 않는다.
    내부적으로 clean_text_verbose를 감싸며, 작업 내역(warnings)만 버린다.

    Args:
        text: 정제할 원문.
        remove_angle_brackets: <,> 문자 제거 여부.
        remove_bullet_dot: • 문자 제거 여부.
        boilerplate_lines: 제거할 boilerplate 라인 집합. None이면 기본값 사용.

    Returns:
        str: 정제된 텍스트.
    """

    cleaned, _ = clean_text_verbose(
        text,
        remove_angle_brackets=remove_angle_brackets,
        remove_bullet_dot=remove_bullet_dot,
        boilerplate_lines=boilerplate_lines,
    )
    return cleaned


# ============================================================
# 13-1. 표 구조 제거 (Markdown 표 -> 셀 구분자만 남긴 평문)
# ============================================================
#
# 실험 결과 표를 Markdown 그리드(|헤더|---|값|)로 살려서 임베딩하는 것보다
# 격자(테두리 파이프, 구분행)를 걷어내고 셀 값만 구분자로 이어붙인 평문이
# 검색 성능이 더 좋다고 판단되어 도입. _render_table/_render_matrix는
# 그대로 두고(성공/부분/실패 판정과 report 집계는 fill_ratio 기준으로
# 여전히 정확하게 동작), 다 만들어진 텍스트를 마지막에 한 번 더 정리하는
# 방식이라 표 판정 로직과는 완전히 분리되어 있다.

PATTERN_TABLE_ESCAPED_MARKUP = re.compile(r"\\([|-])")  # 중첩 표의 \| \- 를 푼다
# [수정] 참고 코드의 원래 패턴은 "[표 파싱...]"을 찾는데, 실제 이 파이프라인이
# 남기는 진단 태그는 "[표 복원 실패: ...]"/"[표 부분 복원: ...]"이라 문자열이
# 다르다. 그대로 쓰면 아무것도 안 지워지므로 실제 태그 문자열에 맞춰 고쳤다.
PATTERN_TABLE_DEBUG_TAG = re.compile(r"\[표 (?:복원 실패|부분 복원)[^\]]{0,200}\]")
PATTERN_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")  # |셀|셀|  (구분행/빈행 포함)
PATTERN_TABLE_SEPARATOR_ROW = re.compile(r"^\s*\|(\s*:?-{2,}:?\s*\|)+\s*$")  # |---|---|
PATTERN_TABLE_EMPTY_ROW = re.compile(r"^\s*\|(\s*\|)*\s*$")  # |||
PATTERN_TABLE_BLANK_RUN = re.compile(r"\n{3,}")
PATTERN_TABLE_CELL_SPLIT = re.compile(
    r"(?<!\\)\|"
)  # 이스케이프 안 된 |만 셀 구분자로 인식
PATTERN_ROW_BLOCK_LINE = re.compile(
    r"\[행 \d+\]\s*"
)  # _render_row_block 결과 태그. ^ 없음: 줄 중간에도 나타날 수 있어 .search()/.sub()로 찾는다. .match() 사용처는 항상 위치 0에서만 보므로 영향 없다.
# 앞이 숫자가 아니거나(헤더-값 구분자) 뒤가 숫자가 아닌 콜론만 고른다.
# 15:00 처럼 양쪽이 숫자인 것만 살아남는다.
PATTERN_HEADER_COLON = re.compile(r"(?<!\d):|:(?!\d)")


def _split_table_row(line: str) -> list[str]:
    """Markdown 표 한 줄을 셀 리스트로 쪼갠다.

    이스케이프된 파이프(`\\|`)는 셀 내용(중첩 표)으로 보고 구분자로 쓰지
    않는다. 셀 단위로 다시 `\\|`, `\\-` 이스케이프를 풀고 `<br>`을 칸 내부
    줄바꿈이었던 자리이므로 공백으로 되돌린다.

    시간복잡도: O(줄 길이)
    """

    inner = re.sub(r"^\s*\|", "", line.strip())
    inner = re.sub(r"\|\s*$", "", inner)

    cells = []
    for raw_cell in PATTERN_TABLE_CELL_SPLIT.split(inner):
        cell = PATTERN_TABLE_ESCAPED_MARKUP.sub(r"\1", raw_cell)
        cell = cell.replace("<br>", " ")
        cells.append(cell.strip())

    return cells


def _flatten_table_block(lines: list[str]) -> list[str]:
    """연속된 Markdown 표 줄 묶음을 '헤더 값 헤더 값 ...' 행 단위 평문으로
    바꾼다.

    [핵심] 기존 방식은 헤더 줄과 값 줄을 각각 독립적으로 공백 구분
    텍스트로 바꿔서, 헤더 "컬럼1 컬럼2 컬럼3"과 값 "1-1 2-1 3-1"이 서로
    다른 줄로 떨어져 나왔다. 청크가 헤더 줄과 값 줄 사이에서 잘리거나
    표가 여러 개 이어지면 어느 값이 어느 헤더의 것인지 위치 추론에만
    의존해야 해서 의미가 끊길 위험이 있었다.

    이 함수는 헤더 행(첫 줄)의 셀과 각 데이터 행의 셀을 같은 열 위치로
    짝지어 "헤더 값" 쌍을 만들고, 한 데이터 행의 모든 쌍을 한 줄에
    이어붙인다. 격자 기호(|, ---)는 사라지지만 헤더-값 결합은 같은 줄
    안에서 유지된다. 행 간 줄바꿈은 그대로 둬서(수정 전과 동일하게) 행
    하나가 다른 행과 한 줄로 섞이지는 않는다.

    시간복잡도: O(행 수 * 열 수)
    """

    data_lines = [
        line
        for line in lines
        if not PATTERN_TABLE_SEPARATOR_ROW.match(line)
        and not PATTERN_TABLE_EMPTY_ROW.match(line)
    ]

    rows = [_split_table_row(line) for line in data_lines]
    rows = [row for row in rows if any(cell for cell in row)]

    if not rows:
        return []

    header, *data_rows = rows

    if not data_rows:
        # 헤더 행만 있고 데이터 행이 없으면(예: 표가 잘려 헤더만 남은 경우)
        # 헤더라도 살려서 정보 손실을 막는다.
        return [" ".join(cell for cell in header if cell)]

    flattened_rows = []

    for row in data_rows:
        pairs = []
        for col_idx, value in enumerate(row):
            if not value:
                continue
            head = header[col_idx] if col_idx < len(header) else ""
            pairs.append(f"{head} {value}".strip() if head else value)

        if pairs:
            flattened_rows.append(" ".join(pairs))

    return flattened_rows


def strip_table_markup(text: str) -> str:
    """Markdown 표 문법을 없애되, 헤더-값 결합은 같은 줄에 유지한 채 순수
    공백 구분 텍스트로 만든다.

    [실험 결과 반영] 격자(테두리 파이프, 구분행)를 걷어내고 셀 값을
    공백으로 이어붙인 평문이 Markdown 그리드보다 검색 성능이 좋다는 것은
    기존 실험으로 확인됨. 다만 헤더 줄과 값 줄을 따로따로 공백 텍스트로
    바꾸면 헤더-값 대응이 줄 위치 추론에만 의존하게 되는 문제가 있어,
    표 블록을 통째로 파싱해 각 데이터 행에 헤더를 인라인으로 묶어주는
    방식으로 바꿨다(`_flatten_table_block` 참고). 값이 어느 헤더에
    속하는지가 텍스트 자체에 남기 때문에 청크가 표 중간에서 잘려도 행
    단위로는 의미가 보존된다.

    이 함수는 `|`로 둘러싸인 표 형태 줄과 `_render_row_block`이 만든
    "[행 N] 헤더: 값 ..." 줄만 건드리므로, 그 외(`|`도 `[행 N]` 태그도
    없는) 계층/불릿 마커(□■○◦※•▶▷), 목차/섹션 제목(Ⅰ.Ⅱ.Ⅲ.), 활동/과업
    번호(Activity N.N.N), 법률/계약 조항(제N조, ①②...)은 전혀 영향을
    받지 않는다. `_render_keyvalue`가 만든 2열 표(이미 "키 값" 형태라
    `|`도 `[행 N]`도 없음)도 손대지 않는다 - 이미 헤더-값이 한 줄에 묶여
    있어 그대로 둬도 안전하다.

    Args:
        text: `_render_table`이 표 형태별로 렌더링을 마친, 생성(LLM)용
            원본 텍스트(`clean_text_verbose`까지 끝난 상태).

    Returns:
        str: 표 마크업을 제거하고, 데이터 행마다 "헤더 값" 쌍을 이어붙인
        검색/임베딩용 평문.
    """

    # 표 진단 태그(실패/부분복원)는 report/warnings에서 이미 집계했으니
    # 본문(임베딩 대상)에는 남길 필요가 없다.
    text = PATTERN_TABLE_DEBUG_TAG.sub("", text)

    lines = text.split("\n")
    output_lines: list[str] = []
    table_buffer: list[str] = []

    def _flush_table_buffer() -> None:
        if table_buffer:
            output_lines.extend(_flatten_table_block(table_buffer))
            table_buffer.clear()

    for line in lines:
        if PATTERN_TABLE_ROW.match(line):
            table_buffer.append(line)
            continue

        _flush_table_buffer()

        if PATTERN_ROW_BLOCK_LINE.search(line):
            # "...[행 N] 헤더1: 값1 [행 M] 헤더2: 값2..." -> "...헤더1 값1 헤더2 값2..."
            # 태그가 줄 시작이 아니라 중간에도 나타날 수 있어(배점 산식
            # 문장 등에 섞여 들어간 사례) sub으로 위치 상관없이 전부
            # 걷어낸다. 헤더-값 구분 콜론도 같이 걷어내되, 앞뒤가 모두
            # 숫자인 콜론(15:00, 13:30 …)은 시각이므로 남긴다.
            content = PATTERN_ROW_BLOCK_LINE.sub("", line)
            content = PATTERN_HEADER_COLON.sub("", content)
            content = re.sub(r"[ \t]{2,}", " ", content).strip()
            output_lines.append(content)
            continue

        # 표 밖(그 줄에 |가 없는 경우)의 <br>은 원래 의도대로 줄바꿈으로 되돌린다.
        output_lines.append(line.replace("<br>", "\n"))

    _flush_table_buffer()

    text = "\n".join(output_lines)

    # 표만 있다가 비어버린 줄(공백만 남은 줄) 제거
    text = re.sub(r"^[ \t]*$", "", text, flags=re.MULTILINE)
    text = PATTERN_TABLE_BLANK_RUN.sub("\n\n", text)

    return text.strip()


PATTERN_TABLE_MARKUP_LEFTOVERS = {
    "잔존_파이프": re.compile(r"\|"),
    "잔존_br": re.compile(r"<br>"),
    "잔존_구분행": re.compile(r"^\s*\|?\s*:?-{2,}", re.MULTILINE),
    "잔존_진단태그": re.compile(r"\[표 (?:복원 실패|부분 복원)"),
    "잔존_행블록태그": re.compile(r"\[행 \d+\]"),
}


def verify_no_table_markup(
    df: pd.DataFrame, text_col: str = "clean_text"
) -> pd.DataFrame:
    """strip_table_markup이 실제로 다 걷어냈는지 QA용으로 검증한다.

    run_pipeline이 반환한 df(또는 write_jsonl에 쓰인 것과 같은 df)를 그대로
    넣으면 된다. text_col을 "clean_text"(문서 단위 df)나 "page_content"
    (청크 단위 df)로 바꿔가며 쓸 수 있다.

    Args:
        df: 검사할 DataFrame. text_col 컬럼과, 있으면 "filename"/"source"
            컬럼을 사용한다.
        text_col: 검사할 텍스트가 담긴 컬럼명.

    Returns:
        pd.DataFrame: 표 마크업이 남아있는 문서만 담은 표(사유별 개수 포함).
        비어 있으면 전부 통과한 것.
    """

    rows = []

    for _, row in df.iterrows():
        text = row.get(text_col) or ""
        hits = {
            name: len(pattern.findall(text))
            for name, pattern in PATTERN_TABLE_MARKUP_LEFTOVERS.items()
        }

        if any(hits.values()):
            identifier = row.get("filename", row.get("source"))
            rows.append({"filename": identifier, **hits})

    return pd.DataFrame(rows)


