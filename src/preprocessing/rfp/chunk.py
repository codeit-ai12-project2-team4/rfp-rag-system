"""청킹. 검색용·생성용 두 벌을 만든다.

원본: `preprocessing/pipeline.py` (전처리팀). 분할 경위는 `src/preprocessing/README.md`.
"""

from preprocessing.rfp.clean import (
    PATTERN_ROW_BLOCK_LINE,
    PATTERN_TABLE_ROW,
    strip_table_markup,
)

# 자를 때 시도하는 경계. 앞에서부터 되는 것을 쓴다.
_CHUNK_SEPARATORS = ["\n\n", "\n", " ", ""]

def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> list:
    """
    RecursiveCharacterTextSplitter 스타일로 text를 chunk_size 근처로 자르고,
    chunk_overlap만큼 겹치게 한다.

    Args:
        text: 청킹할 원문.
        chunk_size: 청크 최대 길이(문자 수).
        chunk_overlap: 인접 청크 간 겹치는 길이(문자 수).

    Returns:
        list[str]: 청크 목록. 빈 텍스트는 빈 리스트를 반환한다.
    """

    if not text:
        return []

    def _split(piece: str, separators: list) -> list:
        if len(piece) <= chunk_size:
            return [piece] if piece.strip() else []

        if not separators:
            # 더 쪼갤 구분자가 없으면 chunk_size 단위로 강제 분할
            return [piece[i : i + chunk_size] for i in range(0, len(piece), chunk_size)]

        sep, rest_separators = separators[0], separators[1:]
        parts = piece.split(sep) if sep else list(piece)

        chunks, buffer = [], ""

        for part in parts:
            candidate = buffer + sep + part if buffer else part

            if len(candidate) <= chunk_size:
                buffer = candidate
                continue

            if buffer:
                chunks.append(buffer)

            if len(part) > chunk_size:
                chunks.extend(_split(part, rest_separators))
                buffer = ""
            else:
                buffer = part

        if buffer:
            chunks.append(buffer)

        return chunks

    raw_chunks = _split(text, _CHUNK_SEPARATORS)

    if chunk_overlap <= 0 or len(raw_chunks) <= 1:
        return raw_chunks

    overlapped = [raw_chunks[0]]

    for chunk in raw_chunks[1:]:
        prev = overlapped[-1]
        carry = prev[-chunk_overlap:] if len(prev) > chunk_overlap else prev
        overlapped.append((carry + chunk)[: chunk_size + chunk_overlap])

    return overlapped


def _split_segments(text: str) -> list:
    """표 블록과 일반 텍스트로 나눈다. 표는 쪼개지 않을 한 덩어리다.

    strip_table_markup()이 쓰는 것과 같은 정규식으로 표 줄을 찾으므로,
    "무엇이 표인가"의 기준이 두 함수 사이에서 어긋나지 않는다.

    Args:
        text: 생성(LLM)용 텍스트.

    Returns:
        list[tuple[str, bool]]: (덩어리, 표인가) 목록. 원문 순서를 지킨다.
    """

    segments, buffer, in_table = [], [], False

    for line in text.split("\n"):
        is_table = bool(
            PATTERN_TABLE_ROW.match(line) or PATTERN_ROW_BLOCK_LINE.search(line)
        )
        if buffer and is_table != in_table:
            segments.append(("\n".join(buffer), in_table))
            buffer = []
        in_table = is_table
        buffer.append(line)

    if buffer:
        segments.append(("\n".join(buffer), in_table))

    return segments


def _trailing_overlap(chunk_pieces: list, chunk_overlap: int) -> str:
    """청크의 마지막 조각에서 overlap용 꼬리 텍스트를 뽑는다.

    [표 원자성 보호] 표 조각은 헤더 행이 있어야 `_flatten_table_block`이
    올바르게 파싱한다. overlap이 표 조각 중간을 문자 단위로 잘라 다음
    청크 앞에 붙이면, 헤더 없는 데이터 행이 뒤섞여 들어가고
    `_flatten_table_block`이 그 중 첫 데이터 행을 헤더로 오인해 나머지
    행에 반복 결합시킨다(예: "행24 행25 내용24 내용25" 형태로 깨짐).
    그래서 마지막 조각이 표면 overlap을 생략한다 — 일반 텍스트 조각만
    문자 단위로 안전하게 잘라 이어붙인다.

    Args:
        chunk_pieces: 해당 청크를 구성한 (조각 텍스트, 표 여부) 목록.
        chunk_overlap: 가져올 최대 길이.

    Returns:
        str: overlap로 앞에 붙일 텍스트. 표로 끝난 청크면 빈 문자열.
    """

    if not chunk_pieces:
        return ""

    last_text, last_is_table = chunk_pieces[-1]
    if last_is_table:
        return ""

    return last_text[-chunk_overlap:] if len(last_text) > chunk_overlap else last_text


# 표 크기가 chunk_size의 이 배수를 넘으면 행 단위로 쪼갠다.
TABLE_SPLIT_MULTIPLIER = 3


def _split_oversized_table(table_text: str, chunk_size: int) -> list:
    """표 하나가 chunk_size의 TABLE_SPLIT_MULTIPLIER배를 넘으면 행 단위로
    쪼갠다. 넘지 않으면 원문을 그대로 담은 1개짜리 리스트를 돌려준다
    (기존 표 원자성 동작 유지).

    [배경] `_flatten_table_block`이 병합 표를 행마다 "헤더: 값"으로
    풀어 쓰기 때문에, 행이 아주 많은 서식표 하나가 chunk_size의 수십
    배까지 부풀 수 있다(실측 29,913자). 그런 청크가 검색에 뽑히면
    컨텍스트 예산 안에서 근거가 그 표 하나로 독점된다.

    마크다운 파이프 표(헤더행+구분행+데이터행)는 앞 두 줄(헤더, 구분행)을
    조각마다 복제해 `_flatten_table_block`이 계속 올바르게 파싱하게
    한다. "[행 N] 헤더: 값" 행블록 표는 이미 행마다 헤더가 박혀있어
    복제 없이 줄 단위로만 나누면 된다.

    Args:
        table_text: 표 원문(생성용, `_split_segments`가 뽑은 원자적 조각).
        chunk_size: 기준 청크 크기(검색용 문자 수).

    Returns:
        list[str]: 나눠진 표 조각들. 임계값 이하면 [table_text].
    """

    if len(strip_table_markup(table_text)) <= chunk_size * TABLE_SPLIT_MULTIPLIER:
        return [table_text]

    lines = table_text.split("\n")
    is_row_block = bool(lines) and bool(PATTERN_ROW_BLOCK_LINE.search(lines[0]))
    header_lines = [] if is_row_block else lines[:2]
    data_lines = lines if is_row_block else lines[2:]

    groups, buffer_lines = [], list(header_lines)

    for line in data_lines:
        candidate_lines = buffer_lines + [line]
        has_data = len(buffer_lines) > len(header_lines)
        if (
            has_data
            and len(strip_table_markup("\n".join(candidate_lines))) > chunk_size
        ):
            groups.append("\n".join(buffer_lines))
            buffer_lines = header_lines + [line]
        else:
            buffer_lines = candidate_lines

    if len(buffer_lines) > len(header_lines):
        groups.append("\n".join(buffer_lines))

    return groups if groups else [table_text]


def chunk_pairs(text: str, chunk_size: int = 1500, chunk_overlap: int = 250) -> list:
    """생성용 텍스트를 잘라 (검색용, 생성용) 청크 쌍을 만든다.

    자르는 대상은 생성용이지만 **길이는 검색용 기준으로 잰다.** 검색팀의
    chunk_size는 검색용 텍스트로 격자 실험을 해서 고른 값이라, 마크업
    몫만큼 짧아지면 그 실험이 무의미해진다.

    표 블록은 통째로 한 청크에 들어간다. 표 하나가 chunk_size보다 크면
    그 표만 단독으로 더 큰 청크가 된다(쪼개서 헤더와 값이 갈리는 것보다
    낫다는 판단) — 단, chunk_size의 TABLE_SPLIT_MULTIPLIER배를 넘는
    극단적인 경우는 `_split_oversized_table`이 행 단위로 쪼갠다(마크다운
    표는 헤더/구분행을 조각마다 복제). 같은 이유로 overlap도 조각(piece)
    단위로 계산해 표 조각 중간을 자르지 않는다(`_trailing_overlap` 참고).

    Args:
        text: 생성(LLM)용 텍스트.
        chunk_size: 청크 최대 길이. **검색용 기준 문자 수.**
        chunk_overlap: 인접 청크 간 겹치는 길이.

    Returns:
        list[tuple[str, str]]: (검색용, 생성용) 쌍 목록.
    """

    if not text:
        return []

    pieces = []
    for segment, is_table in _split_segments(text):
        if is_table:
            pieces.extend(
                (piece, True) for piece in _split_oversized_table(segment, chunk_size)
            )
        else:
            pieces.extend(
                (p, False) for p in chunk_text(segment, chunk_size, chunk_overlap=0)
            )

    chunk_piece_lists, buffer_pieces, buffer_text = [], [], ""

    for piece, is_table in pieces:
        candidate = f"{buffer_text}\n{piece}" if buffer_text else piece
        if buffer_text and len(strip_table_markup(candidate)) > chunk_size:
            chunk_piece_lists.append(buffer_pieces)
            buffer_pieces, buffer_text = [(piece, is_table)], piece
        else:
            buffer_pieces.append((piece, is_table))
            buffer_text = candidate

    if buffer_pieces:
        chunk_piece_lists.append(buffer_pieces)

    chunks = [
        "\n".join(piece for piece, _ in chunk_pieces)
        for chunk_pieces in chunk_piece_lists
    ]

    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for prev_pieces, chunk in zip(chunk_piece_lists, chunks[1:]):
            carry = _trailing_overlap(prev_pieces, chunk_overlap)
            overlapped.append(f"{carry}\n{chunk}" if carry else chunk)
        chunks = overlapped

    return [(strip_table_markup(chunk), chunk) for chunk in chunks]


