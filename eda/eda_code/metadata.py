"""
메타데이터 추출 모듈.
"""

import re

from .patterns import PATTERNS


def extract_field(
    text: str,
    pattern: str,
    window: int = 100,
) -> str | None:
    """패턴 이후 값을 추출한다.

    Args:
        text: 문서 텍스트.
        pattern: 메타데이터 정규식.
        window: 최대 탐색 길이.

    Returns:
        추출된 문자열.
    """

    match = re.search(pattern, text)

    if match is None:
        return None

    value = text[match.end() : match.end() + window]
    value = re.sub(r"^[\s:\)]*", "", value)

    boundary = re.search(r"[□\n]", value)

    if boundary:
        value = value[: boundary.start()]

    value = value.strip()

    return value or None


def extract_metadata(text: str) -> dict:
    """문서에서 주요 메타데이터를 추출한다.

    Args:
        text: 문서 본문.

    Returns:
        메타데이터 딕셔너리.
    """

    return {
        name: extract_field(text, pattern)
        for name, pattern in PATTERNS.items()
    }