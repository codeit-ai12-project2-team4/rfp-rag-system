"""
문서 구조 분석 모듈.
"""

import re

HEADER_PATTERN = re.compile(
    r"^\s*(?:[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.|\d+[.)]|[가-힣]\.|[□■○◦•▪▶])"
)


def split_sections(text: str) -> list[str]:
    """문서를 섹션 단위로 분리한다.

    Args:
        text: 문서 본문.

    Returns:
        섹션 리스트.
    """

    sections = []
    current = []

    for line in text.splitlines():

        if HEADER_PATTERN.match(line) and current:
            sections.append("\n".join(current))
            current = [line]
        else:
            current.append(line)

    if current:
        sections.append("\n".join(current))

    return sections