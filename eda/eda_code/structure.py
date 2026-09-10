"""
문서 구조 분석 모듈.
"""

import re
from collections import Counter

from eda.eda_code.patterns import HEADER_PATTERNS, METADATA_VARIANTS, STRUCTURE_PATTERNS

HEADER_PATTERN = re.compile(r"^\s*(?:[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.|\d+[.)]|[가-힣]\.|[□■○◦•▪▶])")


def analyze_headers(text):
    counts = Counter()
    for line in text.splitlines():
        for name, pattern in HEADER_PATTERNS.items():
            if re.search(pattern, line):
                counts[name] += 1
                break
    return counts


def analyze_metadata_variants(text):
    result = Counter()
    for patterns in METADATA_VARIANTS.values():
        for p in patterns:
            matches = re.findall(p, text)
            if matches:
                result.update(matches)
    return result


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


def analyze_table_like(text):
    patterns = [
        r"□\s*\([^)]+\)",  # 표의 항목명이 풀린 형태
        r"◦",
        r"▪",
        r"▶",
    ]

    return sum(len(re.findall(p, text)) for p in patterns)


def analyze_structure(text: str):

    result = {}

    for name, pattern in STRUCTURE_PATTERNS.items():
        result[name] = len(
            re.findall(
                pattern,
                text,
            )
        )

    return result


def extract_square_items(text: str):
    """
    RFP 문서에서 □ 항목을 추출한다.

    Returns:
        list[dict]: 각 항목에 대해
        - original: 원본 형태 (예: '□ (사업기간)')
        - label: 정제된 라벨 (예: '사업기간')
    """

    pattern = r"□\s*\(\s*([^)]+?)\s*\)"

    results = []

    for match in re.finditer(pattern, text):
        original = match.group(0)
        label = re.sub(r"\s+", "", match.group(1))

        results.append(
            {
                "original": original,
                "label": label,
            }
        )

    return results


def extract_heading_candidates(text: str):

    lines = text.splitlines()

    candidates = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if len(line) > 100:
            continue

        if re.match(
            r"^(Ⅰ|Ⅱ|Ⅲ|Ⅳ|Ⅴ|Ⅵ|"
            r"\d+\.|\d+\)|"
            r"[가-힣]\.|"
            r"[가-힣]\)|"
            r"□|■|○|◦)",
            line,
        ):
            candidates.append(line)

    return candidates
