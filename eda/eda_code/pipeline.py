"""
전체 전처리 파이프라인.
"""

import re
from collections import Counter
from pathlib import Path

import pandas as pd

from .extract import extract_text
from .metadata import extract_metadata
from .patterns import STRUCTURE_PATTERNS
from .structure import (
    analyze_headers,
    analyze_metadata_variants,
    analyze_structure,
    analyze_table_like,
    extract_heading_candidates,
    extract_square_items,
    split_sections,
)

SUPPORTED_EXTENSIONS = {
    ".hwp",
    ".pdf",
}


def run(
    data_dir: Path,
    sample_size: int | None = None,
) -> pd.DataFrame:
    """디렉터리의 문서를 전처리한다.

    Args:
        data_dir: 문서 디렉터리.
        sample_size: 샘플 개수.

    Returns:
        결과 DataFrame.
    """

    files = sorted(p for p in data_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)

    if sample_size:
        files = files[:sample_size]

    rows = []

    for path in files:
        try:
            text = extract_text(path)

            row = {
                "파일명": path.name,
                "확장자": path.suffix,
                "글자수": len(text),
            }

            row.update(extract_metadata(text))

        except RuntimeError as e:
            row = {
                "파일명": path.name,
                "오류": str(e),
            }

        rows.append(row)

    return pd.DataFrame(rows)


def run_eda(data_dir: Path, sample_size=None):
    files = sorted(
        p for p in data_dir.iterdir() if p.suffix.lower() in (".pdf", ".hwp")
    )

    if sample_size:
        files = files[:sample_size]

    header_total = Counter()
    metadata_total = Counter()
    section_lengths = []
    table_summary = []

    for path in files:
        try:
            text = extract_text(path)
        except RuntimeError as e:
            print(f"Error occurred while extracting text from {path}: {e}")
            continue

        header_total.update(analyze_headers(text))
        metadata_total.update(analyze_metadata_variants(text))

        sections = split_sections(text)
        section_lengths.extend(len(s) for s in sections)

        table_summary.append(
            {"파일": path.name, "표추정줄수": analyze_table_like(text)}
        )

    print("=" * 70)
    print("1. 헤더/번호 체계 빈도")
    print("=" * 70)
    print(pd.Series(header_total).sort_values(ascending=False))

    print("\n" + "=" * 70)
    print("2. 메타데이터 표현 다양성")
    print("=" * 70)
    print(pd.Series(metadata_total).sort_values(ascending=False))

    print("\n" + "=" * 70)
    print("3. 섹션 길이 분포")
    print("=" * 70)
    print(pd.Series(section_lengths).describe())

    print("\n" + "=" * 70)
    print("4. 표 추출 품질")
    print("=" * 70)

    table_df = pd.DataFrame(table_summary)

    print(table_df.to_string(index=False))
    print(f"\n표 추정 줄 평균: {table_df['표추정줄수'].mean():.1f}")

    return {
        "header": pd.Series(header_total),
        "metadata": pd.Series(metadata_total),
        "section_lengths": pd.Series(section_lengths),
        "tables": table_df,
    }


def analyze_text_basic(text: str):

    lines = text.splitlines()

    non_empty_lines = [line.strip() for line in lines if line.strip()]

    korean = re.findall(r"[가-힣]", text)
    english = re.findall(r"[A-Za-z]", text)
    numbers = re.findall(r"\d", text)

    special = re.findall(
        r"[^\w\s가-힣]",
        text,
        flags=re.UNICODE,
    )

    return {
        "글자수": len(text),
        "전체줄수": len(lines),
        "실제내용줄수": len(non_empty_lines),
        "빈줄수": len(lines) - len(non_empty_lines),
        "한글수": len(korean),
        "영문수": len(english),
        "숫자수": len(numbers),
        "특수문자수": len(special),
    }


def analyze_text_quality(text: str):

    replacement = text.count(" ")
    null = text.count("\x00")

    control = sum(1 for c in text if ord(c) < 32 and c not in ("\n", "\r", "\t"))

    long_spaces = len(re.findall(r"[ \t]{5,}", text))

    long_newlines = len(re.findall(r"\n{4,}", text))

    return {
        "깨진문자": replacement,
        "NULL문자": null,
        "제어문자": control,
        "긴공백": long_spaces,
        "과도한줄바꿈": long_newlines,
    }


def analyze_document(path: Path):

    result = {
        "파일명": path.name,
        "확장자": path.suffix.lower(),
        "추출성공": False,
        "추출방법": None,
        "오류": None,
    }

    text, method, errors = extract_text(path)

    if not text.strip():
        result["오류"] = (
            "; ".join(f"{k}={v}" for k, v in errors.items()) or "추출된 텍스트 없음" # type: ignore
        )
        return result, ""

    result["추출성공"] = True
    result["추출방법"] = method

    result.update(analyze_text_basic(text))
    result.update(analyze_text_quality(text))
    result.update(analyze_structure(text))

    return result, text


def run_a2(
    data_dir: Path,
    sample_size: int | None = None,
):

    files = sorted(
        p
        for p in data_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if sample_size is not None:
        files = files[:sample_size]

    print("=" * 80)
    print("A-2 원본 문서 텍스트 EDA")
    print("=" * 80)

    print(f"분석 대상: {len(files)}개")

    rows = []

    texts = {}

    all_square_lines = []

    all_heading_candidates = []

    for i, path in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] {path.name}")

        result, text = analyze_document(path)

        if result["추출성공"] and result["추출방법"] != "hwp5txt":
            print(f"    -> 폴백 성공: {result['추출방법']}")
        elif not result["추출성공"]:
            print(f"    -> 추출 실패: {result['오류']}")

        rows.append(result)

        if text:
            texts[path.name] = text

            all_square_lines.extend(extract_square_items(text))

            all_heading_candidates.extend(extract_heading_candidates(text))

    df = pd.DataFrame(rows)

    return (
        df,
        texts,
        all_square_lines,
        all_heading_candidates,
    )


def print_a2_result(
    df,
    texts,
    square_lines,
    heading_candidates,
):

    # --------------------------------------------------------
    # 1. 추출 성공률 / 방법별 분포
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("[1] 텍스트 추출 성공률")
    print("=" * 80)

    total = len(df)
    success = df["추출성공"].sum()

    print(f"성공: {success}/{total}")
    print(f"성공률: {success / total * 100:.1f}%")

    print("\n[추출 방법별 문서 수]")
    print(df["추출방법"].fillna("실패").value_counts().to_string())

    if (df["추출성공"] == False).any():
        print("\n[추출 실패 문서]")
        print(df.loc[~df["추출성공"], ["파일명", "오류"]].to_string(index=False))

    # --------------------------------------------------------
    # 2. 텍스트 길이
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("[2] 텍스트 길이")
    print("=" * 80)

    print(df["글자수"].describe().to_string())

    print("\n[텍스트가 가장 긴 문서 TOP 10]")

    print(
        df[["파일명", "글자수"]]
        .sort_values("글자수", ascending=False)
        .head(10)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # 3. 텍스트 품질
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("[3] 텍스트 품질")
    print("=" * 80)

    print(
        df[
            [
                "파일명",
                "깨진문자",
                "NULL문자",
                "제어문자",
                "긴공백",
                "과도한줄바꿈",
            ]
        ]
        .sort_values("깨진문자", ascending=False)
        .head(20)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # 4. 구조 마커
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("[4] 구조 마커 전체 빈도")
    print("=" * 80)

    marker_columns = list(STRUCTURE_PATTERNS.keys())

    print(df[marker_columns].sum().sort_values(ascending=False).to_string())

    # --------------------------------------------------------
    # 5. □ 시작 항목
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("[5] □ 항목 분석")
    print("=" * 80)

    original_counter = Counter(item["original"] for item in square_lines)

    print("\n[원본 형태 TOP 20]")
    for original, count in original_counter.most_common(20):
        print(f"{count:4} | {original}")

    label_counter = Counter(item["label"] for item in square_lines)

    print("\n[라벨 TOP 20]")
    for label, count in label_counter.most_common(20):
        print(f"{count:4} | {label}")

    # --------------------------------------------------------
    # 6. 제목 후보
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("[6] 구조적 제목 후보 TOP 50")
    print("=" * 80)

    counter = Counter(heading_candidates)

    for line, count in counter.most_common(50):
        print(f"{count:4} | {line[:150]}")

    # --------------------------------------------------------
    # 7. 문서 샘플
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("[7] 문서 텍스트 샘플")
    print("=" * 80)

    for i, (filename, text) in enumerate(
        list(texts.items())[:3],
        start=1,
    ):
        print("\n")
        print("-" * 80)
        print(f"[문서 {i}] {filename}")
        print("-" * 80)

        print(text[:1500])

        print("\n...")
