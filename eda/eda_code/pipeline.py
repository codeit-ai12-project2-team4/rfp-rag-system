"""
전체 전처리 파이프라인.
"""

from pathlib import Path

import pandas as pd

from .extract import extract_text
from .metadata import extract_metadata

SUPPORTED = {".pdf", ".hwp"}


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

    files = sorted(p for p in data_dir.iterdir() if p.suffix.lower() in SUPPORTED)

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
