"""
HWP/PDF 텍스트 추출 모듈.

우선순위

1. hwp5txt
2. LibreOffice
3. Raw HWP Parser
"""

import subprocess
import tempfile
from pathlib import Path


def extract_text_pdf(path: Path) -> str:
    """PDF에서 텍스트를 추출한다.

    Args:
        path: PDF 파일 경로.

    Returns:
        추출된 텍스트.

    Raises:
        RuntimeError:
            pdftotext 실행이 실패한 경우.
    """

    result = subprocess.run(
        ["pdftotext", str(path), "-"],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    return result.stdout


def extract_text_hwp5txt(path: Path) -> str:
    """hwp5txt를 이용해 HWP를 추출한다.

    Args:
        path: HWP 파일 경로.

    Returns:
        추출된 텍스트.

    Raises:
        RuntimeError:
            추출 실패.
    """

    result = subprocess.run(
        ["hwp5txt", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    return result.stdout


def extract_text_libreoffice(path: Path) -> str:
    """LibreOffice를 이용해 HWP를 TXT로 변환한다.

    Args:
        path: HWP 파일 경로.

    Returns:
        변환된 텍스트.

    Raises:
        RuntimeError:
            변환 실패.
    """

    with tempfile.TemporaryDirectory() as tmp:

        result = subprocess.run(
            [
                "soffice",
                "--headless",
                "--convert-to",
                "txt:Text",
                "--outdir",
                tmp,
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(result.stderr)

        return (Path(tmp) / f"{path.stem}.txt").read_text(
            encoding="utf-8",
            errors="replace",
        )


def extract_text(path: Path) -> str:
    """파일 형식에 맞게 텍스트를 추출한다.

    HWP는 여러 추출기를 순차적으로 시도한다.

    Args:
        path: 입력 파일.

    Returns:
        추출된 텍스트.

    Raises:
        RuntimeError:
            모든 추출기가 실패한 경우.
    """

    if path.suffix.lower() == ".pdf":
        return extract_text_pdf(path)

    for extractor in (
        extract_text_hwp5txt,
        extract_text_libreoffice,
    ):
        try:
            return extractor(path)
        except RuntimeError:
            continue

    raise RuntimeError(f"모든 추출 방식 실패: {path.name}")