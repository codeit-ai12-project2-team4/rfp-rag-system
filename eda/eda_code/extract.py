"""
HWP/PDF 텍스트 추출 모듈.

우선순위

1. hwp5txt
2. LibreOffice
3. Raw HWP Parser
"""

import re
import struct
import subprocess
import tempfile
import zlib
from pathlib import Path

import olefile

DATA_DIR = Path("/home/spai1216/workspace/data/files")

SUPPORTED_EXTENSIONS = {
    ".hwp",
    ".pdf",
}

# hwp 텍스트 추출 시도 순서 (앞에서부터 성공할 때까지 시도)
HWP_EXTRACTION_METHODS = (
    "hwp5txt",
    "libreoffice",
    "hwp_raw",
)

HWPTAG_PARA_TEXT = 0x43  # HWPTAG_BEGIN(0x10) + 51

# 인라인 컨트롤 문자 중 부가 데이터(14바이트)를 동반하는 코드
_HWP_CONTROL_WITH_EXTRA = frozenset(
    {1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25}
)
# 완전히 건너뛰는(부가 데이터 없는) 제어 문자 코드
_HWP_CONTROL_SKIP_ONLY = frozenset({0, 9, 26, 27, 28, 29, 30, 31})


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


def _hwp_is_compressed(ole: olefile.OleFileIO) -> bool:
    header = ole.openstream("FileHeader").read()
    flags = struct.unpack("<I", header[36:40])[0]
    return bool(flags & 0x01)


def _hwp_iter_records(data: bytes):
    i = 0
    n = len(data)

    while i + 4 <= n:
        header = struct.unpack("<I", data[i : i + 4])[0]
        tag_id = header & 0x3FF
        size = (header >> 20) & 0xFFF
        i += 4

        if size == 0xFFF:
            size = struct.unpack("<I", data[i : i + 4])[0]
            i += 4

        payload = data[i : i + size]
        i += size

        yield tag_id, payload


def _hwp_parse_para_text(payload: bytes) -> str:
    chars = []
    i = 0
    n = len(payload)

    while i + 2 <= n:
        code = struct.unpack("<H", payload[i : i + 2])[0]
        i += 2

        if code in _HWP_CONTROL_SKIP_ONLY:
            continue

        if code in _HWP_CONTROL_WITH_EXTRA:
            i += 14  # 부가 데이터(근사치)
            continue

        if code == 10 or code == 13:
            chars.append("\n")
            continue

        chars.append(chr(code))

    return "".join(chars)


def extract_text_hwp_raw(path: Path) -> str:

    with olefile.OleFileIO(str(path)) as ole:
        compressed = _hwp_is_compressed(ole)

        sections = sorted(
            (
                entry
                for entry in ole.listdir()
                if len(entry) == 2 and entry[0] == "BodyText"
            ),
            key=lambda entry: int(re.sub(r"\D", "", entry[1]) or 0),
        )

        if not sections:
            raise RuntimeError("BodyText 스트림을 찾을 수 없음")

        texts = []

        for entry in sections:
            raw = ole.openstream(entry).read()

            if compressed:
                data = zlib.decompressobj(-15).decompress(raw)
            else:
                data = raw

            for tag_id, payload in _hwp_iter_records(data):
                if tag_id == HWPTAG_PARA_TEXT:
                    texts.append(_hwp_parse_para_text(payload))

        text = "\n".join(texts)

        if not text.strip():
            raise RuntimeError("raw 파서 결과가 비어 있음")

        return text


HWP_EXTRACTORS = {
    "hwp5txt": extract_text_hwp5txt,
    "libreoffice": extract_text_libreoffice,
    "hwp_raw": extract_text_hwp_raw,
}


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
