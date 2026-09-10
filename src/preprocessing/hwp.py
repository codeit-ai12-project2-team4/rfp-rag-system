"""HWP 5.x 본문 텍스트 추출기.

pyhwp(hwp5)는 Python 3.13에서 빌드가 깨지므로 의존하지 않는다.
HWP 5.0 파일은 OLE 복합 문서이고, 본문은 `BodyText/SectionN` 스트림에
zlib(raw deflate)로 압축된 레코드 스트림으로 들어 있다. 여기서
HWPTAG_PARA_TEXT(67) 레코드만 골라 UTF-16LE 문자열로 복원한다.

표(table) 안의 글자도 결국 별도 문단 레코드로 저장되므로 이 방식으로
같이 딸려 나온다. RFP는 요구사항이 대부분 표에 들어 있어서 이 점이 중요하다.
반대로 그림 안의 글자(스캔 이미지, 다이어그램)는 나오지 않는다.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import olefile

HWPTAG_BEGIN = 0x10
HWPTAG_PARA_TEXT = HWPTAG_BEGIN + 51  # 67

# HWP 본문 문자열 안의 제어문자 분류 (한글 파일 형식 5.0 문서 기준)
# - inline / extended 제어문자는 뒤에 14바이트 부가 정보를 달고 다닌다.
CTRL_EXTENDED = {1, 2, 3, 11, 12, 14, 15, 16, 17, 18, 21, 22, 23}
CTRL_INLINE = {4, 5, 6, 7, 8, 9, 19, 20}
CTRL_CHAR = {0, 10, 13, 24, 25, 26, 27, 28, 29, 30, 31}


class HwpParseError(RuntimeError):
    pass


def _decode_para_text(record: bytes) -> str:
    """PARA_TEXT 레코드 페이로드를 사람이 읽을 문자열로 바꾼다.

    코드 단위를 그때그때 chr()로 바꾸면 BMP 밖 글자의 서러게이트 쌍이 깨져서
    나중에 JSON 직렬화가 터진다. 살릴 코드 단위만 모아 마지막에 UTF-16LE로
    한 번에 디코딩하면 쌍이 제대로 합쳐지고, 짝 잃은 서러게이트는 버려진다.
    """
    buf = bytearray()
    i = 0
    n = len(record) - 1
    while i < n:
        (code,) = struct.unpack_from("<H", record, i)
        if code in CTRL_CHAR:
            # 13(CR)은 문단 구분자 역할을 하므로 줄바꿈으로 살린다.
            if code == 13:
                buf += b"\n\x00"
            i += 2
        elif code in CTRL_EXTENDED:
            i += 16  # 2바이트 코드 + 12바이트 정보 + 2바이트 종료 코드
        elif code in CTRL_INLINE:
            i += 2
        else:
            buf += record[i : i + 2]
            i += 2
    return buf.decode("utf-16-le", errors="ignore")


def _iter_records(data: bytes):
    """레코드 헤더(4바이트: tag_id 10 / level 10 / size 12)를 따라 순회한다."""
    i = 0
    total = len(data)
    while i + 4 <= total:
        (header,) = struct.unpack_from("<I", data, i)
        tag_id = header & 0x3FF
        size = (header >> 20) & 0xFFF
        i += 4
        if size == 0xFFF:  # 확장 크기
            (size,) = struct.unpack_from("<I", data, i)
            i += 4
        yield tag_id, data[i : i + size]
        i += size


def extract_hwp_text(path: str | Path) -> str:
    """HWP 파일에서 섹션 순서대로 본문 텍스트를 뽑는다."""
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
        encrypted = bool(flags & 0x02)
        if encrypted:
            raise HwpParseError(f"암호화된 문서: {path.name}")

        sections = sorted(
            (d for d in ole.listdir() if d[0] == "BodyText"),
            key=lambda d: int(d[1].replace("Section", "")),
        )
        if not sections:
            raise HwpParseError(f"BodyText 섹션 없음: {path.name}")

        parts: list[str] = []
        for entry in sections:
            raw = ole.openstream("/".join(entry)).read()
            if compressed:
                raw = zlib.decompress(raw, -15)
            for tag_id, payload in _iter_records(raw):
                if tag_id == HWPTAG_PARA_TEXT:
                    parts.append(_decode_para_text(payload))
        return "\n".join(parts)
    finally:
        ole.close()


def extract_hwp_preview_text(path: str | Path) -> str:
    """`PrvText` 스트림(미리보기 텍스트). 본문 파싱이 실패했을 때의 최후 수단."""
    ole = olefile.OleFileIO(str(path))
    try:
        if not ole.exists("PrvText"):
            return ""
        return ole.openstream("PrvText").read().decode("utf-16-le", errors="ignore")
    finally:
        ole.close()
