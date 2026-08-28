import difflib
import re
import struct
import subprocess
import tempfile
import zlib
from collections.abc import Generator
from pathlib import Path

import olefile

# 대체 파서로 파싱한 문서 검수 코드

DATA_DIR = Path(r"C:\Users\asd\Desktop\v2_chosim\rfp-rag-system\data\files")

HWPTAG_PARA_TEXT = 0x43
_HWP_CONTROL_WITH_EXTRA = frozenset(
    {1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25}
)
_HWP_CONTROL_SKIP_ONLY = frozenset({0, 9, 26, 27, 28, 29, 30, 31})


def extract_text_libreoffice(path: Path) -> str:
    """LibreOffice CLI(soffice)를 사용해 HWP 문서에서 텍스트를 추출합니다.

    Args:
        path (Path): 변환할 대상 HWP 파일 경로.

    Returns:
        str: 변환 및 추출된 본문 텍스트.

    Raises:
        RuntimeError: LibreOffice 변환 프로세스 실패, 변환 결과 파일 누락,
            또는 결과 텍스트가 비어 있는 경우 발생.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        result = subprocess.run(
            [
                "soffice",
                "--headless",
                "--convert-to",
                "txt:Text",
                "--outdir",
                tmp_dir,
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"soffice 변환 실패: {result.stderr[:300]}")
        out_path = Path(tmp_dir) / f"{path.stem}.txt"
        if not out_path.exists():
            raise RuntimeError("soffice 변환 결과 파일 없음")
        text = out_path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            raise RuntimeError("soffice 변환 결과가 비어 있음")
        return text


def _hwp_is_compressed(ole: olefile.OleFileIO) -> bool:
    """HWP 파일의 압축 여부를 헤더 플래그를 통해 확인합니다.

    Args:
        ole (olefile.OleFileIO): 열려 있는 OLE 파일 객체.

    Returns:
        bool: 본문 스트림이 압축되어 있으면 True, 그렇지 않으면 False.
    """
    header = ole.openstream("FileHeader").read()
    return bool(struct.unpack("<I", header[36:40])[0] & 0x01)


def _hwp_iter_records(data: bytes) -> Generator[tuple[int, bytes], None, None]:
    """HWP 바이너리 데이터 스트림에서 레코드 태그 ID와 페이로드를 순회 생성합니다.

    Args:
        data (bytes): 파싱할 HWP 바이너리 스트림 데이터.

    Yields:
        tuple[int, bytes]: (레코드 태그 ID, 레코드 페이로드 바이트).
    """
    i, n = 0, len(data)
    while i + 4 <= n:
        header = struct.unpack("<I", data[i : i + 4])[0]
        tag_id = header & 0x3FF
        size = (header >> 20) & 0xFFF
        i += 4
        if size == 0xFFF:
            size = struct.unpack("<I", data[i : i + 4])[0]
            i += 4
        yield tag_id, data[i : i + size]
        i += size


def _hwp_parse_para_text(payload: bytes) -> str:
    """HWPTAG_PARA_TEXT 레코드 페이로드를 파싱하여 제어 문자를 제외한 텍스트를 추출합니다.

    Args:
        payload (bytes): HWPTAG_PARA_TEXT 레코드의 바이너리 페이로드.

    Returns:
        str: 복호화된 문단 텍스트.
    """
    chars, i, n = [], 0, len(payload)
    while i + 2 <= n:
        code = struct.unpack("<H", payload[i : i + 2])[0]
        i += 2
        if code in _HWP_CONTROL_SKIP_ONLY:
            continue
        if code in _HWP_CONTROL_WITH_EXTRA:
            i += 14
            continue
        if code in (10, 13):
            chars.append("\n")
            continue
        chars.append(chr(code))
    return "".join(chars)


def extract_text_hwp_raw(path: Path) -> str:
    """OLE 및 바이너리 파싱을 직접 수행하여 HWP 문서의 BodyText 스트림에서 텍스트를 추출합니다.

    Args:
        path (Path): 읽어올 HWP 파일 경로.

    Returns:
        str: 파싱된 HWP 본문 텍스트.

    Raises:
        RuntimeError: BodyText 스트림이 없거나 최종 파싱 텍스트가 비어 있는 경우 발생.
    """
    with olefile.OleFileIO(str(path)) as ole:
        compressed = _hwp_is_compressed(ole)
        sections = sorted(
            (e for e in ole.listdir() if len(e) == 2 and e[0] == "BodyText"),
            key=lambda e: int(re.sub(r"\D", "", e[1]) or 0),
        )
        if not sections:
            raise RuntimeError("BodyText 스트림을 찾을 수 없음")
        texts = []
        for entry in sections:
            raw = ole.openstream(entry).read()
            data = zlib.decompressobj(-15).decompress(raw) if compressed else raw
            for tag_id, payload in _hwp_iter_records(data):
                if tag_id == HWPTAG_PARA_TEXT:
                    texts.append(_hwp_parse_para_text(payload))
        text = "\n".join(texts)
        if not text.strip():
            raise RuntimeError("raw 파서 결과가 비어 있음")
        return text


def diff_ratio(a: str, b: str) -> float:
    """두 텍스트 간의 유사도 비율을 계산합니다.

    Args:
        a (str): 비교할 첫 번째 문자열.
        b (str): 비교할 두 번째 문자열.

    Returns:
        float: 0.0부터 1.0 사이의 유사도 비율.
    """
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def check_two_documents(filenames: list[str], data_dir: Path) -> None:
    """지정된 파일들에 대해 hwp_raw 및 LibreOffice 파싱을 수행하고 유사도를 비교·출력합니다.

    Args:
        filenames (list[str]): 검수할 HWP 파일명 또는 상대 경로 목록.
        data_dir (Path): HWP 파일이 저장된 기본 디렉터리 경로.
    """
    for filename in filenames:
        path = data_dir / filename
        print("=" * 80)
        print(f"[검수] {filename}")
        print("=" * 80)

        try:
            text_raw = extract_text_hwp_raw(path)
            print(f"hwp_raw     : 성공, 글자수 {len(text_raw)}")
        except RuntimeError as e:
            text_raw = None
            print(f"hwp_raw     : 실패 - {e}")

        try:
            text_lo = extract_text_libreoffice(path)
            print(f"libreoffice : 성공, 글자수 {len(text_lo)}")
        except RuntimeError as e:
            text_lo = None
            print(f"libreoffice : 실패 - {e}")

        if text_raw and text_lo:
            ratio = diff_ratio(text_raw, text_lo)
            flag = "" if ratio >= 0.9 else "  ⚠ 낮음, 수동 검수 필요"
            print(f"두 방식 유사도: {ratio:.1%}{flag}")

        sample = text_raw or text_lo or ""
        print("\n[본문 샘플 800자]")
        print(sample[:800])
        print()


check_two_documents(
    filenames=[
        "/home/spai1216/workspace/data/files/대전대학교_대전대학교 2024학년도 다층적 융합 학습경험 플랫폼(MILE) 전.hwp",
        "/home/spai1216/workspace/data/files/한국농어촌공사_아세안+3 식량안보정보시스템(AFSIS) 3단계 협력(캄보디아.hwp",
    ],
    data_dir=DATA_DIR,
)
