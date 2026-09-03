import difflib
import json
import os
import re
import struct
import subprocess
import tempfile
import traceback
import unicodedata
import zlib
from dataclasses import dataclass
from pathlib import Path

import olefile
import pandas as pd
import pdfplumber

# ============================================================
# 0. 설정 - 추출 전략
# ============================================================

# 이 순서만 바꾸면 HWP 추출 전략이 바뀐다.
# hwp_raw를 최우선으로 두는 이유: 표 구조 복원이 가능한 유일한 방법이기 때문.
# 실행 결과:
#   output/cleaned_documents.csv    - 사람 검수/공유용
#   output/cleaned_documents.xlsx   - 위와 동일, 엑셀
#   output/cleaned_documents.jsonl  - LangChain Document 형태(page_content+metadata), 팀 JSONL 파이프라인용
#   output/cleaned_texts/*.txt      - 문서별 정제 텍스트
#   output/preprocessing_report.json - 실행 통계 요약
HWP_EXTRACTION_METHODS = [
    "hwp_raw",
    "hwp5txt",
    "libreoffice",
]

HANGUL_MIN_RATIO = 0.15
HANGUL_CHECK_CHARS = 5000

SUPPORTED_EXTENSIONS = {".hwp", ".pdf"}


# ============================================================
# 1. 설정 - HWP 레코드 태그 / 표 스펙
# ============================================================

HWPTAG_PARA_HEADER = 0x42
HWPTAG_PARA_TEXT = 0x43
HWPTAG_CTRL_HEADER = 0x47
HWPTAG_LIST_HEADER = 0x48
HWPTAG_TABLE = 0x4D

TABLE_CTRL_ID = b"tbl "

# --- [수정 1] ---
# 0x09(탭)는 값이 아니라 "제어문자 + 파라미터 14바이트(리더 문자 종류,
# 폭 등) + 제어문자 반복"으로 인코딩되는 확장 제어문자다. 기존 코드는
# 9를 SKIP_ONLY(추가 데이터 없음)로 분류해 파라미터 14바이트가 텍스트로
# 새어나갔고, 이게 UTF-16LE로 강제 디코딩되며 mojibake(예: "穈ă")를
# 만들었다. 9를 WITH_EXTRA로 옮긴다.
_HWP_CONTROL_WITH_EXTRA = frozenset(
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25}
)
_HWP_CONTROL_SKIP_ONLY = frozenset({0, 26, 27, 28, 29, 30, 31})

# 같은 레벨에서 나타나도 "새 형제 컨테이너의 시작"으로 취급해 이전
# 프레임을 닫아야 하는 태그. PARA_HEADER는 여기 포함되지 않는다 -
# 같은 리스트(칸) 안의 문단이 이어지는 것뿐이기 때문이다. (수정 2 참고)
_SIBLING_CONTAINER_TAGS = frozenset({HWPTAG_CTRL_HEADER, HWPTAG_LIST_HEADER})

KEYVALUE_MAX_KEY_LENGTH = 20

# [진단용] table_pending이 TABLE 레코드를 못 받고 닫히는 원인을 사람이
# 읽을 수 있게 표시하기 위한 태그 이름 매핑.
_HWP_TAG_NAMES = {
    HWPTAG_PARA_HEADER: "PARA_HEADER",
    HWPTAG_PARA_TEXT: "PARA_TEXT",
    HWPTAG_CTRL_HEADER: "CTRL_HEADER",
    HWPTAG_LIST_HEADER: "LIST_HEADER",
    HWPTAG_TABLE: "TABLE",
}


def _describe_hwp_tag(tag_id: int) -> str:
    return _HWP_TAG_NAMES.get(tag_id, f"TAG_0x{tag_id:02X}")


# [수정 1 - 표 성공 판정 완화] 셀 하나만 비어도 표 전체를 실패로 보던
# 기존 판정은 실제로는 거의 다 채워진 표까지 실패로 깎아내렸다.
# 채워진 비율(fill_ratio) 기준으로 성공/부분복원/실패 3단계로 나눈다.
TABLE_FULL_SUCCESS_RATIO = 0.95
TABLE_PARTIAL_MIN_RATIO = 0.70


# ============================================================
# 2. 설정 - 노이즈 정제 정규식
# ============================================================

PATTERN_LONG_SPACE = re.compile(r"[ \t]{5,}")
PATTERN_LONG_NEWLINE = re.compile(r"\n{4,}")
PATTERN_PLACEHOLDER = re.compile(r"[ㅇ○△]{2,}")
PATTERN_EMPTY_PAREN = re.compile(r"\(\s*\)")
PATTERN_HWP_TAG_RESIDUE = re.compile(r"<(표|그림)>")
PATTERN_CONTROL_CHAR = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
PATTERN_PAGE_NUMBER_DASH = re.compile(r"^\s*-\s*\d+\s*-\s*$")

# [수정 6 - 인코딩 노이즈 제거] UTF-16 서로게이트 결합 등 디코딩 과정에서
# 드물게 남는 mojibake(예: 螨, 硿, 肼, 穈, ȃ)를 제거한다. 괄호 안 한자
# 병기(예: "주식회사(株式會社)")는 정상 표기이므로 보존한다.
# 주의: lookaround(?<!\()...(?!\))를 문자클래스+ 안에 직접 걸면, 매치가
# 괄호에 막혀 실패할 때 정규식 엔진이 조용히 백트래킹해 런 중간 일부만
# 삭제하는 버그가 생긴다(예: "(株式會社)" -> "(株社)"). 그래서 lookaround
# 없이 런 전체를 먼저 찾은 뒤, sub 콜백에서 "매치 앞이 ( 이고 매치 뒤가 )"
# 인 경우만 통째로 보존하고 그 외는 통째로 제거하는 방식으로 바꿨다.
# 남은 위험(회귀 가능성): 괄호 밖에 단독으로 쓰인 정상 한자는 여전히
# 함께 제거될 수 있다 (응답 하단 요약 참고).
PATTERN_MOJIBAKE = re.compile(
    r"[\u4E00-\u9FFF\u3400-\u4DBF\u0100-\u024F\u0370-\u03FF]+"
)


def _is_parenthesized_mojibake_match(match: re.Match) -> bool:
    """매치 전체가 괄호로 둘러싸여 있으면(정상 한자 병기로 보고) True."""

    text = match.string
    start, end = match.start(), match.end()

    return start > 0 and text[start - 1] == "(" and end < len(text) and text[end] == ")"


# ============================================================
# 3. 설정 - boilerplate (EDA에서 확인된 49건)
# ============================================================
#
# 상위 5건은 계약서 서식이 아니라 RFP 평가기준 카테고리 헤더라서
# 청킹 앵커로 남길지 팀 논의가 필요함 → 기본값은 "유지"(제거하지 않음).
# 제거하기로 결정되면 REMOVE_CATEGORY_HEADERS = True 로 바꾸면 됨.

BOILERPLATE_CATEGORY_HEADERS = [
    "□ 플랫폼 및 기반구조 분야",
    "□ 요소기술 분야",
    "□ 인터페이스 및 통합 분야",
    "□ 법률 및 고시",
    "□ 서비스 접근 및 전달 분야",
]

REMOVE_CATEGORY_HEADERS = False

BOILERPLATE_PATTERNS = [
    "보안 위약금 부과 기준",
    "사업자 보안위규 처리기준",
    "2. 보안 위약금은 다른 요인에 의해 상쇄, 삭감이 되지 않도록 부과",
    "(단위 : 천원)",
    "1. 위규 수준별로 A~D 등급으로 차등 부과",
    "누출금지 대상정보",
    "일반현황 및 연혁",
    "20 년 월 일",
    "1. 개인정보의 처리 현황",
    "제 안 요 청 서",
    "2. 계약금액 :",
    "3. 개인정보 접근 또는 접속 대상자",
    "2. 주사무소소재지 :",
    "3. 대 표 자 성 명 :",
    "1. 계약건명 :",
    "1. 발주기관과의 계약내용 변경에 따라 계약금액이 증감되었을 경우",
    "* 보안사고는 1회의 사고만으로도 그 파급력이 큰 것을 감안하여 타 항목과 별도 부과",
    "대 표 자 : (인)",
    "1. 발주자 및 구성원 전원이 동의하는 경우",
    "3. 발주자명 :",
    "2. 개인정보의 접근 또는 접속현황",
    "②이 협정서에 규정되지 아니한 사항은 운영위원회에서 정한다.",
    "공동수급체 구성원",
    "6. 국가용 보안시스템 및 정보보호시스템 도입 현황",
    "20 . . .",
    "1. 제안서의 효력",
    "공동수급체 대표자",
    "2. 세부 정보시스템 구성현황 및 정보통신망 구성도",
    "제2조(공동수급체) 공동수급체의 명칭, 사업소의 소재지, 대표자는 다음과 같다.",
    "제3조(공동수급체의 구성원) ①공동수급체의 구성원은 다음과 같다.",
    "2024년 월 일",
    "1. ㅇㅇㅇ회사(공동수급체대표자) : ㅇㅇ은행, 계좌번호 ㅇㅇㅇ, 예금주 ㅇㅇㅇ",
    "위와 같이 공동수급협정을 체결하고 그 증거로서 협정서 ㅇ통을 작성하여 각 통에 공동수급체 구성원이 기명날인하여 각자 보관한다.",
    "2. ㅇㅇㅇ회사 : ㅇㅇ은행, 계좌번호 ㅇㅇㅇ, 예금주 ㅇㅇㅇ",
    "1. 명 칭 : ㅇㅇㅇ",
    "제7조(하도급) 공동수급체 구성원 중 일부 구성원이 단독으로 하도급계약을 체결하고자 하는 경우에는 다른 구성원의 동의를 받아야 한다.",
    "3. 사용자계정ㆍ비밀번호 등 정보시스템 접근권한 정보",
    "주사무소 소재지",
    "4. 정보통신망 취약점 분석·평가 결과물",
    "2. 추진배경 및 필요성",
    "* 등급별 평점이 소수점 이하의 숫자가 있는 경우 소수점 다섯째자리에서 반올림 함",
    "5. 용역사업 결과물 및 프로그램 소스코드",
    "제4조 (위탁업무 기간) 이 계약서에 의한 개인정보 처리업무의 기간은 다음과 같다.",
    "②공동수급체의 대표자는 ㅇㅇㅇ로 한다.",
]


def get_boilerplate_lines() -> set:
    lines = set(BOILERPLATE_PATTERNS)
    if REMOVE_CATEGORY_HEADERS:
        lines |= set(BOILERPLATE_CATEGORY_HEADERS)
    return lines


# ============================================================
# 4. 설정 - 메타데이터 필드
# ============================================================

# 프로젝트 시작 시 받은 원본 메타데이터 CSV의 컬럼.
# 이 필드들은 이미 정답이 있으므로 문서 본문에서 다시 추출하지 않고
# 원본 CSV 값을 그대로 신뢰한다(파일명 기준으로 병합).
ORIGINAL_METADATA_COLUMNS = [
    "공고번호",
    "공고차수",
    "사업명",
    "사업금액",
    "발주기관",
    "공개기관",
    "공개일자",
    "입찰참여시작일",
    "입찰참여마감일",
    "사업요약",
    "파일형식",
    "파일명",
]

# 원본 CSV에 없는, 문서 본문에서만 뽑을 수 있는 필드.
FIELD_ALIASES = {
    "사업기간": ["사업기간", "용역기간", "수행기간", "계약기간"],
    "참가자격": ["참가자격", "입찰참가자격"],
    "계약방식": ["계약방식", "계약방법"],
    "평가방식": ["평가방법", "평가기준"],
    "보안특약": ["보안특약"],
    "하자보수기간": ["하자보수"],
}

# [수정 7 - 메타데이터 여러 줄 추출] 기존에는 \n도 경계로 취급해서
# "사업기간:\n계약일로부터\n12개월"처럼 값이 여러 줄에 걸치면 첫 줄에서
# 잘렸다. \n을 경계에서 빼고, 다음 "큰" 구조 마커(□■○◦※)가 나올 때까지만
# 경계로 삼는다. 대신 몇 줄까지 이어붙일지는 METADATA_VALUE_MAX_LINES로
# 별도 제한한다(무한정 이어지는 것을 방지).
METADATA_VALUE_BOUNDARY = re.compile(r"[□■○◦※]")
METADATA_VALUE_MAX_LEN = 200
METADATA_VALUE_MAX_LINES = 3

# [버그 수정] □■○◦※ 마커가 없어도 "4.", "나." 같은 새 조항 번호가 시작되면
# 값이 끝난 것으로 본다. 이게 없으면 사업기간 값이 다음 줄의 다른 필드
# ("4. 참가자격 : ...")까지 그대로 삼켜버린다.
_METADATA_LINE_BOUNDARY = re.compile(r"^(?:\d+\.|[가-힣]\.)\s")


# ============================================================
# 5. 예외 / 결과 타입
# ============================================================


class HwpParseError(Exception):
    """HWP 파싱 실패(품질 검증 실패 포함)를 나타낸다."""


@dataclass
class TableParseResult:
    success: bool
    tables_total: int
    tables_failed: int
    # [수정 1] 완전 실패도 완전 성공도 아닌 "부분 복원" 표 개수를 별도로 센다.
    tables_partial: int = 0


@dataclass
class ExtractionResult:
    text: str
    extractor: str | None
    table_parse_success: bool
    tables_total: int
    tables_failed: int
    error_reason: str | None
    fallback_used: bool
    attempted_errors: dict  # 성공 여부와 무관하게, 시도했다가 실패한 방법들의 에러 기록
    tables_partial: int = 0  # [수정 1] 부분 복원된 표 개수


# ============================================================
# 6. 품질 검증
# ============================================================


def _check_hangul(
    text: str,
    sample_size: int = HANGUL_CHECK_CHARS,
    min_ratio: float = HANGUL_MIN_RATIO,
) -> bool:
    """앞 sample_size자의 한글 비율이 min_ratio 미만이면 비정상 추출로 본다."""

    sample = text[:sample_size]

    if not sample:
        return False

    hangul = len(re.findall(r"[가-힣]", sample))

    return (hangul / len(sample)) >= min_ratio


# ============================================================
# 7. HWP 추출 - hwp5txt / LibreOffice (텍스트만, 표 구조 없음)
# ============================================================


def extract_text_hwp5txt(path: Path) -> str:

    result = subprocess.run(
        ["hwp5txt", str(path)], capture_output=True, text=True, timeout=60, check=False
    )

    if result.returncode != 0:
        raise HwpParseError(f"hwp5txt 실패: {result.stderr[:300]}")

    if not result.stdout.strip():
        raise HwpParseError("hwp5txt 결과가 비어 있음")

    return result.stdout


def extract_text_libreoffice(path: Path) -> str:

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
            raise HwpParseError(f"soffice 변환 실패: {result.stderr[:300]}")

        produced = list(Path(tmp_dir).glob("*.txt"))

        if not produced:
            raise HwpParseError("soffice 변환 결과 파일 없음")

        text = produced[0].read_text(encoding="utf-8", errors="replace")

        if not text.strip():
            raise HwpParseError("soffice 변환 결과가 비어 있음")

        return text


# ============================================================
# 8. HWP 추출 - OLE 레코드 저수준 유틸
# ============================================================


def _hwp_is_compressed(ole: olefile.OleFileIO) -> bool:
    header = ole.openstream("FileHeader").read()
    return bool(struct.unpack("<I", header[36:40])[0] & 0x01)


def _hwp_iter_records(data: bytes):
    """(tag_id, level, payload)를 순서대로 yield한다."""

    i, n = 0, len(data)

    while i + 4 <= n:
        header = struct.unpack("<I", data[i : i + 4])[0]
        tag_id = header & 0x3FF
        level = (header >> 10) & 0x3FF
        size = (header >> 20) & 0xFFF
        i += 4

        if size == 0xFFF:
            size = struct.unpack("<I", data[i : i + 4])[0]
            i += 4

        payload = data[i : i + size]
        i += size

        yield tag_id, level, payload


def _hwp_decode_para_text(payload: bytes) -> str:

    chars = []
    i, n = 0, len(payload)

    while i + 2 <= n:
        code = struct.unpack("<H", payload[i : i + 2])[0]
        i += 2

        if code in _HWP_CONTROL_SKIP_ONLY:
            continue

        if code in _HWP_CONTROL_WITH_EXTRA:
            # [수정 1] 탭(9)은 뒤따르는 14바이트가 리더문자/폭 등의
            # 바이너리 파라미터일 뿐 텍스트가 아니므로 건너뛴다.
            # 사람이 읽는 텍스트에서는 탭 문자 하나로만 남긴다.
            if code == 9:
                chars.append("\t")
            i += 14
            continue

        if code in (10, 13):
            chars.append("\n")
            continue

        if 0xD800 <= code <= 0xDBFF:
            # 상위 서로게이트: 다음 2바이트가 하위 서로게이트면 하나의 문자로 결합
            if i + 2 <= n:
                low = struct.unpack("<H", payload[i : i + 2])[0]
                if 0xDC00 <= low <= 0xDFFF:
                    i += 2
                    codepoint = 0x10000 + (code - 0xD800) * 0x400 + (low - 0xDC00)
                    chars.append(chr(codepoint))
                    continue
            # 짝이 없는 상위 서로게이트 - 손상된 데이터로 보고 폐기
            continue

        if 0xDC00 <= code <= 0xDFFF:
            # 짝 없이 단독으로 나온 하위 서로게이트 - 폐기
            continue

        chars.append(chr(code))

    return "".join(chars)


def _hwp_parse_table_dims(payload: bytes) -> tuple[int, int]:
    rows = struct.unpack_from("<H", payload, 4)[0]
    cols = struct.unpack_from("<H", payload, 6)[0]
    return rows, cols


def _hwp_parse_list_header_addr(payload: bytes) -> dict:

    if len(payload) < 16:
        return {"col": 0, "row": 0, "colspan": 1, "rowspan": 1}

    return {
        "col": struct.unpack_from("<H", payload, 8)[0],
        "row": struct.unpack_from("<H", payload, 10)[0],
        "colspan": struct.unpack_from("<H", payload, 12)[0],
        "rowspan": struct.unpack_from("<H", payload, 14)[0],
    }


# ============================================================
# 9. HWP 추출 - 표 구조 복원 (핵심)
# ============================================================


def extract_text_hwp_raw_structured(path: Path) -> tuple[str, TableParseResult]:
    """
    OLE 레코드를 레벨 기준 스택으로 순회하며 표를 복원한다.

    레코드 계층:
        PARA_HEADER
          PARA_TEXT           본문 글자
          CTRL_HEADER         payload[:4][::-1] == b"tbl " 이면 표 시작
            TABLE             행수/열수
            LIST_HEADER       칸 하나의 시작 (칸 주소 포함)
              PARA_HEADER     칸 안 문단
                PARA_TEXT     칸 안 글자

    표가 일부 실패해도(격자를 못 채우면) 문서 전체를 실패로 보지 않고
    '-'로 채워 표시한 채 나머지는 그대로 사용한다.

    [수정 2 - 핵심 버그]
    HWP5 포맷에서 LIST_HEADER(칸/리스트 시작)와 그 안의 "첫 번째 이후"
    문단들의 PARA_HEADER는 같은 level 값을 공유한다(문단마다 새 레벨이
    아니라, 같은 리스트의 형제 항목이기 때문). 그런데 기존 pop 조건
    `stack[-1]["level"] >= level`은 "같음(==)"도 pop 대상으로 취급해서,
    칸이 열리자마자 그 칸의 첫 PARA_HEADER가 도착하는 순간 칸 프레임이
    내용물이 채워지기도 전에 즉시 닫혀버렸다. 그 결과 칸의 실제 텍스트는
    이미 부모(표) 프레임으로 새어나가 버리고, cells[addr]["text"]는
    거의 항상 빈 문자열로 남았다. (실측: 이 문서 기준 표 렌더링 라인
    1148개 중 1141개, 즉 99.4%가 완전히 빈 값으로 나왔음 - 표
    "성공"으로 집계된 178개 표도 사실상 내용이 전부 유실된 상태였다.)

    고친 조건:
      (a) 더 얕은 레벨(level < frame.level)이면 무조건 닫는다.
      (b) 같은 레벨(level == frame.level)이어도, 들어오는 레코드가
          "새 형제 컨테이너의 시작"을 뜻하는 CTRL_HEADER/LIST_HEADER일
          때만 닫는다. PARA_HEADER(같은 리스트의 다음 문단)는 같은
          레벨이어도 열어둔 채로 둔다.

    이 수정은 표 셀 안에 인라인 개체(텍스트박스 등)가 들어있어 더 깊은
    level에서 발생하는 자체 LIST_HEADER(표와 무관한 문단 리스트)를
    표의 칸으로 잘못 인식해 주소가 덮어써지는 문제(이번에 실패로
    집계됐던 유일한 표, table_id=72)도 함께 해결한다 - 그 LIST_HEADER는
    level이 셀 프레임의 level보다 깊으므로 이제는 셀이 아닌 별도의
    "list" 프레임으로 올바르게 격리된다.
    """

    with olefile.OleFileIO(str(path)) as ole:
        compressed = _hwp_is_compressed(ole)

        sections = sorted(
            (e for e in ole.listdir() if len(e) == 2 and e[0] == "BodyText"),
            key=lambda e: int(re.sub(r"\D", "", e[1]) or 0),
        )

        if not sections:
            raise HwpParseError("BodyText 스트림을 찾을 수 없음")

        root = {"type": "root", "level": -1, "buffer": [""]}
        stack = [root]

        tables_total = 0
        tables_failed = 0
        tables_partial = 0  # [수정 1] 부분 복원된 표 개수 추적

        for entry in sections:
            raw = ole.openstream(entry).read()
            data = zlib.decompressobj(-15).decompress(raw) if compressed else raw

            for tag_id, level, payload in _hwp_iter_records(data):
                # [수정 2] 레벨 기준으로 형제/부모 경계 정리
                while len(stack) > 1 and (
                    stack[-1]["level"] > level
                    or (
                        stack[-1]["level"] == level
                        and tag_id in _SIBLING_CONTAINER_TAGS
                    )
                ):
                    finished = stack.pop()
                    if finished["type"] == "table_pending":
                        # [진단용] TABLE 레코드를 못 받은 채 닫히는 경우,
                        # 어떤 태그/레벨이 이 프레임을 닫았는지 기록한다.
                        # pop 조건 자체는 그대로이고, 닫히는 순간의 정보만
                        # 남긴다.
                        finished["_closed_by"] = (
                            f"{_describe_hwp_tag(tag_id)}(level={level})"
                        )
                    tables_total, tables_failed, tables_partial = _finalize_frame(
                        finished,
                        stack[-1],
                        tables_total,
                        tables_failed,
                        tables_partial,
                    )

                if tag_id == HWPTAG_CTRL_HEADER:
                    if payload[:4][::-1] == TABLE_CTRL_ID:
                        stack.append(
                            {
                                "type": "table_pending",
                                "level": level,
                                "rows": None,
                                "cols": None,
                                "cells": {},
                                "buffer": [""],  # 표 레벨에 낀 비정상 텍스트 대비
                            }
                        )
                    # 표가 아닌 컨트롤(그림 등)은 텍스트에 영향 없으므로 무시

                elif tag_id == HWPTAG_TABLE and stack[-1]["type"] == "table_pending":
                    rows, cols = _hwp_parse_table_dims(payload)
                    stack[-1].update(type="table", rows=rows, cols=cols)

                elif tag_id == HWPTAG_LIST_HEADER:
                    addr = _hwp_parse_list_header_addr(payload)

                    if stack[-1]["type"] == "table":
                        stack.append(
                            {
                                "type": "cell",
                                "level": level,
                                "addr": (addr["row"], addr["col"]),
                                "rowspan": max(1, addr["rowspan"]),
                                "colspan": max(1, addr["colspan"]),
                                "buffer": [""],
                            }
                        )
                    else:
                        # 표 밖의 리스트(머리말/꼬리말/각주/텍스트박스 등)
                        # - 일반 텍스트로 취급
                        stack.append(
                            {
                                "type": "list",
                                "level": level,
                                "buffer": [""],
                            }
                        )

                elif tag_id == HWPTAG_PARA_HEADER:
                    stack[-1]["buffer"].append("")

                elif tag_id == HWPTAG_PARA_TEXT:
                    piece = _hwp_decode_para_text(payload)
                    if not stack[-1]["buffer"]:
                        stack[-1]["buffer"].append("")
                    stack[-1]["buffer"][-1] += piece

        while len(stack) > 1:
            finished = stack.pop()
            if finished["type"] == "table_pending":
                # [진단용] 여기서 닫히는 건 새 태그 때문이 아니라 문서/
                # 섹션이 그냥 끝나버린 경우다.
                finished["_closed_by"] = "문서(섹션) 끝(EOF)"
            tables_total, tables_failed, tables_partial = _finalize_frame(
                finished,
                stack[-1],
                tables_total,
                tables_failed,
                tables_partial,
            )

        text = "\n\n".join(p for p in root["buffer"] if p.strip())

        # 방어: 어떤 경로로든 단독 서로게이트가 남아있으면 UTF-8 인코딩이
        # 깨지므로 여기서 한 번 더 제거한다.
        text = re.sub(r"[\ud800-\udfff]", "", text)

        if not text.strip():
            raise HwpParseError("raw 파서 결과가 비어 있음")

        result = TableParseResult(
            # [수정 1] 완전 실패(<70% 채움) 표가 없으면 success=True로 유지.
            # 부분 복원은 더 이상 실패로 집계하지 않고 tables_partial에 따로 남긴다.
            success=(tables_total == 0 or tables_failed == 0),
            tables_total=tables_total,
            tables_failed=tables_failed,
            tables_partial=tables_partial,
        )

        return text, result


# [수정 4 - 셀 내부 구조 유지] ○/◦/□/■/※/•/▶/▷, (1)(2)식 번호, 가./나./다.
# 같은 한글 순번, ①~⑳ 원문자로 시작하는 줄을 Markdown 리스트("- ...")로
# 바꾼다. 구조가 살아있어야 청킹 후에도 항목이 뭉개지지 않는다.
# 주의: [가-힣]\. 은 "답." 처럼 한 글자 + 마침표로 끝나는 일반 문장도
# 리스트 항목으로 오인할 수 있다 (회귀 가능성 - 응답 하단 요약 참고).
_CELL_STRUCTURE_MARKER = re.compile(
    r"^(?:[○◦□■※•▶▷]|\([0-9]+\)|[0-9]+\.|[가-힣]\.|[①-⑳])\s*"
)


def _join_cell_paragraphs(paragraphs: list) -> str:
    """셀 안 문단 목록을, 구조 마커로 시작하는 줄은 Markdown 리스트로 바꿔 합친다."""

    lines = []

    for paragraph in paragraphs:
        stripped = paragraph.strip()

        if not stripped:
            continue

        match = _CELL_STRUCTURE_MARKER.match(stripped)

        if match:
            content = stripped[match.end() :].strip()
            lines.append(f"- {content}" if content else "-")
        else:
            lines.append(stripped)

    return "\n".join(lines)


def _finalize_frame(
    frame: dict,
    parent: dict,
    tables_total: int,
    tables_failed: int,
    tables_partial: int = 0,
):
    """스택에서 닫히는 frame을 부모 컨텍스트에 반영한다."""

    if frame["type"] == "cell":
        # [수정 4] 단순 "\n".join(...) 대신 구조 마커를 인식하는 join 사용
        text = _join_cell_paragraphs(frame["buffer"])

        if parent["type"] == "table":
            parent["cells"][frame["addr"]] = {
                "text": text,
                "colspan": frame["colspan"],
                "rowspan": frame["rowspan"],
            }
        else:
            parent["buffer"].append(text)

    elif frame["type"] in ("table", "table_pending"):
        tables_total += 1
        # [수정 1] bool 대신 "success"/"partial"/"failed" 3단계 상태를 받는다.
        rendered, status = _render_table(frame)

        if status == "failed":
            tables_failed += 1
        elif status == "partial":
            tables_partial += 1

        parent["buffer"].append(rendered)

        # 표 레벨에 직접 낀 비정상 텍스트(칸이 아닌 문단)는 버리지 않고 뒤에 붙인다
        stray_text = "\n".join(p for p in frame.get("buffer", []) if p.strip())
        if stray_text:
            parent["buffer"].append(stray_text)

    elif frame["type"] == "list":
        text = "\n".join(p for p in frame["buffer"] if p.strip())
        parent["buffer"].append(text)

    return tables_total, tables_failed, tables_partial


def _render_table(frame: dict) -> tuple[str, str]:
    """
    표 frame을 문자열로 렌더링한다.
    2열 표(첫 열이 짧으면)는 key=value, 그 외는 Markdown 표 형태로.

    [수정 1 - 표 성공 판정 완화]
    셀 하나만 비어도 통째로 실패 처리하던 기존 방식은 실제로는 대부분
    채워진 표까지 실패로 깎아내렸다. 채워진 비율(fill_ratio) 기준으로
    success(>=95%) / partial(70~95%) / failed(<70%) 3단계로 나누고,
    실패여도 채워진 내용은 '-'로만 빈 칸을 메워 최대한 유지한다.
    반환 타입이 bool -> str(status)로 바뀌어 호출부(_finalize_frame)도
    함께 수정했다.
    """

    rows, cols = frame.get("rows"), frame.get("cols")

    if not rows or not cols:
        # [진단용] 실패 사유를 구분해서 남긴다 - 재실행 없이 기존
        # cleaned_texts/*.txt를 grep만 해도 "표 레코드 누락"으로 인한
        # 실패가 몇 건인지 바로 셀 수 있다. 무엇 때문에 table_pending이
        # TABLE 레코드를 못 받고 닫혔는지(_closed_by)도 함께 남긴다.
        closed_by = frame.get("_closed_by", "알 수 없음")
        return f"[표 복원 실패: TABLE 레코드 누락 (닫힌 원인: {closed_by})]", "failed"

    grid = [[None] * cols for _ in range(rows)]

    try:
        for (row0, col0), cell in frame["cells"].items():
            for r in range(row0, min(row0 + cell["rowspan"], rows)):
                for c in range(col0, min(col0 + cell["colspan"], cols)):
                    grid[r][c] = cell["text"]
    except Exception as e:  # noqa: BLE001
        # [버그 수정] 원본 코드는 RuntimeError만 잡았는데, 여기서 실제로
        # 날 수 있는 오류(IndexError 등)는 RuntimeError가 아니라서 사실상
        # 한 번도 안 잡혔다. 사용자가 실제 100건 실행에서 동일 패턴의
        # 버그(NotOleFileError 미포착)를 발견해 함께 넓혀 잡는다.
        # [진단용] 실제 예외 종류/메시지를 태그에 남긴다.
        return f"[표 복원 실패: 셀 주소 처리 오류 - {type(e).__name__}: {e}]", "failed"

    total_cells = rows * cols
    filled_cells = sum(1 for row in grid for v in row if v is not None)
    fill_ratio = filled_cells / total_cells if total_cells else 0.0

    # [수정 3 - 병합셀 렌더링 개선] fill_ratio/키-값 판정용 grid는 기존처럼
    # 병합 범위 전체에 텍스트를 복제한 상태를 그대로 쓴다(주소 처리 로직은
    # 건드리지 않음). 화면에 보여줄 display_grid만 별도로 만들어, 병합된
    # 칸 중 origin이 아닌 칸은 빈 문자열로 비운다 - 같은 텍스트가 셀마다
    # 반복 출력되는 것을 막기 위함(진짜 병합 표현은 Markdown 표가 지원하지
    # 않으므로, 빈 칸으로 이어짐을 표시하는 절충안).
    display_grid = [row[:] for row in grid]
    for (row0, col0), cell in frame["cells"].items():
        if cell["rowspan"] <= 1 and cell["colspan"] <= 1:
            continue
        for r in range(row0, min(row0 + cell["rowspan"], rows)):
            for c in range(col0, min(col0 + cell["colspan"], cols)):
                if (r, c) != (row0, col0):
                    display_grid[r][c] = ""

    filled_display = [
        [v if v is not None else "-" for v in row] for row in display_grid
    ]

    # [표현 이원화] 표 형태에 따라 생성(LLM)용 렌더링 방식을 분기한다.
    # - key-value(2열, 짧은 키): 이미 "키 값" 한 줄이라 그대로도 안전
    # - 병합 셀 있음: Markdown 표는 병합을 표현 못 해서(빈 칸으로만 처리)
    #   헤더-값 결합이 애매해짐 -> 행 단위로 "헤더: 값"을 명시하는
    #   row-block 형태로 렌더링
    # - 그 외(단순 격자): 기존처럼 Markdown 표
    # 이 함수가 반환하는 텍스트는 "생성용" 원본이고, 검색/임베딩용 평문은
    # 이후 strip_table_markup()이 이 결과에서 마크업만 제거해 만든다
    # (렌더링을 두 번 하지 않고 한 번의 결과에서 파생시켜 정합성을 보장).
    has_merged_cells = any(
        cell["rowspan"] > 1 or cell["colspan"] > 1 for cell in frame["cells"].values()
    )

    if cols == 2 and _is_keyvalue_table(grid):
        rendered = _render_keyvalue(filled_display)
    elif has_merged_cells:
        # [버그 수정] display_grid는 Markdown 표용으로 병합된 칸 중
        # origin이 아닌 칸을 빈 문자열로 비워둔다(중복 출력 방지 목적).
        # row-block은 "그 행 하나만 봐도 의미가 통해야" 하므로 정반대로,
        # 병합 값이 spanned된 모든 행에 그대로 반복돼야 한다. 그래서
        # 병합을 죽이지 않은 grid(셀 population 단계에서 이미 rowspan/
        # colspan 범위 전체에 텍스트가 복제돼 있음)를 그대로 쓴다.
        filled_full = [[v if v is not None else "-" for v in row] for row in grid]
        rendered = _render_row_block(filled_full)
    else:
        rendered = _render_matrix(filled_display)

    if fill_ratio >= TABLE_FULL_SUCCESS_RATIO:
        return rendered, "success"

    if fill_ratio >= TABLE_PARTIAL_MIN_RATIO:
        return f"[표 부분 복원: 채움비율 {fill_ratio:.0%}]\n" + rendered, "partial"

    # [진단용] 실제 fill_ratio 수치를 태그에 남긴다. 0%대(=거의 다 비어
    # 있는 표, table_pending 프레임 자체가 잘못 열렸을 가능성)인지
    # 60%대(=진짜 아깝게 임계값 미달)인지 구분해서 볼 수 있다.
    return f"[표 복원 실패: 채움비율 {fill_ratio:.0%}]\n" + rendered, "failed"


def _is_keyvalue_table(grid: list) -> bool:
    keys = [row[0] for row in grid if row[0]]
    if not keys:
        return False
    avg_len = sum(len(k) for k in keys) / len(keys)
    return avg_len <= KEYVALUE_MAX_KEY_LENGTH


def _render_keyvalue(grid: list) -> str:
    # [표 마커 완전 제거] "키 = 값"의 "="도 표에서 왔다는 걸 드러내는
    # 마커로 보고 없앤다. 실험 결과 셀 구분자까지 없앤 순수 공백 구분
    # 텍스트가 성능이 더 좋다고 판단되어, 그냥 공백으로 이어붙인다.
    return "\n".join(f"{row[0].strip()} {row[1].strip()}" for row in grid)


def _render_row_block(grid: list) -> str:
    """병합 셀이 있는 표를 '[행 N] 헤더1: 값1 헤더2: 값2 ...' 형태로
    렌더링한다.

    [표현 이원화] Markdown 표는 rowspan/colspan을 표현할 방법이 없어서,
    병합된 칸을 빈 칸으로 남기면(_render_table의 display_grid 처리) 그
    행에서 어떤 값이 어떤 헤더에 속하는지가 다시 애매해진다. 병합 셀이
    있는 표만 이 형태로 바꿔서, 행마다 헤더-값 쌍을 명시적으로 풀어쓴다.
    한 행을 한 줄로 유지해 다른 행과 섞이지 않게 한다.

    시간복잡도: O(행 수 * 열 수)
    """

    header = [cell.strip() for cell in grid[0]]
    lines = []

    for row_idx, row in enumerate(grid[1:], start=1):
        pairs = [
            f"{header[c]}: {row[c].strip()}"
            for c in range(len(header))
            if row[c].strip()
        ]
        if pairs:
            lines.append(f"[행 {row_idx}] " + " ".join(pairs))

    return "\n".join(lines)


def _render_matrix(grid: list) -> str:
    """
    [수정 2 - Markdown 표 출력] 기존 "헤더=값 · 헤더=값" 형태는 사람이
    읽기엔 괜찮지만 임베딩 시 표 구조가 뭉개진다. 청크 검색 품질을 위해
    표준 Markdown 표(|헤더|...|, |---|...|)로 바꾼다. 셀 안에 이미 줄바꿈이
    있으면(예: 수정 4의 Markdown 리스트) Markdown 표 한 줄이 깨지므로
    <br>로 치환한다. 파이프 문자(|)도 이스케이프한다.
    """

    def _escape_cell(value) -> str:
        text = str(value).strip()
        text = text.replace("|", "\\|")
        text = text.replace("\n", "<br>")
        return text

    header = [_escape_cell(cell) for cell in grid[0]]
    lines = [
        "|" + "|".join(header) + "|",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]

    for row in grid[1:]:
        cells = [_escape_cell(row[c]) for c in range(len(header))]
        lines.append("|" + "|".join(cells) + "|")

    return "\n".join(lines)


# ============================================================
# 10. HWP 통합 디스패처
# ============================================================


def extract_hwp_document(path: Path) -> ExtractionResult:
    """
    HWP_EXTRACTION_METHODS 순서대로 시도한다.
    hwp_raw는 한글 비율 검증만 통과하면 채택한다 - 표 일부가 실패해도
    표를 아예 포기하는 hwp5txt/libreoffice보다 낫다고 판단하기 때문.
    (표 전부/일부 성공 여부는 table_parse_success로 별도 기록)

    성공한 문서라도, 먼저 시도했다가 실패한 방법이 있으면 그 에러를
    attempted_errors에 남긴다. hwp_raw가 조용히 실패하고 hwp5txt로
    넘어가는 경우(표 손실)를 추적하기 위함.
    """

    errors = {}

    for i, method in enumerate(HWP_EXTRACTION_METHODS):
        fallback_used = i > 0

        try:
            if method == "hwp_raw":
                text, table_result = extract_text_hwp_raw_structured(path)

                if not _check_hangul(text):
                    raise HwpParseError("한글 비율 미달")

                return ExtractionResult(
                    text=text,
                    extractor="hwp_raw",
                    table_parse_success=table_result.success,
                    tables_total=table_result.tables_total,
                    tables_failed=table_result.tables_failed,
                    error_reason=None,
                    fallback_used=fallback_used,
                    attempted_errors=dict(errors),
                    tables_partial=table_result.tables_partial,  # [수정 1]
                )

            elif method == "hwp5txt":
                text = extract_text_hwp5txt(path)

                if not _check_hangul(text):
                    raise HwpParseError("한글 비율 미달")

                return ExtractionResult(
                    text=text,
                    extractor="hwp5txt",
                    table_parse_success=False,
                    tables_total=0,
                    tables_failed=0,
                    error_reason=None,
                    fallback_used=fallback_used,
                    attempted_errors=dict(errors),
                )

            elif method == "libreoffice":
                text = extract_text_libreoffice(path)

                if not _check_hangul(text):
                    raise HwpParseError("한글 비율 미달")

                return ExtractionResult(
                    text=text,
                    extractor="libreoffice",
                    table_parse_success=False,
                    tables_total=0,
                    tables_failed=0,
                    error_reason=None,
                    fallback_used=fallback_used,
                    attempted_errors=dict(errors),
                )

            else:
                raise ValueError(f"알 수 없는 추출 방법: {method}")

        except Exception as e:  # noqa: BLE001
            # [버그 수정] HwpParseError(Exception 상속)와 olefile의
            # NotOleFileError(OSError 상속) 모두 RuntimeError가 아니라서
            # 기존 "except RuntimeError"로는 못 잡았다. 그러면 hwp_raw가
            # OLE2가 아닌 파일(HWPX 오분류, 손상 파일 등)에서 죽었을 때
            # hwp5txt/libreoffice 폴백을 타지 못하고 예외가 extract_hwp_document
            # 밖으로 그대로 새어나가 파이프라인 전체가 멈췄다(사용자가 100건
            # 실행 중 NotOleFileError로 직접 재현). 폴백 체인이 실제로
            # 작동하려면 여기서 넓게 잡아야 한다.
            errors[method] = str(e)

    return ExtractionResult(
        text="",
        extractor=None,
        table_parse_success=False,
        tables_total=0,
        tables_failed=0,
        error_reason="; ".join(f"{k}={v}" for k, v in errors.items()),
        fallback_used=True,
        attempted_errors=dict(errors),
    )


# ============================================================
# 11. PDF 추출 - 본문/표 분리
# ============================================================


def _bbox_inside(word_bbox: tuple, table_bboxes: list) -> bool:
    x0, top, x1, bottom = word_bbox
    for bx0, btop, bx1, bbottom in table_bboxes:
        if x0 >= bx0 and x1 <= bx1 and top >= btop and bottom <= bbottom:
            return True
    return False


# [수정 5 - PDF 표 추출 강화] find_tables()를 선(lines)/텍스트(text) 두
# 전략으로 각각 시도해, 셀이 더 많이 채워진 쪽을 채택한다. 페이지 단위로
# 전략 하나만 고르고 두 결과를 합치지 않으므로, 같은 표가 두 번 추출되는
# 중복 문제는 생기지 않는다.
_PDF_TABLE_STRATEGIES = [
    {"vertical_strategy": "lines", "horizontal_strategy": "lines"},
    {"vertical_strategy": "text", "horizontal_strategy": "text"},
]


def _find_best_pdf_tables(page) -> list:
    """두 표 탐지 전략 중, 채워진 셀 수가 더 많은 결과를 반환한다."""

    best_tables, best_filled = [], -1

    for settings in _PDF_TABLE_STRATEGIES:
        try:
            tables = page.find_tables(table_settings=settings)
        except Exception:  # noqa: BLE001, S112
            # [버그 수정] pdfplumber/pdfminer가 던지는 파싱 오류도
            # RuntimeError가 아닐 수 있다. 한 전략이 실패해도 다른 전략은
            # 계속 시도해야 하므로 넓게 잡는다.
            continue

        filled = 0
        for table in tables:
            grid = table.extract() or []
            filled += sum(1 for row in grid for cell in row if cell)

        if filled > best_filled:
            best_filled, best_tables = filled, tables

    return best_tables


def extract_pdf_document(path: Path) -> ExtractionResult:
    """
    pdfplumber.find_tables()로 표 영역을 먼저 찾아 본문에서 제외하고,
    표는 별도로 렌더링해 문서 끝에 붙인다. (같은 표 중복 방지)
    """

    try:
        body_parts = []
        table_blocks = []
        tables_total = 0
        tables_failed = 0

        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                found_tables = _find_best_pdf_tables(page)  # [수정 5]
                table_bboxes = [t.bbox for t in found_tables]

                words = page.extract_words()
                body_words = [
                    w
                    for w in words
                    if not _bbox_inside(
                        (w["x0"], w["top"], w["x1"], w["bottom"]),
                        table_bboxes,
                    )
                ]
                body_parts.append(" ".join(w["text"] for w in body_words))

                for t in found_tables:
                    tables_total += 1
                    grid = t.extract()

                    if not grid or not grid[0]:
                        tables_failed += 1
                        table_blocks.append("[표 복원 실패] 빈 표")
                        continue

                    grid = [[cell or "" for cell in row] for row in grid]

                    if len(grid[0]) == 2 and _is_keyvalue_table(grid):
                        table_blocks.append(_render_keyvalue(grid))
                    else:
                        table_blocks.append(_render_matrix(grid))

        text = "\n\n".join(body_parts) + "\n\n" + "\n\n".join(table_blocks)

        if not text.strip():
            raise HwpParseError("pdfplumber 결과가 비어 있음")

        return ExtractionResult(
            text=text,
            extractor="pdfplumber",
            table_parse_success=(tables_failed == 0),
            tables_total=tables_total,
            tables_failed=tables_failed,
            error_reason=None,
            fallback_used=False,
            attempted_errors={},
        )

    except Exception as e:  # noqa: BLE001
        # [버그 수정] pdfplumber.open() 등에서 나는 실제 예외(OSError,
        # PDFSyntaxError 등)는 RuntimeError가 아니어서 기존 except로는
        # 못 잡고 그대로 새어나갔다.
        return ExtractionResult(
            text="",
            extractor=None,
            table_parse_success=False,
            tables_total=0,
            tables_failed=0,
            error_reason=str(e),
            fallback_used=False,
            attempted_errors={"pdfplumber": str(e)},
        )


# ============================================================
# 12. 확장자 통합 디스패처
# ============================================================


def extract_document(path: Path) -> ExtractionResult:

    suffix = path.suffix.lower()

    if suffix == ".hwp":
        return extract_hwp_document(path)

    if suffix == ".pdf":
        return extract_pdf_document(path)

    return ExtractionResult(
        text="",
        extractor=None,
        table_parse_success=False,
        tables_total=0,
        tables_failed=0,
        error_reason=f"지원하지 않는 확장자: {suffix}",
        fallback_used=False,
        attempted_errors={},
    )


# ============================================================
# 13. 텍스트 정제
# ============================================================


def clean_text_verbose(
    text: str,
    remove_angle_brackets: bool = False,
    remove_bullet_dot: bool = False,
    boilerplate_lines: set | None = None,
) -> tuple[str, list[str]]:
    """
    Args:
        text: 정제할 원문.
        remove_angle_brackets: <,> 문자 제거 여부.
        remove_bullet_dot: • 문자 제거 여부.
        boilerplate_lines: 제거할 boilerplate 라인 집합. None이면 기본값 사용.

    Returns:
        tuple[str, list[str]]: (정제된 텍스트, 적용된 정제 작업 설명 목록).
    """

    text = re.sub(r"[\ud800-\udfff]", "", text)

    if boilerplate_lines is None:
        boilerplate_lines = get_boilerplate_lines()

    notes = []
    page_number_removed = 0
    boilerplate_removed = 0

    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        if PATTERN_PAGE_NUMBER_DASH.match(stripped):
            page_number_removed += 1
            continue

        normalized = re.sub(r"\s+", " ", stripped)

        if boilerplate_lines and normalized in boilerplate_lines:
            boilerplate_removed += 1
            continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    if page_number_removed:
        notes.append(f"페이지번호 라인 {page_number_removed}개 제거")

    if boilerplate_removed:
        notes.append(f"boilerplate 라인 {boilerplate_removed}개 제거")

    tag_count = len(PATTERN_HWP_TAG_RESIDUE.findall(text))
    text = PATTERN_HWP_TAG_RESIDUE.sub("", text)
    if tag_count:
        notes.append(f"<표>/<그림> 태그 잔재 {tag_count}개 제거")

    placeholder_count = len(PATTERN_PLACEHOLDER.findall(text))
    text = PATTERN_PLACEHOLDER.sub("", text)
    if placeholder_count:
        notes.append(f"플레이스홀더(ㅇㅇㅇ 등) {placeholder_count}개 제거")

    empty_paren_count = len(PATTERN_EMPTY_PAREN.findall(text))
    text = PATTERN_EMPTY_PAREN.sub("", text)
    if empty_paren_count:
        notes.append(f"빈 괄호 {empty_paren_count}개 제거")

    control_count = len(PATTERN_CONTROL_CHAR.findall(text))
    text = PATTERN_CONTROL_CHAR.sub("", text)
    if control_count:
        notes.append(f"제어문자 {control_count}개 제거")

    # [수정 6 - 인코딩 노이즈 제거] 괄호로 둘러싸인 런(정상 한자 병기)은
    # 보존하고, 그 외 런만 통째로 제거한다.
    mojibake_count = sum(
        1
        for mt in PATTERN_MOJIBAKE.finditer(text)
        if not _is_parenthesized_mojibake_match(mt)
    )
    text = PATTERN_MOJIBAKE.sub(
        lambda mt: mt.group(0) if _is_parenthesized_mojibake_match(mt) else "",
        text,
    )
    if mojibake_count:
        notes.append(f"mojibake 추정 문자 묶음 {mojibake_count}개 제거")

    if remove_angle_brackets:
        angle_count = len(re.findall(r"[<>]", text))
        text = re.sub(r"[<>]", "", text)
        if angle_count:
            notes.append(f"<,> 문자 {angle_count}개 제거")

    if remove_bullet_dot:
        bullet_count = text.count("•")
        text = text.replace("•", "")
        if bullet_count:
            notes.append(f"• 문자 {bullet_count}개 제거")

    text = PATTERN_LONG_SPACE.sub(" ", text)
    text = PATTERN_LONG_NEWLINE.sub("\n\n", text)

    return text.strip(), notes


def clean_text(
    text: str,
    remove_angle_brackets: bool = False,
    remove_bullet_dot: bool = False,
    boilerplate_lines: set | None = None,
) -> str:
    """
    구조 마커(숫자_점, 가., ○, □, ※, ◦, (1), 로마숫자, •)는 절대 건드리지 않는다.
    내부적으로 clean_text_verbose를 감싸며, 작업 내역(warnings)만 버린다.

    Args:
        text: 정제할 원문.
        remove_angle_brackets: <,> 문자 제거 여부.
        remove_bullet_dot: • 문자 제거 여부.
        boilerplate_lines: 제거할 boilerplate 라인 집합. None이면 기본값 사용.

    Returns:
        str: 정제된 텍스트.
    """

    cleaned, _ = clean_text_verbose(
        text,
        remove_angle_brackets=remove_angle_brackets,
        remove_bullet_dot=remove_bullet_dot,
        boilerplate_lines=boilerplate_lines,
    )
    return cleaned


# ============================================================
# 13-1. 표 구조 제거 (Markdown 표 -> 셀 구분자만 남긴 평문)
# ============================================================
#
# 실험 결과 표를 Markdown 그리드(|헤더|---|값|)로 살려서 임베딩하는 것보다
# 격자(테두리 파이프, 구분행)를 걷어내고 셀 값만 구분자로 이어붙인 평문이
# 검색 성능이 더 좋다고 판단되어 도입. _render_table/_render_matrix는
# 그대로 두고(성공/부분/실패 판정과 report 집계는 fill_ratio 기준으로
# 여전히 정확하게 동작), 다 만들어진 텍스트를 마지막에 한 번 더 정리하는
# 방식이라 표 판정 로직과는 완전히 분리되어 있다.

PATTERN_TABLE_ESCAPED_MARKUP = re.compile(r"\\([|-])")  # 중첩 표의 \| \- 를 푼다
# [수정] 참고 코드의 원래 패턴은 "[표 파싱...]"을 찾는데, 실제 이 파이프라인이
# 남기는 진단 태그는 "[표 복원 실패: ...]"/"[표 부분 복원: ...]"이라 문자열이
# 다르다. 그대로 쓰면 아무것도 안 지워지므로 실제 태그 문자열에 맞춰 고쳤다.
PATTERN_TABLE_DEBUG_TAG = re.compile(r"\[표 (?:복원 실패|부분 복원)[^\]]{0,200}\]")
PATTERN_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")  # |셀|셀|  (구분행/빈행 포함)
PATTERN_TABLE_SEPARATOR_ROW = re.compile(r"^\s*\|(\s*:?-{2,}:?\s*\|)+\s*$")  # |---|---|
PATTERN_TABLE_EMPTY_ROW = re.compile(r"^\s*\|(\s*\|)*\s*$")  # |||
PATTERN_TABLE_BLANK_RUN = re.compile(r"\n{3,}")
PATTERN_TABLE_CELL_SPLIT = re.compile(
    r"(?<!\\)\|"
)  # 이스케이프 안 된 |만 셀 구분자로 인식
PATTERN_ROW_BLOCK_LINE = re.compile(
    r"\[행 \d+\]\s*"
)  # _render_row_block 결과 태그. ^ 없음: 줄 중간에도 나타날 수 있어 .search()/.sub()로 찾는다. .match() 사용처는 항상 위치 0에서만 보므로 영향 없다.
# 앞이 숫자가 아니거나(헤더-값 구분자) 뒤가 숫자가 아닌 콜론만 고른다.
# 15:00 처럼 양쪽이 숫자인 것만 살아남는다.
PATTERN_HEADER_COLON = re.compile(r"(?<!\d):|:(?!\d)")


def _split_table_row(line: str) -> list[str]:
    """Markdown 표 한 줄을 셀 리스트로 쪼갠다.

    이스케이프된 파이프(`\\|`)는 셀 내용(중첩 표)으로 보고 구분자로 쓰지
    않는다. 셀 단위로 다시 `\\|`, `\\-` 이스케이프를 풀고 `<br>`을 칸 내부
    줄바꿈이었던 자리이므로 공백으로 되돌린다.

    시간복잡도: O(줄 길이)
    """

    inner = re.sub(r"^\s*\|", "", line.strip())
    inner = re.sub(r"\|\s*$", "", inner)

    cells = []
    for raw_cell in PATTERN_TABLE_CELL_SPLIT.split(inner):
        cell = PATTERN_TABLE_ESCAPED_MARKUP.sub(r"\1", raw_cell)
        cell = cell.replace("<br>", " ")
        cells.append(cell.strip())

    return cells


def _flatten_table_block(lines: list[str]) -> list[str]:
    """연속된 Markdown 표 줄 묶음을 '헤더 값 헤더 값 ...' 행 단위 평문으로
    바꾼다.

    [핵심] 기존 방식은 헤더 줄과 값 줄을 각각 독립적으로 공백 구분
    텍스트로 바꿔서, 헤더 "컬럼1 컬럼2 컬럼3"과 값 "1-1 2-1 3-1"이 서로
    다른 줄로 떨어져 나왔다. 청크가 헤더 줄과 값 줄 사이에서 잘리거나
    표가 여러 개 이어지면 어느 값이 어느 헤더의 것인지 위치 추론에만
    의존해야 해서 의미가 끊길 위험이 있었다.

    이 함수는 헤더 행(첫 줄)의 셀과 각 데이터 행의 셀을 같은 열 위치로
    짝지어 "헤더 값" 쌍을 만들고, 한 데이터 행의 모든 쌍을 한 줄에
    이어붙인다. 격자 기호(|, ---)는 사라지지만 헤더-값 결합은 같은 줄
    안에서 유지된다. 행 간 줄바꿈은 그대로 둬서(수정 전과 동일하게) 행
    하나가 다른 행과 한 줄로 섞이지는 않는다.

    시간복잡도: O(행 수 * 열 수)
    """

    data_lines = [
        line
        for line in lines
        if not PATTERN_TABLE_SEPARATOR_ROW.match(line)
        and not PATTERN_TABLE_EMPTY_ROW.match(line)
    ]

    rows = [_split_table_row(line) for line in data_lines]
    rows = [row for row in rows if any(cell for cell in row)]

    if not rows:
        return []

    header, *data_rows = rows

    if not data_rows:
        # 헤더 행만 있고 데이터 행이 없으면(예: 표가 잘려 헤더만 남은 경우)
        # 헤더라도 살려서 정보 손실을 막는다.
        return [" ".join(cell for cell in header if cell)]

    flattened_rows = []

    for row in data_rows:
        pairs = []
        for col_idx, value in enumerate(row):
            if not value:
                continue
            head = header[col_idx] if col_idx < len(header) else ""
            pairs.append(f"{head} {value}".strip() if head else value)

        if pairs:
            flattened_rows.append(" ".join(pairs))

    return flattened_rows


def strip_table_markup(text: str) -> str:
    """Markdown 표 문법을 없애되, 헤더-값 결합은 같은 줄에 유지한 채 순수
    공백 구분 텍스트로 만든다.

    [실험 결과 반영] 격자(테두리 파이프, 구분행)를 걷어내고 셀 값을
    공백으로 이어붙인 평문이 Markdown 그리드보다 검색 성능이 좋다는 것은
    기존 실험으로 확인됨. 다만 헤더 줄과 값 줄을 따로따로 공백 텍스트로
    바꾸면 헤더-값 대응이 줄 위치 추론에만 의존하게 되는 문제가 있어,
    표 블록을 통째로 파싱해 각 데이터 행에 헤더를 인라인으로 묶어주는
    방식으로 바꿨다(`_flatten_table_block` 참고). 값이 어느 헤더에
    속하는지가 텍스트 자체에 남기 때문에 청크가 표 중간에서 잘려도 행
    단위로는 의미가 보존된다.

    이 함수는 `|`로 둘러싸인 표 형태 줄과 `_render_row_block`이 만든
    "[행 N] 헤더: 값 ..." 줄만 건드리므로, 그 외(`|`도 `[행 N]` 태그도
    없는) 계층/불릿 마커(□■○◦※•▶▷), 목차/섹션 제목(Ⅰ.Ⅱ.Ⅲ.), 활동/과업
    번호(Activity N.N.N), 법률/계약 조항(제N조, ①②...)은 전혀 영향을
    받지 않는다. `_render_keyvalue`가 만든 2열 표(이미 "키 값" 형태라
    `|`도 `[행 N]`도 없음)도 손대지 않는다 - 이미 헤더-값이 한 줄에 묶여
    있어 그대로 둬도 안전하다.

    Args:
        text: `_render_table`이 표 형태별로 렌더링을 마친, 생성(LLM)용
            원본 텍스트(`clean_text_verbose`까지 끝난 상태).

    Returns:
        str: 표 마크업을 제거하고, 데이터 행마다 "헤더 값" 쌍을 이어붙인
        검색/임베딩용 평문.
    """

    # 표 진단 태그(실패/부분복원)는 report/warnings에서 이미 집계했으니
    # 본문(임베딩 대상)에는 남길 필요가 없다.
    text = PATTERN_TABLE_DEBUG_TAG.sub("", text)

    lines = text.split("\n")
    output_lines: list[str] = []
    table_buffer: list[str] = []

    def _flush_table_buffer() -> None:
        if table_buffer:
            output_lines.extend(_flatten_table_block(table_buffer))
            table_buffer.clear()

    for line in lines:
        if PATTERN_TABLE_ROW.match(line):
            table_buffer.append(line)
            continue

        _flush_table_buffer()

        if PATTERN_ROW_BLOCK_LINE.search(line):
            # "...[행 N] 헤더1: 값1 [행 M] 헤더2: 값2..." -> "...헤더1 값1 헤더2 값2..."
            # 태그가 줄 시작이 아니라 중간에도 나타날 수 있어(배점 산식
            # 문장 등에 섞여 들어간 사례) sub으로 위치 상관없이 전부
            # 걷어낸다. 헤더-값 구분 콜론도 같이 걷어내되, 앞뒤가 모두
            # 숫자인 콜론(15:00, 13:30 …)은 시각이므로 남긴다.
            content = PATTERN_ROW_BLOCK_LINE.sub("", line)
            content = PATTERN_HEADER_COLON.sub("", content)
            content = re.sub(r"[ \t]{2,}", " ", content).strip()
            output_lines.append(content)
            continue

        # 표 밖(그 줄에 |가 없는 경우)의 <br>은 원래 의도대로 줄바꿈으로 되돌린다.
        output_lines.append(line.replace("<br>", "\n"))

    _flush_table_buffer()

    text = "\n".join(output_lines)

    # 표만 있다가 비어버린 줄(공백만 남은 줄) 제거
    text = re.sub(r"^[ \t]*$", "", text, flags=re.MULTILINE)
    text = PATTERN_TABLE_BLANK_RUN.sub("\n\n", text)

    return text.strip()


PATTERN_TABLE_MARKUP_LEFTOVERS = {
    "잔존_파이프": re.compile(r"\|"),
    "잔존_br": re.compile(r"<br>"),
    "잔존_구분행": re.compile(r"^\s*\|?\s*:?-{2,}", re.MULTILINE),
    "잔존_진단태그": re.compile(r"\[표 (?:복원 실패|부분 복원)"),
    "잔존_행블록태그": re.compile(r"\[행 \d+\]"),
}


def verify_no_table_markup(
    df: pd.DataFrame, text_col: str = "clean_text"
) -> pd.DataFrame:
    """strip_table_markup이 실제로 다 걷어냈는지 QA용으로 검증한다.

    run_pipeline이 반환한 df(또는 write_jsonl에 쓰인 것과 같은 df)를 그대로
    넣으면 된다. text_col을 "clean_text"(문서 단위 df)나 "page_content"
    (청크 단위 df)로 바꿔가며 쓸 수 있다.

    Args:
        df: 검사할 DataFrame. text_col 컬럼과, 있으면 "파일명"/"source"
            컬럼을 사용한다.
        text_col: 검사할 텍스트가 담긴 컬럼명.

    Returns:
        pd.DataFrame: 표 마크업이 남아있는 문서만 담은 표(사유별 개수 포함).
        비어 있으면 전부 통과한 것.
    """

    rows = []

    for _, row in df.iterrows():
        text = row.get(text_col) or ""
        hits = {
            name: len(pattern.findall(text))
            for name, pattern in PATTERN_TABLE_MARKUP_LEFTOVERS.items()
        }

        if any(hits.values()):
            identifier = row.get("파일명", row.get("source"))
            rows.append({"파일명": identifier, **hits})

    return pd.DataFrame(rows)


# ============================================================
# 14. 메타데이터 추출
# ============================================================


def extract_metadata(text: str) -> dict:
    """
    FIELD_ALIASES의 라벨을 찾아, 다음 구조 마커(□■○◦※) 또는 줄바꿈 전까지를
    값으로 추출한다. 정제 전 원문(구조 마커가 살아있는 상태)에 대해 실행할 것.
    """

    # 타입 힌트 명시: 힌트 없이 {k: None for k in ...}만 쓰면 정적 타입
    # 검사기(Pylance 등)가 dict[str, None]으로 추론해, 뒤에서 str 값을
    # 넣을 때 "None에 str을 대입할 수 없다"는 오탐 경고를 띄운다. 실행에는
    # 영향이 없지만(Python은 동적 타입), 경고 자체를 없애기 위해 명시한다.
    metadata: dict[str, str | None] = {field_name: None for field_name in FIELD_ALIASES}

    for field_name, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            pattern = re.compile(
                rf"(?:[□■○◦※]\s*\(\s*{re.escape(alias)}\s*\)|{re.escape(alias)})"
                rf"\s*[:：]?\s*"
            )

            value = None

            # [버그 수정] 원래는 pattern.search()로 문서 전체에서 "첫 매치
            # 하나"만 보고, 거기서 값이 안 나오면 그 별칭은 바로 포기했다.
            # RFP 문서는 앞쪽에 "○ 사업기간 / ○ 참가자격" 같은 목차·개요성
            # 불릿이 먼저 나오고 실제 값은 훨씬 뒤 본문에 있는 경우가
            # 흔한데, 그 목차 줄에서 걸려 항상 빈 값으로 끝났다(100건 전부
            # 0%로 나온 원인). finditer로 같은 별칭의 모든 등장 위치를
            # 순회하며, 값이 실제로 채워지는 첫 매치를 채택하도록 바꿨다.
            #
            # [개선] 콜론(:/：)이 실제로 붙어 있는 매치("사업기간: 값")는
            # "라벨: 값" 표기일 가능성이 높고, 콜론 없는 매치는 목차/헤더일
            # 가능성이 높다. 콜론 있는 매치를 먼저 시도해, 목차 줄 근처의
            # 무관한 서술문을 값으로 잘못 채택하는 경우를 줄인다.
            matches = list(pattern.finditer(text))
            with_colon = [mt for mt in matches if re.search(r"[:：]", mt.group())]
            without_colon = [mt for mt in matches if mt not in with_colon]

            for match in with_colon + without_colon:
                start = match.end()
                boundary = METADATA_VALUE_BOUNDARY.search(text, start)
                end = (
                    boundary.start()
                    if boundary
                    else min(len(text), start + METADATA_VALUE_MAX_LEN)
                )

                # [수정 7] \n이 더 이상 경계가 아니므로, 여기서 최대
                # METADATA_VALUE_MAX_LINES줄까지만 모아 값으로 쓴다. 빈 줄이
                # 나오면 값이 끝난 것으로 보고 더 이어붙이지 않는다.
                raw_value = text[start:end]
                value_lines = []
                for line in raw_value.splitlines():
                    stripped_line = line.strip()
                    if not stripped_line:
                        break
                    if value_lines and _METADATA_LINE_BOUNDARY.match(stripped_line):
                        break
                    value_lines.append(stripped_line)
                    if len(value_lines) >= METADATA_VALUE_MAX_LINES:
                        break

                candidate = " ".join(value_lines).strip(" :：\t")

                if candidate:
                    value = candidate
                    break

            if value:
                metadata[field_name] = value
                break

    return metadata


# ============================================================
# 15. 문서 1건 처리
# ============================================================


def process_document(path: Path) -> dict:

    extraction = extract_document(path)

    base = {
        "파일명": path.name,
        "extractor": extraction.extractor,
        "table_parse_success": extraction.table_parse_success,
        "tables_total": extraction.tables_total,
        "tables_failed": extraction.tables_failed,
        "tables_partial": extraction.tables_partial,  # [수정 1]
        "fallback_used": extraction.fallback_used,
        "error_reason": extraction.error_reason,
        # hwp_raw가 채택되지 못한 이유 (성공한 문서라도 조용한 폴백을 추적하기 위함)
        "hwp_raw_skipped_reason": extraction.attempted_errors.get("hwp_raw"),
        "clean_text": "",
        "clean_text_for_generation": "",
        "_warnings": [],
    }
    base.update({k: None for k in FIELD_ALIASES})

    if not extraction.text:
        return base

    clean, warnings_list = clean_text_verbose(extraction.text)

    # [표현 이원화] clean은 _render_table이 표 형태별로(단순 격자는
    # Markdown, 병합 셀은 row-block, key-value는 그대로) 이미 구조화해
    # 놓은 상태 - 이걸 그대로 생성(LLM)용으로 쓴다. 검색/임베딩용은 여기서
    # 마크업만 걷어낸 평문을 별도로 만든다. 표 성공/부분/실패 판정과 경고
    # 집계(위 tables_failed/tables_partial)는 fill_ratio 기준 그대로 유지.
    clean_for_generation = clean
    clean_for_embedding = strip_table_markup(clean)

    if extraction.tables_failed > 0:
        warnings_list.append(
            f"표 {extraction.tables_total}개 중 {extraction.tables_failed}개 복원 실패"
        )

    # [수정 1] 부분 복원된 표도 경고로 남겨 검수 시 눈에 띄게 한다.
    if extraction.tables_partial > 0:
        warnings_list.append(
            f"표 {extraction.tables_total}개 중 {extraction.tables_partial}개 부분 복원"
        )

    base["clean_text"] = clean_for_embedding
    base["clean_text_for_generation"] = clean_for_generation
    base["_warnings"] = warnings_list
    base.update(extract_metadata(extraction.text))

    return base


# ============================================================
# 16. 원본 메타데이터 CSV 병합
# ============================================================


def load_original_metadata(csv_path: Path) -> pd.DataFrame:
    """
    프로젝트 시작 시 받은 원본 메타데이터 CSV를 읽는다.
    컬럼명에 공백이 섞여 있어도("공고 번호" == "공고번호") 정규화해서 매칭하고,
    ORIGINAL_METADATA_COLUMNS 중 실제로 없는 컬럼은 에러 없이 건너뛴다.

    Args:
        csv_path: 원본 메타데이터 CSV 경로.

    Returns:
        pd.DataFrame: 실제로 매칭된 컬럼만으로 구성된 DataFrame(파일명 포함).
    """

    df = pd.read_csv(csv_path)

    normalized_to_actual = {col.replace(" ", ""): col for col in df.columns}

    rename_map = {}
    missing = []

    for canonical in ORIGINAL_METADATA_COLUMNS:
        actual = normalized_to_actual.get(canonical)
        if actual:
            rename_map[actual] = canonical
        else:
            missing.append(canonical)

    if missing:
        print(f"원본 메타데이터 CSV에 없는 컬럼(건너뜀): {missing}")

    df = df.rename(columns=rename_map)

    available = [c for c in ORIGINAL_METADATA_COLUMNS if c in df.columns]

    return df[available]


def _normalize_filename(name) -> str:
    """
    유니코드 정규화(NFC) + 공백 제거 + 확장자 제거.
    한글 완성형/자모분리형(NFC/NFD) 차이나 확장자 유무 차이로
    동일한 파일이 다른 문자열로 취급되는 것을 막기 위함.
    """

    if not isinstance(name, str):
        return name

    normalized = unicodedata.normalize("NFC", name.strip())

    return re.sub(r"\.(hwp|pdf)$", "", normalized, flags=re.IGNORECASE)


def _match_by_common_prefix(
    unmatched_keys: list,
    candidate_keys: list,
    min_prefix_len: int = 15,
) -> dict:
    """
    긴 제목이 서로 다른 지점에서 잘려 파일명이 된 경우를 위한 2차 매칭.
    공통 접두사가 min_prefix_len자 이상이면 같은 문서로 간주한다.

    Args:
        unmatched_keys: 1차 매칭에 실패한 우리 쪽 병합키 목록.
        candidate_keys: 1차 매칭에 쓰이지 않은 원본 CSV 병합키 목록.
        min_prefix_len: 같은 문서로 판단할 최소 공통 접두사 길이.

    Returns:
        dict: {우리쪽키: 원본CSV키} 매칭 결과.
    """

    matches = {}
    remaining_candidates = list(candidate_keys)

    for df_key in unmatched_keys:
        best_match, best_len = None, 0

        for meta_key in remaining_candidates:
            prefix_len = len(os.path.commonprefix([df_key, meta_key]))

            if prefix_len > best_len:
                best_len, best_match = prefix_len, meta_key

        if best_match and best_len >= min_prefix_len:
            matches[df_key] = best_match
            remaining_candidates.remove(best_match)

    return matches


def merge_original_metadata(
    df: pd.DataFrame,
    original_metadata_csv: Path,
    fuzzy_min_prefix_len: int = 15,
) -> pd.DataFrame:
    """
    추출 결과 df에 원본 메타데이터 CSV를 파일명 기준으로 병합한다.
    1차: 유니코드 정규화 + 확장자 무시 정확 매칭.
    2차: 실패한 것만 공통 접두사 기반으로 재시도(제목이 서로 다른
    지점에서 잘려 파일명이 된 경우 대응). 어떤 방식으로 매칭됐는지
    '메타매칭방식' 컬럼에 남긴다.
    """

    meta_df = load_original_metadata(original_metadata_csv)

    df = df.copy()
    meta_df = meta_df.copy()

    df["_병합키"] = df["파일명"].map(_normalize_filename)
    meta_df["_병합키"] = meta_df["파일명"].map(_normalize_filename)

    meta_df.set_index("_병합키")

    check_col = next(
        (c for c in meta_df.columns if c not in ("파일명", "_병합키")),
        None,
    )

    exact_matched_keys = set(df["_병합키"]) & set(meta_df["_병합키"])

    unmatched_df_keys = [k for k in df["_병합키"] if k not in exact_matched_keys]
    unused_meta_keys = [k for k in meta_df["_병합키"] if k not in exact_matched_keys]

    fuzzy_map = _match_by_common_prefix(
        unmatched_df_keys,
        unused_meta_keys,
        fuzzy_min_prefix_len,
    )

    df["_메타키"] = df["_병합키"].map(lambda k: fuzzy_map.get(k, k))
    df["메타매칭방식"] = df["_병합키"].map(
        lambda k: (
            "fuzzy"
            if k in fuzzy_map
            else ("exact" if k in exact_matched_keys else None)
        )
    )

    merged = df.merge(
        meta_df.drop(columns=["파일명"]),
        left_on="_메타키",
        right_on="_병합키",
        how="left",
        suffixes=("", "_원본"),
    )
    merged = merged.drop(
        columns=["_병합키", "_메타키", "_병합키_원본"], errors="ignore"
    )

    if check_col:
        matched = merged[check_col].notna().sum()
        fuzzy_count = (merged["메타매칭방식"] == "fuzzy").sum()
        print(
            f"원본 메타데이터 병합: {matched}/{len(merged)}건 매칭"
            f" (정확 {matched - fuzzy_count}건 + 근사 {fuzzy_count}건)"
        )

        if matched < len(merged):
            unmatched = merged.loc[merged[check_col].isna(), "파일명"].tolist()
            print(f"  끝까지 매칭 안 된 파일명({len(unmatched)}건): {unmatched}")

            used_meta_keys = exact_matched_keys | set(fuzzy_map.values())
            remaining_meta_keys = [
                k for k in meta_df["_병합키"] if k not in used_meta_keys
            ]
            meta_key_to_name = dict(zip(meta_df["_병합키"], meta_df["파일명"]))

            print(
                "\n  [근사 후보 제안] (참고용 - 자동 반영 안 됨, 직접 확인 후 CSV 수정 권장)"
            )

            for name in unmatched:
                key = _normalize_filename(name)
                candidates = difflib.get_close_matches(
                    key,
                    remaining_meta_keys,
                    n=1,
                    cutoff=0.4,
                )

                if candidates:
                    ratio = difflib.SequenceMatcher(None, key, candidates[0]).ratio()
                    print(
                        f"    {name!r}\n"
                        f"      → 후보: {meta_key_to_name[candidates[0]]!r}"
                        f" (유사도 {ratio:.0%})"
                    )
                else:
                    print(
                        f"    {name!r}\n      → 후보 없음 (원본 CSV에 없는 문서일 수 있음)"
                    )

    return merged


# ============================================================
# 17. 팀 공유 JSON 스키마 변환
# ============================================================


def build_doc_schema_record(row: dict) -> dict:
    """
    한 문서(merge_original_metadata까지 끝난 df의 한 행)를
    {meta:{...}, text, chars, extractor, warnings} 스키마로 변환한다.
    원본 CSV 필드가 없는 문서(병합 실패)는 meta의 해당 값이 None으로 남는다.

    Args:
        row: 최종 df의 한 행을 dict로 변환한 것 (DataFrame.to_dict() 등).

    Returns:
        dict: meta/text/chars/extractor/warnings 구조의 레코드.
    """

    def _clean(value):
        return None if pd.isna(value) else value

    notice_no = _clean(row.get("공고번호"))
    revision = _clean(row.get("공고차수"))

    doc_id = (
        f"{notice_no}-{int(revision)}"
        if notice_no is not None and revision is not None
        else None
    )

    text = row.get("clean_text") or ""

    return {
        "meta": {
            "doc_id": doc_id,
            "notice_no": notice_no,
            "title": _clean(row.get("사업명")),
            "agency": _clean(row.get("발주기관")),
            "budget": _clean(row.get("사업금액")),
            "published_at": _clean(row.get("공개일자")),
            "bid_close_at": _clean(row.get("입찰참여마감일")),
            "summary": _clean(row.get("사업요약")),
            "file_type": _clean(row.get("파일형식")),
            "file_name": row.get("파일명"),
        },
        "text": text,
        "chars": len(text),
        "extractor": row.get("extractor"),
        "warnings": row.get("_warnings") or [],
    }


def write_doc_schema_jsonl(df: pd.DataFrame, path: Path):
    """
    df 전체를 build_doc_schema_record 스키마로 변환해 JSONL로 저장한다.
    원본 메타데이터가 병합된 df여야 meta 필드가 채워진다.

    Args:
        df: merge_original_metadata까지 끝난 최종 DataFrame.
        path: 저장 경로.
    """

    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = build_doc_schema_record(row.to_dict())
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


# ============================================================
# 18. 전체 파이프라인 실행
# ============================================================


def run_pipeline(
    data_dir: Path,
    output_dir: Path,
    original_metadata_csv: Path | None = None,
    sample_size: int | None = None,
    enable_chunk_output: bool = False,  # [수정 8] RAG용 청킹 JSONL 추가 생성 여부
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> tuple[pd.DataFrame, dict]:

    output_dir = Path(output_dir)
    texts_dir = output_dir / "cleaned_texts"
    output_dir.mkdir(parents=True, exist_ok=True)
    texts_dir.mkdir(exist_ok=True)

    files = sorted(
        p
        for p in Path(data_dir).iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if sample_size is not None:
        files = files[:sample_size]

    rows = []

    for i, path in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] {path.name}")

        try:
            row = process_document(path)

        except Exception as e:  # noqa: BLE001
            # [버그 수정] 원본 코드는 RuntimeError만 잡았지만, 실제로
            # process_document 내부(특히 olefile)에서 나는 예외는 대부분
            # RuntimeError가 아니다(예: NotOleFileError는 OSError 상속).
            # 그 결과 문서 1건의 파싱 실패가 100건 전체 배치를 중단시켰다
            # (사용자가 실제 실행에서 NotOleFileError로 재현). 한 문서 실패가
            # 전체를 막지 않도록 넓게 잡고 다음 문서로 넘어간다.
            row = {
                "파일명": path.name,
                "extractor": None,
                "table_parse_success": False,
                "tables_total": 0,
                "tables_failed": 0,
                "tables_partial": 0,  # [수정 1]
                "fallback_used": True,
                "error_reason": f"미처리 예외: {e}\n{traceback.format_exc(limit=1)}",
                "hwp_raw_skipped_reason": None,
                "clean_text": "",
                "_warnings": [],
            }
            row.update({k: None for k in FIELD_ALIASES})

        rows.append(row)

        if row["clean_text"]:
            (texts_dir / f"{path.stem}.txt").write_text(
                row["clean_text"],
                encoding="utf-8",
            )

    df = pd.DataFrame(rows)

    if original_metadata_csv is not None:
        df = merge_original_metadata(df, Path(original_metadata_csv))

    df.to_csv(
        output_dir / "cleaned_documents.csv",
        index=False,
        encoding="utf-8-sig",
    )
    df.to_excel(output_dir / "cleaned_documents.xlsx", index=False)
    write_jsonl(
        df,
        output_dir / "cleaned_documents.jsonl",
        enable_chunk_output=enable_chunk_output,
        chunk_output_path=output_dir / "cleaned_documents_chunks.jsonl",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    report = _build_report(df)

    (output_dir / "preprocessing_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n완료: {len(df)}건 처리, 결과는 {output_dir}에 저장됨")

    return df, report


# [수정 8 - RAG 청킹] LangChain의 RecursiveCharacterTextSplitter와 동일한
# 방식(구분자 우선순위: 문단 -> 줄 -> 공백 -> 문자)으로 직접 구현했다.
# langchain을 새 의존성으로 추가하지 않고도 같은 chunk_size/overlap
# 동작을 낸다.
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


def write_jsonl(
    df: pd.DataFrame,
    path: Path,
    enable_chunk_output: bool = False,
    chunk_output_path: Path | None = None,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
):
    """
    문서별로 한 줄씩 JSON을 기록한다. LangChain Document와 동일한 구조
    (page_content + metadata)로 저장해 팀원의 JSONL 기반 코드에서
    바로 읽을 수 있게 한다. 문서 단위 JSONL은 항상 그대로 유지된다.

    [수정 8] enable_chunk_output=True면, 같은 내용을 chunk_size/overlap
    기준으로 잘라 chunk_output_path에 추가로 저장한다(하이브리드 검색용
    청크 단위 JSONL). chunk_output_path를 안 주면 원래 경로에 "_chunks"를
    붙인 파일명으로 저장한다.

    형태:
        {"page_content": "...", "metadata": {"source": "A.hwp", "extractor": "hwp_raw", ...}}
    """

    metadata_columns = [
        col
        for col in df.columns
        if col not in ("파일명", "clean_text", "clean_text_for_generation")
    ]

    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = {
                "page_content": row["clean_text"] or "",  # 검색/임베딩용 (평문)
                "page_content_for_generation": row.get("clean_text_for_generation")
                or "",  # 생성/LLM 컨텍스트용 (표 형태별 구조 유지)
                "metadata": {
                    "source": row["파일명"],
                    **{col: row[col] for col in metadata_columns},
                },
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if not enable_chunk_output:
        return

    if chunk_output_path is None:
        path = Path(path)
        chunk_output_path = path.with_name(f"{path.stem}_chunks{path.suffix}")

    with open(chunk_output_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            # [청킹 정합성] chunk_pairs()가 한 번만 자르고 두 판본을 함께
            # 돌려주므로 chunk_index가 늘 같은 구간을 가리킨다. 길이는
            # 검색용 기준으로 재고, 표는 중간에서 안 잘린다.
            generation_source = (
                row.get("clean_text_for_generation") or row["clean_text"] or ""
            )
            pairs = chunk_pairs(
                generation_source,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            for i, (embedding_chunk, gen_chunk) in enumerate(pairs):
                record = {
                    "page_content": embedding_chunk,  # 검색/임베딩용
                    "page_content_for_generation": gen_chunk,  # 생성/LLM 컨텍스트용
                    "metadata": {
                        "source": row["파일명"],
                        "chunk_index": i,
                        "chunk_total": len(pairs),
                        **{col: row[col] for col in metadata_columns},
                    },
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _build_report(df: pd.DataFrame) -> dict:

    total = len(df)
    success = int((df["clean_text"].str.len() > 0).sum())

    table_applicable = df[df["extractor"].isin(["hwp_raw", "pdfplumber"])]
    table_success_rate = (
        float(table_applicable["table_parse_success"].mean())
        if len(table_applicable) > 0
        else None
    )

    metadata_success = {
        field_name: float(df[field_name].notna().mean()) for field_name in FIELD_ALIASES
    }

    hwp_raw_skipped = df.loc[
        df["hwp_raw_skipped_reason"].notna(),
        ["파일명", "extractor", "hwp_raw_skipped_reason"],
    ]

    # [수정 1] 부분 복원된 표 총 개수도 리포트에 남긴다(품질 모니터링용).
    partial_tables_total = (
        int(df["tables_partial"].sum()) if "tables_partial" in df.columns else 0
    )

    return {
        "총문서수": total,
        "추출성공률": success / total if total else 0,
        "표복원성공률": table_success_rate,
        "부분복원_표개수": partial_tables_total,
        "extractor_비율": df["extractor"].value_counts(dropna=False).to_dict(),
        "메타데이터_추출성공률": metadata_success,
        "실패문서목록": df.loc[df["clean_text"] == "", "파일명"].tolist(),
        "hwp_raw_조용한_폴백건수": len(hwp_raw_skipped),
        "hwp_raw_조용한_폴백목록": hwp_raw_skipped.to_dict(orient="records"),
        "오류원인": (
            df.loc[df["error_reason"].notna(), ["파일명", "error_reason"]].to_dict(
                orient="records"
            )
        ),
    }


# ============================================================
# 18. 실행
# ============================================================

if __name__ == "__main__":
    DATA_DIR = Path("여기에 원본 파일 경로")
    OUTPUT_DIR = Path("./output")
    ORIGINAL_METADATA_CSV = Path("./original_metadata.csv")  # 처음 받은 CSV 경로로 수정

    run_pipeline(
        DATA_DIR,
        OUTPUT_DIR,
        original_metadata_csv=ORIGINAL_METADATA_CSV,
    )


# ============================================================
# 19. 청킹 데이터 파일 뽑기
# ============================================================
df = pd.read_csv("")

DOC_PATH = Path(
    r"C:\Users\asd\Desktop\중급 프젝\v2_chosim\rfp-rag-system\src\preprocessing\output\cleaned_documents.jsonl"
)
DOC_PATH.parent.mkdir(parents=True, exist_ok=True)  # 폴더 없으면 생성

write_jsonl(
    df,
    DOC_PATH,
    enable_chunk_output=True,
    chunk_output_path=DOC_PATH.with_name("cleaned_documents_v7__chunks_1500_250.jsonl"),
    chunk_size=1500,
    chunk_overlap=250,
)
