"""설정 상수·예외·결과 타입. 다른 모듈이 전부 여기를 본다.

원본: `preprocessing/pipeline.py` (전처리팀). 분할 경위는 `src/preprocessing/README.md`.
"""

import re
from dataclasses import dataclass

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
_HWP_CONTROL_WITH_EXTRA = frozenset({
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    11,
    12,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
})
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


