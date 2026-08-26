import difflib
import re
import struct
import subprocess
import tempfile
import zlib
from collections import Counter
from pathlib import Path

import olefile
import pandas as pd

df = pd.read_csv(r"/home/spai1216/workspace/data/data_list.csv")
print(df.shape)
print(df.tail(5))
print(df.head(5))
print(df.info())


print(df["텍스트"].str.contains("□").sum())
print(df.loc[0, "텍스트"][:501])


print("여기 원본 데이터에서 메타데이터 추출")

DATA_DIR = Path("/home/spai1216/workspace/data/files")

PATTERNS = {
    "사업기간": (
        r"(?:[□■▪▶○◦•]?\s*(?:\d+[\.\)]\s*)?)"
        r"(?:사업\s*기간|사업기간|"
        r"사\s*업\s*수\s*행\s*기\s*간|사업수행기간|"
        r"수\s*행\s*기\s*간|수행기간|"
        r"용\s*역\s*기\s*간|용역기간|"
        r"계\s*약\s*기\s*간|계약기간|"
        r"과\s*업\s*기\s*간|과업기간)"
    ),
    "참가자격": (
        r"(?:[□■▪▶○◦•]?\s*(?:\d+[\.\)]\s*)?)"
        r"(?:입찰\s*참가\s*자격|참가\s*자격|참가자격|"
        r"신\s*청\s*자\s*격|신청자격|"
        r"지\s*원\s*자\s*격|지원자격|"
        r"응\s*모\s*자\s*격|응모자격|"
        r"자\s*격\s*요\s*건|참가\s*요건)"
    ),
    "보안특약": r"□[^가-힣]{0,10}보안\s*특약",
    "하자보수": r"□[^가-힣]{0,10}하자\s*보수",
}


def extract_text(path: Path) -> str:
    """hwp 또는 pdf 파일에서 텍스트를 추출한다.

    Args:
        path: 원본 파일 경로.

    Returns:
        추출된 텍스트. 실패 시 빈 문자열.
    """
    suffix = path.suffix.lower()

    if suffix == ".hwp":
        result = subprocess.run(
            ["hwp5txt", str(path)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"hwp5txt 실패: {result.stderr[:200]}")
        return result.stdout

    if suffix == ".pdf":
        result = subprocess.run(
            ["pdftotext", str(path), "-"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"pdftotext 실패: {result.stderr[:200]}")
        return result.stdout

    return ""


def extract_field(text: str, pattern: str, max_window: int = 100) -> str | None:
    """패턴 매치 지점부터 다음 항목 경계(□) 또는 줄바꿈 전까지를 값으로 반환한다.

    Args:
        text: 본문 텍스트.
        pattern: 정규식 패턴 (라벨 자체).
        max_window: 값 구간을 찾지 못했을 때의 최대 길이.

    Returns:
        추출된 값, 매치가 없으면 None.
    """
    match = re.search(pattern, text)
    if match is None:
        return None

    start = match.end()
    remainder = text[start : start + max_window]
    remainder = re.sub(r"^[\s:\)ㅇ⚬◦*]+", "", remainder)

    boundary = re.search(r"[□\n]", remainder)
    value = remainder[: boundary.start()] if boundary else remainder

    value = value.strip()
    return value if value else None


def run(data_dir: Path, sample_size: int | None = None) -> pd.DataFrame:
    """원본 파일들을 순회하며 패턴별 매칭 결과를 표로 정리한다.

    Args:
        data_dir: 원본 파일이 있는 디렉토리.
        sample_size: 앞에서부터 확인할 파일 수. None이면 전체.

    Returns:
        파일명, □ 포함 여부, 패턴별 추출값을 담은 DataFrame.
    """
    files = sorted(
        p for p in data_dir.iterdir() if p.suffix.lower() in (".hwp", ".pdf")
    )
    if sample_size:
        files = files[:sample_size]

    rows = []
    for path in files:
        try:
            text = extract_text(path)
        except RuntimeError as e:
            rows.append({"파일명": path.name, "오류": str(e)})
            continue

        row = {
            "파일명": path.name,
            "확장자": path.suffix.lower(),
            "글자수": len(text),
            "□포함여부": "□" in text,
        }
        for label, pattern in PATTERNS.items():
            row[label] = extract_field(text, pattern)
        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # 먼저 10개만 빠르게 확인 (전체 100개는 시간이 걸릴 수 있음)
    df_result = run(DATA_DIR, sample_size=10)
    pd.set_option("display.max_colwidth", 40)

    if "오류" in df_result.columns:
        errors = df_result[df_result["오류"].notna()]
        if not errors.empty:
            print(f"오류 발생 {len(errors)}건:")
            print(errors[["파일명", "오류"]].to_string(index=False))
            print()

    if "□포함여부" in df_result.columns:
        print(df_result.to_string(index=False))
        print(f"\n□ 기호 포함 파일 비율: {df_result['□포함여부'].mean() * 100:.1f}%")
    else:
        print("모든 파일에서 텍스트 추출이 실패했습니다. 위 오류 내용을 확인하세요.")


text = extract_text(min(DATA_DIR.iterdir()))

for line in text.splitlines():
    line = line.strip()
    if re.match(r"^(?:[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+|\d+[\.\)]?|[가-힣][\.\)]?|[□■○◦•▪▶])", line):
        print(line)


# 기존 extract_text() 함수 그대로 사용

HEADER_PATTERNS = {
    "Ⅰ.": r"^\s*[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.",
    "1.": r"^\s*\d+\.",
    "1)": r"^\s*\d+\)",
    "가.": r"^\s*[가-힣]\.",
    "□": r"^\s*□",
    "■": r"^\s*■",
    "○": r"^\s*[○◦•▪▶]",
}

METADATA_VARIANTS = {
    "사업기간": [
        r"사업\s*기간",
        r"사업기간",
        r"사업\s*수행\s*기간",
        r"수행\s*기간",
        r"용역\s*기간",
        r"계약\s*기간",
        r"과업\s*기간",
    ],
    "참가자격": [
        r"입찰\s*참가\s*자격",
        r"참가\s*자격",
        r"참가자격",
        r"신청\s*자격",
        r"지원\s*자격",
        r"응모\s*자격",
        r"자격\s*요건",
        r"참가\s*요건",
    ],
}


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


def split_sections(text):
    """
    헤더 기준으로 문서를 섹션 단위로 분리
    """
    lines = text.splitlines()

    sections = []
    current = []

    header_regex = re.compile(r"^\s*(?:[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.|\d+[.)]|[가-힣]\.|[□■○◦•▪▶])")

    for line in lines:
        if header_regex.match(line) and current:
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


run_eda(DATA_DIR, sample_size=10)


def df_a2(a: int, b: str) -> str:
    """_summary_

    Args:
        a (int): _description_
        b (str): _description_

    Raises:
        FileNotFoundError: _description_
        RuntimeError: _description_
        RuntimeError: _description_
        RuntimeError: _description_
        RuntimeError: _description_
        RuntimeError: _description_
        RuntimeError: _description_
        RuntimeError: _description_
        RuntimeError: _description_
        RuntimeError: _description_
        RuntimeError: _description_
        RuntimeError: _description_
        RuntimeError: _description_
        RuntimeError: _description_

    Returns:
        str: _description_

    Yields:
        Iterator[str]: _description_
    """


# 추출 실패한 문서 확인 (run_a2 실행 후 확인)
if "df_a2" in globals():
    failed = df_a2[df_a2["추출성공"] == False]
    print(f"실패 문서 수: {len(failed)}")
    print(failed[["파일명", "확장자", "오류"]].to_string(index=False))


for name in ["MILE", "AFSIS"]:
    for path in DATA_DIR.glob(f"*{name}*.hwp"):
        print(f"\n=== {path.name} ===")

        result = subprocess.run(
            ["hwp5txt", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )

        print("returncode:", result.returncode)
        print("stdout 길이:", len(result.stdout))
        print("stderr 마지막 500자:\n", result.stderr[-500:])


## csv eda 코드


# ============================================================
# 0. 설정
# ============================================================

CSV_PATH = Path("/home/spai1216/workspace/data/data_list.csv")


# ============================================================
# 1. CSV 로드
# ============================================================


def load_csv(path: Path) -> pd.DataFrame:
    """
    CSV 파일을 읽는다.
    UTF-8이 안 될 경우 CP949로 재시도한다.
    """

    try:
        df = pd.read_csv(path, encoding="utf-8")

    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp949")

    return df


# ============================================================
# 2. 기본 데이터셋 정보
# ============================================================
print("찬형님 바보")


def analyze_basic(df: pd.DataFrame):

    print("=" * 80)
    print("1. 데이터셋 기본 정보")
    print("=" * 80)

    print(f"행(row) 수 : {len(df):,}")
    print(f"열(column) 수 : {len(df.columns):,}")

    print("\n[컬럼 목록]")
    for i, column in enumerate(df.columns, start=1):
        print(f"{i:2}. {column}")


# ============================================================
# 3. 컬럼별 데이터 타입 / 결측치
# ============================================================


def analyze_columns(df: pd.DataFrame):

    print("\n")
    print("=" * 80)
    print("2. 컬럼별 데이터 타입 / 결측치")
    print("=" * 80)

    result = pd.DataFrame(
        {
            "데이터타입": df.dtypes.astype(str),
            "결측치수": df.isna().sum(),
            "결측치비율(%)": (df.isna().mean() * 100).round(2),
            "고유값수": df.nunique(dropna=True),
        }
    )

    print(result.to_string())


# ============================================================
# 4. 중복 데이터
# ============================================================


def analyze_duplicates(df: pd.DataFrame):

    print("\n")
    print("=" * 80)
    print("3. 중복 데이터")
    print("=" * 80)

    duplicate_rows = df.duplicated().sum()

    print(f"완전 중복 행 : {duplicate_rows:,}")

    # 파일명 컬럼 후보 탐색
    filename_candidates = [
        column
        for column in df.columns
        if any(
            keyword in column.lower()
            for keyword in [
                "file",
                "filename",
                "파일",
                "파일명",
                "문서",
                "document",
            ]
        )
    ]

    if filename_candidates:
        print("\n[파일명 후보 컬럼]")

        for column in filename_candidates:
            duplicate_count = df[column].duplicated().sum()

            print(f"{column}: 중복 {duplicate_count:,}건")


# ============================================================
# 5. 컬럼별 고유값 확인
# ============================================================


def analyze_unique_values(
    df: pd.DataFrame,
    top_n: int = 20,
):

    print("\n")
    print("=" * 80)
    print("4. 컬럼별 값 분포")
    print("=" * 80)

    for column in df.columns:
        print("\n" + "-" * 70)
        print(f"[{column}]")

        # 값이 너무 많은 컬럼은 상위 N개만
        value_counts = df[column].value_counts(dropna=False).head(top_n)

        print(value_counts.to_string())


# ============================================================
# 6. 텍스트 컬럼 길이
# ============================================================


def analyze_text_columns(df: pd.DataFrame):

    print("\n")
    print("=" * 80)
    print("5. 텍스트 컬럼 길이")
    print("=" * 80)

    text_columns = df.select_dtypes(include=["object", "string"]).columns

    for column in text_columns:
        lengths = df[column].fillna("").astype(str).str.len()

        print("\n" + "-" * 70)
        print(f"[{column}]")

        print(lengths.describe().to_string())


# ============================================================
# 7. 파일 형식 분석
# ============================================================


def analyze_file_extensions(df: pd.DataFrame):

    print("\n")
    print("=" * 80)
    print("6. 파일 형식 분석")
    print("=" * 80)

    # 확장자 컬럼이 이미 존재하는 경우
    extension_columns = [
        column
        for column in df.columns
        if "확장자" in column
        or "extension" in column.lower()
        or "ext" in column.lower()
    ]

    if extension_columns:
        for column in extension_columns:
            print(f"\n[{column}]")

            print(df[column].value_counts(dropna=False).to_string())

        return

    # 파일명에서 확장자를 추출할 수 있는 컬럼 찾기
    filename_candidates = [
        column
        for column in df.columns
        if any(
            keyword in column.lower()
            for keyword in [
                "file",
                "filename",
                "파일",
                "파일명",
                "문서",
            ]
        )
    ]

    if not filename_candidates:
        print("파일명 컬럼을 찾지 못했습니다.")
        return

    for column in filename_candidates:
        extensions = (
            df[column]
            .dropna()
            .astype(str)
            .str.extract(
                r"(\.[^.]+)$",
                expand=False,
            )
            .str.lower()
        )

        print(f"\n[{column}]")

        print(extensions.value_counts(dropna=False).to_string())


# ============================================================
# 8. 실행
# ============================================================


def run_csv_eda(csv_path: Path):

    print("=" * 80)
    print("CSV EDA 시작")
    print("=" * 80)

    print(f"파일: {csv_path}")

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

    df = load_csv(csv_path)

    print(f"CSV 로드 완료: {df.shape}")

    analyze_basic(df)

    analyze_columns(df)

    analyze_duplicates(df)

    analyze_unique_values(df)

    analyze_text_columns(df)

    analyze_file_extensions(df)

    return df


## csv 결측치 파일 추출


def show_missing_files(df: pd.DataFrame):
    """
    결측치가 하나라도 있는 행의 파일명을 출력한다.
    탐색기에서 해당 파일을 직접 찾을 수 있도록
    파일명을 그대로 출력한다.
    """

    print("=" * 100)
    print("결측치가 있는 원본 파일")
    print("=" * 100)

    # --------------------------------------------------------
    # 파일명 컬럼 찾기
    # --------------------------------------------------------

    filename_candidates = [
        col
        for col in df.columns
        if any(
            keyword in col.lower()
            for keyword in [
                "file",
                "filename",
                "파일",
                "파일명",
                "첨부",
                "document",
                "문서",
            ]
        )
    ]

    if not filename_candidates:
        print("❌ 파일명 컬럼을 찾지 못했습니다.")
        print("\n현재 CSV 컬럼:")
        for col in df.columns:
            print(f" - {col}")

        return

    print(f"파일명 컬럼 후보: {filename_candidates}")
    print()

    # 첫 번째 후보 사용
    filename_column = filename_candidates[0]

    # --------------------------------------------------------
    # 결측치가 하나라도 있는 행
    # --------------------------------------------------------

    missing_rows = df[df.isna().any(axis=1)].copy()

    print(f"결측치가 있는 행: {len(missing_rows):,}건")

    print()

    if missing_rows.empty:
        print("결측치가 있는 문서가 없습니다.")
        return

    # --------------------------------------------------------
    # 출력
    # --------------------------------------------------------

    for i, (_, row) in enumerate(
        missing_rows.iterrows(),
        start=1,
    ):
        filename = row[filename_column]

        missing_columns = [col for col in df.columns if pd.isna(row[col])]

        print(f"[{i}] {filename}")
        print(f"    결측 컬럼: {', '.join(missing_columns)}")
        print()


df = run_csv_eda(CSV_PATH)


## 텍스트 추출 실패 대체 파서 코드


# ============================================================
# 0. 설정
# ============================================================

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


# ============================================================
# 1. 텍스트 추출 - PDF
# ============================================================


def extract_text_pdf(path: Path) -> str:
    result = subprocess.run(
        ["pdftotext", str(path), "-"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(f"pdftotext 실패: {result.stderr[:300]}")

    return result.stdout


# ============================================================
# 2. 텍스트 추출 - HWP (1차: hwp5txt)
# ============================================================


def extract_text_hwp5txt(path: Path) -> str:
    """
    pyhwp의 hwp5txt CLI로 텍스트를 추출한다.
    내부적으로 스타일/XML을 파싱하므로 일부 문서에서
    UnicodeDecodeError, XMLSyntaxError 등이 발생할 수 있다.
    """

    result = subprocess.run(
        ["hwp5txt", str(path)], capture_output=True, text=True, timeout=60, check=False
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"hwp5txt 실패(returncode={result.returncode}): {result.stderr[:300]}"
        )

    if not result.stdout.strip():
        raise RuntimeError("hwp5txt 결과가 비어 있음")

    return result.stdout


# ============================================================
# 3. 텍스트 추출 - HWP (2차: LibreOffice 변환)
# ============================================================


def extract_text_libreoffice(path: Path) -> str:
    """
    LibreOffice(soffice)로 HWP -> TXT 변환 후 읽는다.
    hwp5txt가 내부 파싱에서 실패하는 문서에 대한 대체 경로.
    soffice가 설치되어 있어야 한다(apt install libreoffice).
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
            raise RuntimeError(
                f"soffice 변환 실패(returncode={result.returncode}): "
                f"{result.stderr[:300]}"
            )

        out_path = Path(tmp_dir) / f"{path.stem}.txt"

        if not out_path.exists():
            raise RuntimeError("soffice 변환 결과 파일 없음")

        text = out_path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        if not text.strip():
            raise RuntimeError("soffice 변환 결과가 비어 있음")

        return text


# ============================================================
# 4. 텍스트 추출 - HWP (3차: OLE 스트림 직접 파싱, best-effort)
# ============================================================
#
# 주의: hwp5txt / soffice 둘 다 실패하는 손상/비표준 문서를 위한
# 최후 수단이다. HWP 5.0 BodyText 레코드를 직접 읽어 문단 텍스트만
# 뽑아내며, 표/그림 등 인라인 컨트롤 객체의 부가 데이터 길이는
# 근사치로 처리한다. 완벽한 복원이 아니므로 결과는 반드시 샘플
# 육안 검수 후 사용할 것.
# ============================================================


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


# ============================================================
# 5. 통합 추출 (폴백 체인)
# ============================================================

_HWP_EXTRACTORS = {
    "hwp5txt": extract_text_hwp5txt,
    "libreoffice": extract_text_libreoffice,
    "hwp_raw": extract_text_hwp_raw,
}


def extract_text(path: Path):
    """
    확장자에 맞는 추출기를 순서대로 시도한다.

    Returns:
        tuple[str, str | None, dict[str, str]]:
            (텍스트, 성공한 방법명 또는 None, 방법별 오류 메시지)
    """

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return extract_text_pdf(path), "pdftotext", {}

    if suffix != ".hwp":
        return "", None, {}

    errors = {}

    for method in HWP_EXTRACTION_METHODS:
        try:
            text = _HWP_EXTRACTORS[method](path)
            return text, method, errors

        except RuntimeError as e:
            errors[method] = str(e)

    return "", None, errors


# ============================================================
# 6. 기본 텍스트 통계
# ============================================================


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


# ============================================================
# 7. 텍스트 품질 분석
# ============================================================


def analyze_text_quality(text: str):

    replacement = text.count("�")
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


# ============================================================
# 8. 구조 마커 분석
# ============================================================

STRUCTURE_PATTERNS = {
    "□": r"□",
    "■": r"■",
    "○": r"○",
    "◦": r"◦",
    "●": r"●",
    "※": r"※",
    "숫자_점": r"(?m)^\s*\d+\.",
    "숫자_괄호": r"(?m)^\s*\d+\)",
    "한글_점": r"(?m)^\s*[가-힣]\.",
    "한글_괄호": r"(?m)^\s*[가-힣]\)",
    "로마숫자": r"(?m)^\s*[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.",
    "괄호숫자": r"(?m)^\s*\(\d+\)",
}


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


# ============================================================
# 9. □로 시작하는 라인 분석
# ============================================================


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


# ============================================================
# 10. 제목처럼 보이는 라인 분석
# ============================================================


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


# ============================================================
# 11. 문서 하나 분석
# ============================================================


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
            "; ".join(f"{k}={v}" for k, v in errors.items()) or "추출된 텍스트 없음"
        )
        return result, ""

    result["추출성공"] = True
    result["추출방법"] = method

    result.update(analyze_text_basic(text))
    result.update(analyze_text_quality(text))
    result.update(analyze_structure(text))

    return result, text


# ============================================================
# 12. 전체 원본 문서 분석
# ============================================================


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


# ============================================================
# 13. 결과 출력
# ============================================================


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


df_a2, texts, square_lines, heading_candidates = run_a2(
    DATA_DIR,
)

print_a2_result(
    df_a2,
    texts,
    square_lines,
    heading_candidates,
)


## 파일명 추출

print(df[df["파일명"].str.contains("MILE", na=False)]["파일명"].tolist())
print(df[df["파일명"].str.contains("AFSIS", na=False)]["파일명"].tolist())


## 대체 파서로 파싱한 문서 검수 코드


HWPTAG_PARA_TEXT = 0x43
_HWP_CONTROL_WITH_EXTRA = frozenset(
    {1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25}
)
_HWP_CONTROL_SKIP_ONLY = frozenset({0, 9, 26, 27, 28, 29, 30, 31})


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
            raise RuntimeError(f"soffice 변환 실패: {result.stderr[:300]}")
        out_path = Path(tmp_dir) / f"{path.stem}.txt"
        if not out_path.exists():
            raise RuntimeError("soffice 변환 결과 파일 없음")
        text = out_path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            raise RuntimeError("soffice 변환 결과가 비어 있음")
        return text


def _hwp_is_compressed(ole):
    header = ole.openstream("FileHeader").read()
    return bool(struct.unpack("<I", header[36:40])[0] & 0x01)


def _hwp_iter_records(data: bytes):
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
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def check_two_documents(filenames: list[str], data_dir: Path):
    """지정한 문서만 hwp_raw / libreoffice 두 방식으로 재추출해 비교한다."""
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
        "/home/spai1216/workspace/data/files/대전대학교_대전대학교 2024학년도 다층적 융합 학습경험 플랫폼(MILE) 전.hwp",
        "/home/spai1216/workspace/data/files/한국농어촌공사_아세안+3 식량안보정보시스템(AFSIS) 3단계 협력(캄보디아.hwp",
    ],
    data_dir=DATA_DIR,
)


### 원본 데이터 공백, 줄바꿈, 특수문자 등 EDA 코드


# ============================================================
# 0. 정규식 패턴 정의
# ============================================================

# 연속 공백/탭 (5칸 이상)
PATTERN_LONG_SPACE = re.compile(r"[ \t]{5,}")

# 연속 줄바꿈 (4개 이상)
PATTERN_LONG_NEWLINE = re.compile(r"\n{4,}")

# 개인정보/자리표시자용 마스킹 문자
PATTERN_PLACEHOLDER = re.compile(r"[ㅇ○]{2,}")

# 빈 괄호 또는 공백만 있는 괄호  예: "( )", "(   )"
PATTERN_EMPTY_PAREN = re.compile(r"\(\s*\)")

# 페이지번호 형태  예: "- 12 -", "12 / 100"
PATTERN_PAGE_NUMBER = re.compile(r"^\s*-?\s*\d+\s*(-|/\s*\d+)?\s*-?\s*$")

# 서명/날인 흔적
PATTERN_SIGNATURE = re.compile(r"\(\s*인\s*\)")

# 밑줄/점선 등 구분선  예: "________", "………", "----------"
PATTERN_DIVIDER_LINE = re.compile(r"^[_\-—=~・.…\s]{5,}$")


# ============================================================
# 1. 문서별 공백/줄바꿈 노이즈 통계
# ============================================================


def analyze_whitespace_noise(texts: dict) -> pd.DataFrame:
    """
    문서별 긴 공백/과도한 줄바꿈/구분선 라인 수를 집계한다.
    """

    rows = []

    for filename, text in texts.items():
        lines = text.splitlines()

        divider_lines = sum(
            1 for line in lines if PATTERN_DIVIDER_LINE.match(line.strip())
        )

        rows.append(
            {
                "파일명": filename,
                "긴공백": len(PATTERN_LONG_SPACE.findall(text)),
                "과도한줄바꿈": len(PATTERN_LONG_NEWLINE.findall(text)),
                "구분선라인": divider_lines,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 2. 특수문자 빈도 (전체 코퍼스 기준)
# ============================================================


def analyze_special_char_frequency(
    texts: dict,
    top_n: int = 30,
) -> pd.DataFrame:
    """
    한글/영문/숫자/공백을 제외한 특수문자의 전체 빈도와,
    몇 개 문서에 걸쳐 등장하는지를 함께 집계한다.
    등장 문서 수가 많을수록 '노이즈성 공통 기호'일 가능성이 높다.
    """

    char_count = Counter()
    char_doc_count = Counter()

    for text in texts.values():
        chars_in_doc = re.findall(r"[^\w\s가-힣]", text, flags=re.UNICODE)

        char_count.update(chars_in_doc)
        char_doc_count.update(set(chars_in_doc))

    rows = [
        {
            "문자": char,
            "전체빈도": count,
            "등장문서수": char_doc_count[char],
        }
        for char, count in char_count.most_common(top_n)
    ]

    return pd.DataFrame(rows)


# ============================================================
# 3. 플레이스홀더 / 페이지번호 / 서명란 탐지
# ============================================================


def analyze_noise_patterns(texts: dict) -> pd.DataFrame:
    """
    문서별로 플레이스홀더(ㅇㅇㅇ, ○○○), 빈 괄호, 페이지번호,
    서명란("(인)") 패턴의 등장 횟수를 집계한다.
    """

    rows = []

    for filename, text in texts.items():
        lines = text.splitlines()

        page_number_lines = sum(
            1
            for line in lines
            if line.strip() and PATTERN_PAGE_NUMBER.match(line.strip())
        )

        rows.append(
            {
                "파일명": filename,
                "플레이스홀더": len(PATTERN_PLACEHOLDER.findall(text)),
                "빈괄호": len(PATTERN_EMPTY_PAREN.findall(text)),
                "페이지번호라인": page_number_lines,
                "서명란": len(PATTERN_SIGNATURE.findall(text)),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 4. 문서 간 반복되는 boilerplate 라인 탐지
# ============================================================


def detect_boilerplate_lines(
    texts: dict,
    min_doc_ratio: float = 0.2,
    min_line_length: int = 8,
) -> pd.DataFrame:
    """
    여러 문서에 걸쳐 동일하게(공백 정규화 후) 반복되는 라인을 찾는다.
    특정 문서 고유 내용이 아니라 표준 계약조항일 가능성이 높은 라인.

    Args:
        min_doc_ratio: 전체 문서 수 대비 이 비율 이상에서 등장해야 boilerplate로 간주
        min_line_length: 너무 짧은 라인(번호만 있는 등)은 제외
    """

    total_docs = len(texts)
    line_doc_map: dict[str, set] = {}

    for filename, text in texts.items():
        seen_in_doc = set()

        for line in text.splitlines():
            normalized = re.sub(r"\s+", " ", line.strip())

            if len(normalized) < min_line_length:
                continue

            if normalized in seen_in_doc:
                continue

            seen_in_doc.add(normalized)
            line_doc_map.setdefault(normalized, set()).add(filename)

    threshold = max(2, int(total_docs * min_doc_ratio))

    rows = [
        {
            "라인": line,
            "등장문서수": len(docs),
            "전체문서비율": len(docs) / total_docs,
        }
        for line, docs in line_doc_map.items()
        if len(docs) >= threshold
    ]

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values("등장문서수", ascending=False)

    return df


# ============================================================
# 5. 전체 요약 출력
# ============================================================


def print_preprocessing_diagnostics(texts: dict):

    print("=" * 80)
    print("[A] 공백/줄바꿈 노이즈 (문서별)")
    print("=" * 80)

    ws_df = analyze_whitespace_noise(texts)
    print(ws_df.describe().to_string())
    print("\n[상위 10건]")
    print(ws_df.sort_values("긴공백", ascending=False).head(10).to_string(index=False))

    print("\n")
    print("=" * 80)
    print("[B] 특수문자 빈도 TOP 30 (전체 코퍼스)")
    print("=" * 80)

    char_df = analyze_special_char_frequency(texts)
    print(char_df.to_string(index=False))

    print("\n")
    print("=" * 80)
    print("[C] 플레이스홀더 / 페이지번호 / 서명란")
    print("=" * 80)

    noise_df = analyze_noise_patterns(texts)
    print(noise_df.describe().to_string())
    print("\n[상위 10건]")
    print(
        noise_df.sort_values("플레이스홀더", ascending=False)
        .head(10)
        .to_string(index=False)
    )

    print("\n")
    print("=" * 80)
    print("[D] 문서 간 반복되는 boilerplate 라인 (표준 계약조항 후보)")
    print("=" * 80)

    boilerplate_df = detect_boilerplate_lines(texts)
    print(f"boilerplate 후보 라인 수: {len(boilerplate_df)}")
    print("\n[등장문서수 TOP 30]")
    print(boilerplate_df.head(30).to_string(index=False))

    return {
        "whitespace": ws_df,
        "special_char": char_df,
        "noise_pattern": noise_df,
        "boilerplate": boilerplate_df,
    }


# ============================================================
# 6. 정제 함수 (통계 확인 후 필요한 것만 적용)
# ============================================================


def clean_text(
    text: str,
    boilerplate_lines: set | None = None,
    remove_page_numbers: bool = True,
) -> str:
    """
    통계로 확인한 노이즈를 제거/정규화한다.
    boilerplate_lines는 detect_boilerplate_lines() 결과에서
    실제로 제거하기로 결정한 라인 집합만 넘길 것
    (구조 마커가 포함된 라인은 절대 넣지 말 것).
    """

    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        normalized = re.sub(r"\s+", " ", line.strip())

        if remove_page_numbers and PATTERN_PAGE_NUMBER.match(normalized):
            continue

        if PATTERN_DIVIDER_LINE.match(normalized):
            continue

        if boilerplate_lines and normalized in boilerplate_lines:
            continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # 연속 공백/줄바꿈 정규화
    text = PATTERN_LONG_SPACE.sub(" ", text)
    text = PATTERN_LONG_NEWLINE.sub("\n\n", text)

    return text.strip()


texts: dict[str, str]  # run_a2()에서 얻은 결과

stats = print_preprocessing_diagnostics(texts)

# boilerplate 후보 중 실제로 제거할 라인만 골라서
to_remove = set(stats["boilerplate"].query("전체문서비율 > 0.3")["라인"])
cleaned_texts = {
    filename: clean_text(text, boilerplate_lines=to_remove)
    for filename, text in texts.items()
}


## 원문 데이터 특수기호 등 헤더 검수 코드


# ============================================================
# 0. 패턴 정의 (기존 진단 코드와 동일)
# ============================================================

PATTERN_ANGLE_BRACKET = re.compile(r"[<>]")
PATTERN_BULLET_DOT = re.compile(r"•")
PATTERN_PLACEHOLDER = re.compile(r"[ㅇ○]{2,}")
PATTERN_PAGE_NUMBER = re.compile(r"^\s*-?\s*\d+\s*(-|/\s*\d+)?\s*-?\s*$")

CONTEXT_CHARS = 60  # 매치 전후로 보여줄 글자 수
MAX_EXAMPLES = 8  # 문서당 출력할 최대 예시 수


# ============================================================
# 1. 문서별 카운트 집계 (대표 문서 선정용)
# ============================================================


def _count_per_doc(texts: dict, pattern: re.Pattern) -> pd.Series:
    counts = {filename: len(pattern.findall(text)) for filename, text in texts.items()}
    return pd.Series(counts).sort_values(ascending=False)


def _count_page_number_lines_per_doc(texts: dict) -> pd.Series:
    counts = {}

    for filename, text in texts.items():
        n = sum(
            1
            for line in text.splitlines()
            if line.strip() and PATTERN_PAGE_NUMBER.match(line.strip())
        )
        counts[filename] = n

    return pd.Series(counts).sort_values(ascending=False)


# ============================================================
# 2. 매치 구간 문맥 출력 (글자 단위 패턴용)
# ============================================================


def show_char_context(
    text: str,
    pattern: re.Pattern,
    max_examples: int = MAX_EXAMPLES,
    context_chars: int = CONTEXT_CHARS,
):
    matches = list(pattern.finditer(text))

    print(
        f"  총 매치 수: {len(matches)}건 (예시 {min(max_examples, len(matches))}건 출력)"
    )

    step = max(1, len(matches) // max_examples)

    for match in matches[::step][:max_examples]:
        start = max(0, match.start() - context_chars)
        end = min(len(text), match.end() + context_chars)

        snippet = text[start:end].replace("\n", "⏎")

        print(f"    ...{snippet}...")


# ============================================================
# 3. 매치 라인 문맥 출력 (라인 단위 패턴용)
# ============================================================


def show_line_context(
    text: str,
    pattern: re.Pattern,
    max_examples: int = MAX_EXAMPLES,
):
    lines = text.splitlines()

    matched_indices = [
        i
        for i, line in enumerate(lines)
        if line.strip() and pattern.match(line.strip())
    ]

    print(
        f"  총 매치 라인 수: {len(matched_indices)}건 (예시 {min(max_examples, len(matched_indices))}건 출력)"
    )

    step = max(1, len(matched_indices) // max_examples)

    for i in matched_indices[::step][:max_examples]:
        before = lines[i - 1].strip() if i > 0 else ""
        current = lines[i].strip()
        after = lines[i + 1].strip() if i + 1 < len(lines) else ""

        print(f"    [윗줄] {before}")
        print(f"    [매치] {current}")
        print(f"    [아랫줄] {after}")
        print()


# ============================================================
# 4. 검증 1: <, > 문자 (표/태그 잔재 의심)
# ============================================================


def verify_angle_brackets(texts: dict):

    print("=" * 80)
    print("[검증 1] <, > 문자 — 표/태그 잔재 의심")
    print("=" * 80)

    doc_counts = _count_per_doc(texts, PATTERN_ANGLE_BRACKET)
    top_doc = doc_counts.index[0]

    print(f"대표 문서(최다 빈도): {top_doc}  ({doc_counts.iloc[0]}건)\n")

    show_char_context(texts[top_doc], PATTERN_ANGLE_BRACKET)


# ============================================================
# 5. 검증 2: • 집중 문서
# ============================================================


def verify_bullet_dot(texts: dict):

    print("\n")
    print("=" * 80)
    print("[검증 2] • 문자 — 특정 문서 집중 여부")
    print("=" * 80)

    doc_counts = _count_per_doc(texts, PATTERN_BULLET_DOT)
    doc_counts = doc_counts[doc_counts > 0]

    print(f"• 문자가 등장하는 문서 수: {len(doc_counts)}건")
    print(doc_counts.to_string())

    top_doc = doc_counts.index[0]
    print(f"\n대표 문서(최다 빈도): {top_doc}  ({doc_counts.iloc[0]}건)\n")

    show_char_context(texts[top_doc], PATTERN_BULLET_DOT)


# ============================================================
# 6. 검증 3: 플레이스홀더 (ㅇㅇㅇ, ○○○)
# ============================================================


def verify_placeholder(texts: dict):

    print("\n")
    print("=" * 80)
    print("[검증 3] 플레이스홀더(ㅇㅇㅇ, ○○○) — 제거/마스킹 대상 확인")
    print("=" * 80)

    doc_counts = _count_per_doc(texts, PATTERN_PLACEHOLDER)
    top_doc = doc_counts.index[0]

    print(f"대표 문서(최다 빈도): {top_doc}  ({doc_counts.iloc[0]}건)\n")

    show_char_context(texts[top_doc], PATTERN_PLACEHOLDER)


# ============================================================
# 7. 검증 4: 페이지번호라인 (정규식 과매칭 의심)
# ============================================================


def verify_page_number_lines(texts: dict):

    print("\n")
    print("=" * 80)
    print("[검증 4] 페이지번호라인 — 정규식 과매칭(표 숫자 셀 오탐) 의심")
    print("=" * 80)

    doc_counts = _count_page_number_lines_per_doc(texts)
    top_doc = doc_counts.index[0]

    print(f"대표 문서(최다 빈도): {top_doc}  ({doc_counts.iloc[0]}건)\n")

    show_line_context(texts[top_doc], PATTERN_PAGE_NUMBER)


# ============================================================
# 8. 검증 5: boilerplate 49건 (라인 목록 자체를 확인 — 대표 문서 개념 적용 불가)
# ============================================================


def verify_boilerplate_lines(
    texts: dict,
    min_doc_ratio: float = 0.2,
    min_line_length: int = 8,
):

    print("\n")
    print("=" * 80)
    print("[검증 5] 문서 간 반복 boilerplate 라인 — 전체 목록")
    print("=" * 80)

    total_docs = len(texts)
    line_doc_map: dict[str, set] = {}

    for filename, text in texts.items():
        seen_in_doc = set()

        for line in text.splitlines():
            normalized = re.sub(r"\s+", " ", line.strip())

            if len(normalized) < min_line_length or normalized in seen_in_doc:
                continue

            seen_in_doc.add(normalized)
            line_doc_map.setdefault(normalized, set()).add(filename)

    threshold = max(2, int(total_docs * min_doc_ratio))

    rows = [
        {"라인": line, "등장문서수": len(docs)}
        for line, docs in line_doc_map.items()
        if len(docs) >= threshold
    ]

    df = pd.DataFrame(rows).sort_values("등장문서수", ascending=False)

    print(f"boilerplate 후보: {len(df)}건\n")
    print(df.to_string(index=False))

    return df


# ============================================================
# 9. 전체 실행
# ============================================================


def run_all_verifications(texts: dict) -> pd.DataFrame:
    verify_angle_brackets(texts)
    verify_bullet_dot(texts)
    verify_placeholder(texts)
    verify_page_number_lines(texts)
    return verify_boilerplate_lines(texts)


# ============================================================
# 10. 실행 예시
# ============================================================
#
# texts: dict[str, str]  # run_a2()에서 얻은 결과
# boilerplate_df = run_all_verifications(texts)
