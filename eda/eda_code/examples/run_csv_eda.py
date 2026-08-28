from pathlib import Path

import pandas as pd

# ============================================================
# 0. 설정
# ============================================================

CSV_PATH = Path("/home/spai1216/workspace/data/data_list.csv")


# ============================================================
# 1. CSV 로드
# ============================================================


def load_csv(path: Path) -> pd.DataFrame:
    """CSV 파일을 읽어 Pandas DataFrame으로 반환합니다.

    기본적으로 UTF-8 인코딩으로 시도하며, 인코딩 에러 발생 시 CP949로 재시도합니다.

    Args:
        path (Path): 읽어올 CSV 파일 경로.

    Returns:
        pd.DataFrame: 로드된 데이터프레임.
    """
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp949")

    return df


# ============================================================
# 2. 기본 데이터셋 정보
# ============================================================


def analyze_basic(df: pd.DataFrame) -> None:
    """데이터셋의 전체 행/열 개수 및 컬럼 목록 등 기본 정보를 출력합니다.

    Args:
        df (pd.DataFrame): 분석할 데이터프레임.
    """
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


def analyze_columns(df: pd.DataFrame) -> None:
    """컬럼별 데이터 타입, 결측치 수/비율 및 고유값 수를 분석해 출력합니다.

    Args:
        df (pd.DataFrame): 분석할 데이터프레임.
    """
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


def analyze_duplicates(df: pd.DataFrame) -> None:
    """전체 행의 완전 중복 건수 및 파일명 관련 컬럼의 중복 건수를 분석해 출력합니다.

    Args:
        df (pd.DataFrame): 분석할 데이터프레임.
    """
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


def analyze_unique_values(df: pd.DataFrame, top_n: int = 20) -> None:
    """각 컬럼별 빈도수 상위 N개의 고유값 분포를 출력합니다.

    Args:
        df (pd.DataFrame): 분석할 데이터프레임.
        top_n (int, optional): 출력할 상위 고유값 개수. 기본값은 20.
    """
    print("\n")
    print("=" * 80)
    print("4. 컬럼별 값 분포")
    print("=" * 80)

    for column in df.columns:
        print("\n" + "-" * 70)
        print(f"[{column}]")
        value_counts = df[column].value_counts(dropna=False).head(top_n)
        print(value_counts.to_string())


# ============================================================
# 6. 텍스트 컬럼 길이
# ============================================================


def analyze_text_columns(df: pd.DataFrame) -> None:
    """문자열(object/string) 컬럼들의 글자 수 통계치(describe)를 계산해 출력합니다.

    Args:
        df (pd.DataFrame): 분석할 데이터프레임.
    """
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


def analyze_file_extensions(df: pd.DataFrame) -> None:
    """파일명 또는 확장자 관련 컬럼을 탐색하여 확장자별 빈도수를 집계 및 출력합니다.

    Args:
        df (pd.DataFrame): 분석할 데이터프레임.
    """
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


def show_missing_files(df: pd.DataFrame) -> None:
    """결측치(NaN)가 하나라도 포함된 행을 찾아 파일명과 결측 컬럼 목록을 출력합니다.

    Args:
        df (pd.DataFrame): 결측치를 검사할 데이터프레임.
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

    print(f"파일명 컬럼 후보: {filename_candidates}\n")

    filename_column = filename_candidates[0]

    # --------------------------------------------------------
    # 결측치가 하나라도 있는 행
    # --------------------------------------------------------
    missing_rows = df[df.isna().any(axis=1)].copy()
    print(f"결측치가 있는 행: {len(missing_rows):,}건\n")

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
        print(f"    결측 컬럼: {', '.join(missing_columns)}\n")


if __name__ == "__main__":
    df = run_csv_eda(CSV_PATH)
