df = pd.read_csv(r"/home/spai1216/workspace/data/data_list.csv")
print(df.shape)
print(df.tail(5))
print(df.head(5))
print(df.info())


print(df['텍스트'].str.contains('□').sum())
print(df.loc[0, '텍스트'][:501])

 
DATA_DIR = Path("/home/spai1216/workspace/data/files")
 
PATTERNS = {
    "사업기간": r"□[^가-힣]{0,10}사\s*업\s*기\s*간",
    "참가자격": r"□.{0,15}참가자격",
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
        except Exception as e:
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