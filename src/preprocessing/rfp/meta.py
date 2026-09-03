"""메타데이터 추출, 원본 CSV 병합, 팀 공유 스키마 변환.

원본: `preprocessing/pipeline.py` (전처리팀). 분할 경위는 `src/preprocessing/README.md`.
"""

import difflib
import json
import os
import re
import unicodedata
from pathlib import Path

import pandas as pd

from preprocessing.rfp.common import (
    _METADATA_LINE_BOUNDARY,
    FIELD_ALIASES,
    METADATA_VALUE_BOUNDARY,
    METADATA_VALUE_MAX_LEN,
    METADATA_VALUE_MAX_LINES,
    ORIGINAL_METADATA_COLUMNS,
)

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


