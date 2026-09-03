"""파이프라인 진입점. 원본 폴더 → 전처리본 + 청크 jsonl.

원본: `preprocessing/pipeline.py` (전처리팀). 분할 경위는 `src/preprocessing/README.md`.
"""

import json
import traceback
from pathlib import Path

import pandas as pd

from config import retrieval as retrieval_settings
from config import settings as path_settings
from preprocessing.rfp.chunk import chunk_pairs
from preprocessing.rfp.common import FIELD_ALIASES, SUPPORTED_EXTENSIONS
from preprocessing.rfp.extract import process_document
from preprocessing.rfp.meta import merge_original_metadata

# ============================================================
# 18. 전체 파이프라인 실행
# ============================================================


def run_pipeline(
    raw_path: str,
    eda_output_path: str,
    prep_output_path: str,
    metadata_path: str | None = None,
    sample_size: int | None = None,
    enable_chunk_output: bool = False,  # [수정 8] RAG용 청킹 JSONL 추가 생성 여부
    chunk_size: int = retrieval_settings.SIZE,
    chunk_overlap: int = retrieval_settings.OVERLAP,
) -> None:
    """전처리 단계의 파이프라인. EDA 결과물과 전처리 결과물을 파일로 출력함.

    Args:
        raw_path (str): 원본 문서 디렉토리
        eda_output_path (str): EDA 후 cleaned_document 저장할 디렉토리
        prep_output_path (str): 전처리 후 청크 파일을 저장할 디렉토리
        metadata_path (str | None, optional): 메타데이터(csv 파일). Defaults to None.
        sample_size (int | None, optional): 문서의 일부만 샘플링해서 쓸 경우에 사용. Defaults to None.
        enable_chunk_output (bool, optional): 청크 파일을 내놓을지 결정. Defaults to False.
        chunk_size (int, optional): 청크 크기. 기본값은 config 의 SIZE.
        chunk_overlap (int, optional): 청크 겹침. 기본값은 config 의 OVERLAP.

    Returns:
        tuple[pd.DataFrame, dict]: _description_
    """

    eda_output_path = Path(eda_output_path)
    eda_output_path.mkdir(exist_ok=True)
    prep_output_path = Path(prep_output_path)
    prep_output_path.mkdir(parents=True, exist_ok=True)

    files = sorted(
        p
        for p in Path(raw_path).iterdir()
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
                "filename": path.name,
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
            (eda_output_path / f"{path.stem}.txt").write_text(
                row["clean_text"],
                encoding="utf-8",
            )

    df = pd.DataFrame(rows)

    if metadata_path is not None:
        df = merge_original_metadata(df, Path(metadata_path))

    write_jsonl(
        df,
        path=Path(path_settings.PROCESSED / f"{retrieval_settings.DOCS}.jsonl"),
        enable_chunk_output=enable_chunk_output,
        chunk_output_path=Path(
            path_settings.CHUNKS / f"{retrieval_settings.CHUNKS}.jsonl"
        ),
        # chunk_output_path=prep_output_path / "cleaned_documents_chunks.jsonl",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    print(f"\n완료: {len(df)}건 처리, 결과는 {prep_output_path}에 저장됨")


# [수정 8 - RAG 청킹] LangChain의 RecursiveCharacterTextSplitter와 동일한
# 방식(구분자 우선순위: 문단 -> 줄 -> 공백 -> 문자)으로 직접 구현했다.
# langchain을 새 의존성으로 추가하지 않고도 같은 chunk_size/overlap
# 동작을 낸다.


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
    run_pipeline(
        raw_path=path_settings.RAW,
        eda_output_path=path_settings.PROCESSED,
        prep_output_path=path_settings.OUTPUTS,
        metadata_path=path_settings.META_CSV,
        # 안 켜면 전처리본만 나오고 청크가 안 나온다. 그러면 색인을 못 만든다.
        enable_chunk_output=True,
    )
