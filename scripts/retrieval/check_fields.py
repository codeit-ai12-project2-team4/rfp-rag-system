"""전처리 → 청크 → 검색단으로 필드가 온전히 건너오는지.  python scripts/retrieval/check_fields.py

표가 네 군데로 흩어져 있어서 `입찰참여시작일` 이 검색단에서 조용히 사라져
있었다. 표를 `preprocessing/fields.py` 하나로 합쳤고, 이 파일이 그게 유지되는지
본다. 망도 모델도 필요 없다.

    추출 결과 행       filename (한글 아님)
    CSV 병합          사업명·발주기관·사업금액 …
    write_jsonl       metadata 에 source + 한글 필드
    from_langchain    한글 → 영문. **11개가 전부 건너와야 한다**
"""

import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import pandas as pd

from preprocessing.fields import CSV_COLUMNS, FIELDS, TO_ENGLISH
from preprocessing.rfp.build import write_jsonl
from preprocessing.rfp.meta import merge_original_metadata
from preprocessing.run import from_langchain

CSV = """공고 번호,공고 차수,사업명,사업 금액,발주 기관,공개 일자,입찰 참여 시작일,입찰 참여 마감일,사업 요약,파일형식,파일명,텍스트
20240330003,0,클라우드 전환 사업,500000000,한국전력공사,2024-03-30,2024-04-01,2024-04-15,요약이다,hwp,가.hwp,잘린텍스트
"""


def test_한_바퀴():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        csv = tmp / "data_list.csv"
        csv.write_text(CSV, encoding="utf-8-sig")

        # 1. 추출 결과 — extract.process_document 가 내놓는 모양
        df = pd.DataFrame([{
            "filename": "가.hwp",
            "extractor": "hwp5-table",
            "clean_text": "본문이다. " * 200,
            "clean_text_for_generation": "표가 살아 있는 본문. " * 200,
            "error_reason": None,
            "_warnings": [],
            "사업기간": "12개월",
        }])
        assert "파일명" not in df.columns, "추출 결과에 한글 키가 남았다"

        # 2. CSV 병합
        merged = merge_original_metadata(df, csv)
        assert merged["메타매칭방식"].iloc[0] == "exact", merged["메타매칭방식"].iloc[0]
        for korean in CSV_COLUMNS:
            if korean == "파일명":
                continue
            assert korean in merged.columns, f"{korean} 컬럼이 안 붙었다"

        # 3. 전처리본 + 청크
        docs = tmp / "documents.jsonl"
        chunks = tmp / "chunks.jsonl"
        write_jsonl(merged, path=docs, enable_chunk_output=True,
                    chunk_output_path=chunks, chunk_size=500, chunk_overlap=100)
        rows = [json.loads(l) for l in chunks.open(encoding="utf-8")]
        assert rows, "청크가 하나도 안 나왔다"
        assert rows[0]["metadata"]["source"] == "가.hwp"

        # 4. 검색단으로 건너오기 — **여기가 예전에 새던 곳이다**
        meta = from_langchain(rows[0])["meta"]
        for korean, ours in FIELDS:
            assert ours in meta, f"{korean} → {ours} 가 검색단에 안 왔다"
        assert meta["title"] == "클라우드 전환 사업", meta
        assert meta["agency"] == "한국전력공사", meta
        assert meta["bid_open_at"] == "2024-04-01", meta   # 예전에 사라지던 필드
        assert str(meta["notice_seq"]) in ("0", "0.0"), meta
        assert meta["doc_id"] == "20240330003-0", meta["doc_id"]
        assert meta["file_name"] == "가.hwp", meta

        # 5. 본문에서 뽑은 필드는 한글 그대로 남는다 (표에 없는 것들)
        assert rows[0]["metadata"]["사업기간"] == "12개월"
        print(f"필드 {len(FIELDS)}개 전부 통과 · doc_id {meta['doc_id']}")


def test_표가_하나인가():
    """세 곳이 같은 표를 보고 있나. 하나라도 손으로 다시 적으면 여기서 걸린다."""
    from preprocessing.rfp.common import ORIGINAL_METADATA_COLUMNS
    from preprocessing.run import COLUMNS, _LANGCHAIN_META

    assert ORIGINAL_METADATA_COLUMNS == CSV_COLUMNS
    assert COLUMNS == TO_ENGLISH
    assert set(_LANGCHAIN_META.values()) == set(TO_ENGLISH.values())
    assert "공개기관" not in CSV_COLUMNS, "CSV 에 없는 유령 컬럼이 돌아왔다"
    print("표 한 개 유지 OK")


def test_생성_컨텍스트():
    """메타데이터가 프롬프트 머리로 실제로 건너가는가. **NaN 이 안 새는가.**

    머리는 `format_context` 가 만든다. 여기 `nan` 이 박히면 모델이 그걸
    사업명으로 읽는다 — 메타데이터가 pandas 에서 오므로 빈 칸이 float('nan') 이다.
    """
    from langchain_core.documents import Document

    from retriever import format_context

    nan = float("nan")
    chunks = [
        Document(page_content="본문 하나", metadata={
            "title": "클라우드 전환 사업", "agency": "한국전력공사",
            "notice_no": "20240330003", "bid_close_at": "2024-04-15 17:00:00",
            "gen": "표가 살아 있는 본문"}),
        # CSV 병합이 안 된 문서. 전부 비어 있다.
        Document(page_content="본문 둘", metadata={
            "title": nan, "agency": None, "notice_no": nan, "bid_close_at": nan}),
    ]
    got = format_context(chunks, generation=True)
    assert "[1] 클라우드 전환 사업 · 한국전력공사 · 20240330003 · 마감 2024-04-15" in got, got
    assert "nan" not in got.lower(), "NaN 이 프롬프트에 샜다"
    # 생성용 본문이 들어갔나 (9/8 결정: 표 구조를 살린 쪽)
    assert "표가 살아 있는 본문" in got, got
    # 인용 번호는 `sources()` 의 n 과 짝이 맞아야 한다
    assert [int(n) for n in re.findall(r"\[(\d+)\]", got)] == [1, 2], got
    print("생성 컨텍스트 OK — 머리 4항목 · NaN 차단 · gen 본문 · 인용 번호")


if __name__ == "__main__":
    test_표가_하나인가()
    test_한_바퀴()
    test_생성_컨텍스트()
    print("\n전부 통과")
