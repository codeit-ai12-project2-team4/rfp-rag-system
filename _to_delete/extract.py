#!/usr/bin/env python
"""원본 hwp / pdf 에서 본문을 뽑는다.

    python scripts/extract.py
    python scripts/extract.py --keep-noise      정제를 약하게 (비교용)

만드는 것
    data/processed/documents.jsonl         본문 + 메타
    outputs/reports/preprocessing_report.json   문서별 추출 결과 (엑셀로 열어 보기 좋다)
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))   # 평평한 import: chunking, preprocessing …
sys.path.insert(0, str(ROOT))           # config.settings


from config import settings
from preprocessing import build_documents, documents_table, load_metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--keep-noise",
        action="store_true",
        help="반복되는 머리말/꼬리말을 지우지 않는다",
    )
    args = parser.parse_args()

    print("원본 폴더:", settings.RAW)
    print("메타 CSV :", settings.META_CSV)
    if not settings.META_CSV.exists():
        raise SystemExit(
            f"\n{settings.META_CSV} 가 없습니다. data/ 에 원본을 먼저 놓으세요."
        )
    print()

    documents = build_documents(aggressive_clean=not args.keep_noise)
    table = documents_table(documents)

    # CSV 텍스트와 비교해서 얼마나 더 뽑았는지
    csv_lengths = load_metadata().set_index("doc_id")["텍스트"].fillna("").str.len()
    table["CSV글자수"] = table["doc_id"].map(csv_lengths).fillna(0).astype(int)
    table["배수"] = (table["글자수"] / table["CSV글자수"].clip(lower=1)).round(1)

    out = settings.PREPROCESSING_REPORT
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_json(out, orient="records", force_ascii=False, indent=2)

    print()
    print("=" * 60)
    print("추출 방법별 건수")
    print(table["추출방법"].value_counts().to_string())
    print()
    print("본문 길이 (글자)")
    print(table["글자수"].describe().round(0).to_string())

    real = table[table["추출방법"].isin(["hwp5-ole", "pdfplumber"])]
    if len(real):
        print()
        print(f"직접 추출 성공 {len(real)}건")
        print(f"  CSV 텍스트 중앙값 : {real['CSV글자수'].median():>9,.0f}자")
        print(f"  직접 추출 중앙값  : {real['글자수'].median():>9,.0f}자")
        print(f"  배수 중앙값       : {real['배수'].median():>9.1f}배")

    weak = table[table["글자수"] < 1000].sort_values("글자수")
    if len(weak):
        print()
        print(f"[확인 필요] 1,000자 미만 {len(weak)}건")
        print(
            weak[["doc_id", "형식", "글자수", "추출방법", "사업명"]]
            .head(20)
            .to_string(index=False)
        )

    print()
    print(f"리포트 → {out}")
    print("다음: 노트북 1번에서 눈으로 확인하세요")


if __name__ == "__main__":
    main()
