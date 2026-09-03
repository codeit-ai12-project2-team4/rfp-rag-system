"""원본 파일 100건 → data/processed/documents.jsonl

## 왜 CSV 의 `텍스트` 컬럼을 안 쓰나

data_list.csv 에 `텍스트` 컬럼이 있어서 그냥 쓰면 이 단계를 건너뛸 수 있다.
그런데 그 컬럼은 잘려 있다. 평균 3,843자, 최소 89자.

실제로 대조해 본 결과:

    한국발명진흥회 (hwp)    CSV 89자    →  직접 뽑으면 51,164자
    대한상공회의소 (hwp)    CSV 2,036자  →  직접 뽑으면 48,772자
    기초과학연구원 (pdf)    CSV 2,716자  →  직접 뽑으면 65,759자

사업개요(예산·과업기간·계약방식)까지는 CSV 에도 들어 있다. 빠진 건
**요구사항 명세, 참가자격, 제출서류, 평가배점** 이다. 컨설턴트가 알아야 하는
부분이 정확히 그것이다. CSV 텍스트로 만들면 "문서 앞부분만 아는" 시스템이
되는데, 겉으로는 잘 도는 것처럼 보여서 더 위험하다.

그래서 원본에서 직접 뽑고, 실패한 문서만 CSV 텍스트로 되돌린다.
어떤 경로로 뽑았는지는 `extractor` 칸에 남겨서 나중에 셀 수 있게 한다.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

# `python src/preprocessing/run.py` 로 직접 돌릴 때 config 와 src 를 찾게 한다.
# import 로 쓸 때는 이미 경로에 있어서 아무 일도 안 한다.
_ROOT = Path(__file__).resolve().parents[2]
for _folder in (_ROOT / "src", _ROOT):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import pandas as pd

from config import settings
from preprocessing.clean import clean_text
from preprocessing.fields import FIELDS, TO_ENGLISH, normalize_columns
from preprocessing.hwp import HwpParseError, extract_hwp_preview_text, extract_hwp_text
from preprocessing.hwp_table import extract_hwp_tables
from preprocessing.pdf import PdfParseError, extract_pdf_text, looks_scanned
from preprocessing.toc import drop_toc

# CSV 컬럼 이름을 코드에서 쓸 이름으로. **표는 `preprocessing/fields.py` 하나다.**
# 여기 다시 적으면 언젠가 갈린다 — 실제로 갈려서 bid_open_at 이 사라져 있었다.
COLUMNS = TO_ENGLISH

# 이보다 짧으면 추출 실패로 본다 (표지만 나온 경우)
MIN_CHARS = 1000

# HWP 를 표 구조를 살려 뽑을지. False 면 예전 방식(문단만).
# 두 방식을 나란히 비교하려고 스위치로 뒀다. extractor 칸에 어느 쪽인지 남는다.
USE_TABLE_PARSER = True

# 문서 앞머리 목차를 걷어낼지
DROP_TOC = True


def tidy_doc_id(doc_id):
    """`20240330003.0-0` → `20240330003-0`.

    pandas 가 공고번호 컬럼에 빈칸이 있으면 float 로 읽어서 `.0` 이 붙는다.
    그러면 팀원 전처리본과 doc_id 가 안 맞아 A/B 비교가 깨진다.
    """
    head, dash, tail = str(doc_id).partition("-")
    head = head.removesuffix(".0")
    return f"{head}{dash}{tail}"


def make_doc_id(row):
    """공고번호가 있으면 그걸로, 없으면 파일명 해시로 아이디를 만든다."""
    notice = row.get("notice_no")
    seq = row.get("notice_seq")
    if pd.notna(notice) and str(notice).strip():
        suffix = f"-{int(seq)}" if pd.notna(seq) else ""
        return tidy_doc_id(f"{str(notice).strip()}{suffix}")
    digest = hashlib.sha1(str(row["file_name"]).encode("utf-8")).hexdigest()[:12]
    return f"nofile-{digest}"


def load_metadata(csv_path=None):
    """data_list.csv 를 읽어 컬럼 이름을 정리하고 doc_id 를 붙인다."""
    df = pd.read_csv(csv_path or settings.META_CSV)
    # 공백을 먼저 지워 정규 이름으로 맞춘 뒤 영문으로 바꾼다. CSV 원본은
    # `공고 번호`, 크롤러가 쓰는 건 같은 이름이지만 언제 바뀔지 모른다.
    df = df.rename(columns=normalize_columns(df.columns)).rename(columns=COLUMNS)
    for column in ("published_at", "bid_open_at", "bid_close_at"):
        df[column] = pd.to_datetime(df[column], errors="coerce")
    df["doc_id"] = df.apply(make_doc_id, axis=1)
    return df


def extract_one(path, file_type):
    """파일 하나에서 본문을 뽑는다. (본문, 방법, 경고들) 을 돌려준다."""
    warnings = []

    if file_type == "hwp":
        try:
            if USE_TABLE_PARSER:
                return extract_hwp_tables(path), "hwp5-table", warnings
            return extract_hwp_text(path), "hwp5-ole", warnings
        except HwpParseError as error:
            warnings.append(f"hwp 본문 파싱 실패: {error}")
            try:
                return extract_hwp_preview_text(path), "prvtext-fallback", warnings
            except HwpParseError as error2:
                warnings.append(f"미리보기 텍스트도 실패: {error2}")
                return "", "failed", warnings

    if file_type == "pdf":
        try:
            if looks_scanned(path):
                warnings.append("스캔 PDF 로 의심됨 — OCR 필요")
            return extract_pdf_text(path), "pdfplumber", warnings
        except PdfParseError as error:
            warnings.append(f"pdf 파싱 실패: {error}")
            return "", "failed", warnings

    warnings.append(f"지원하지 않는 형식: {file_type}")
    return "", "failed", warnings


def build_documents(
    csv_path=None, raw_dir=None, out_path=None, aggressive_clean=True, verbose=True
):
    """원본을 전부 훑어 documents.jsonl 을 만든다. 100건에 몇 분 걸린다."""
    settings.make_dirs()
    raw_dir = raw_dir or settings.RAW
    out_path = out_path or settings.DOCUMENTS_JSONL

    df = load_metadata(csv_path)
    documents = []

    for position, (_, row) in enumerate(df.iterrows(), 1):
        meta = {
            "doc_id": row["doc_id"],
            "notice_no": None if pd.isna(row["notice_no"]) else str(row["notice_no"]),
            "title": str(row["title"]),
            "agency": str(row["agency"]),
            "budget": None if pd.isna(row["budget"]) else float(row["budget"]),
            "published_at": None
            if pd.isna(row["published_at"])
            else row["published_at"].strftime("%Y-%m-%d"),
            "bid_close_at": None
            if pd.isna(row["bid_close_at"])
            else row["bid_close_at"].strftime("%Y-%m-%d"),
            "summary": None if pd.isna(row["summary"]) else str(row["summary"]),
            "file_type": str(row["file_type"]).lower(),
            "file_name": str(row["file_name"]),
        }

        path = raw_dir / meta["file_name"]
        if not path.exists():
            text, extractor, warnings = "", "missing", [f"원본 파일 없음: {path.name}"]
        else:
            text, extractor, warnings = extract_one(path, meta["file_type"])

        text = clean_text(text, aggressive=aggressive_clean)
        if DROP_TOC:
            text, cut = drop_toc(text)
            if cut:
                warnings.append(f"목차 {len(cut)}자 제거")

        # 마지막 안전망: 그래도 너무 짧으면 CSV 텍스트라도 쓴다
        if len(text) < MIN_CHARS:
            csv_text = clean_text(str(row.get("텍스트", "") or ""), aggressive=False)
            if len(csv_text) > len(text):
                warnings.append(
                    f"직접 추출 {len(text)}자 → CSV 텍스트 {len(csv_text)}자로 대체"
                )
                text, extractor = csv_text, "csv-fallback"

        documents.append({
            "meta": meta,
            "text": text,
            "chars": len(text),
            "extractor": extractor,
            "warnings": warnings,
        })

        if verbose and position % 20 == 0:
            print(f"  {position}/{len(df)} …")

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(
            json.dumps(document, ensure_ascii=False) + "\n" for document in documents
        )

    if verbose:
        print(f"문서 {len(documents)}건 → {out_path}")
    return documents


# 팀원이 만든 cleaned_documents.jsonl 의 메타 키 → 우리 키
# 파이프라인 청크 메타 → 우리 이름. 파일명은 `source` 로 들어온다.
#
# **손으로 적지 않는다.** 예전엔 여기에 아홉 줄을 직접 적었는데 `공고차수` 와
# `입찰참여시작일` 이 빠져 있었다. CSV 에도 있고 청크 메타에도 남아 있는데
# 검색단에서만 존재하지 않는 필드가 됐다. 표에서 파생하면 그 일이 안 생긴다.
_LANGCHAIN_META = {
    "source": "file_name",
    **{korean: ours for korean, ours in FIELDS if korean != "파일명"},
}


def from_langchain(row):
    """LangChain Document 꼴 `{page_content, metadata}` 을 우리 꼴로 맞춘다.

    팀원이 만든 `cleaned_documents.jsonl` 이 이 형식이다.
    **doc_id 를 우리와 똑같은 규칙(공고번호-차수)으로 만든다.** 그래야 같은
    질문 세트로 두 전처리본을 나란히 비교할 수 있다.
    """
    meta_in = row["metadata"]
    meta = {ours: meta_in.get(theirs) for theirs, ours in _LANGCHAIN_META.items()}
    meta["doc_id"] = make_doc_id(meta)  # notice_seq 도 이제 meta 안에 있다
    return {
        "meta": meta,
        "text": row["page_content"],
        "chars": len(row["page_content"]),
        "extractor": meta_in.get("extractor") or "unknown",
        "warnings": meta_in.get("_warnings") or [],
    }


def load_documents(path=None, only_real=False):
    """전처리 결과 jsonl 을 읽는다. 두 가지 형식을 다 받는다.

        load_documents()                        data/processed/documents.jsonl
        load_documents("cleaned_documents")     data/processed/cleaned_documents.jsonl
        load_documents(Path("/어디/그거.jsonl"))

    우리 형식 `{meta, text, ...}` 과 팀원 형식 `{page_content, metadata}` 을
    자동으로 구분해 우리 꼴로 돌려준다. 그래서 청킹·검색·평가 코드는 안 바뀐다.

    only_real=True 면 CSV 로 되돌아간 문서를 뺀다.
    """
    if path is None:
        path = settings.DOCUMENTS_JSONL
    else:
        path = Path(path)
        if path.suffix == "":  # 이름만 준 경우
            path = settings.PROCESSED / f"{path}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"추출 결과가 없습니다: {path}\n먼저 실행하세요:  python scripts/extract.py"
        )
    with open(path, encoding="utf-8") as f:
        documents = [json.loads(line) for line in f if line.strip()]
    documents = [from_langchain(d) if "page_content" in d else d for d in documents]
    # 이미 만들어 둔 jsonl 에 `.0` 이 붙은 doc_id 가 있을 수 있다. 읽을 때 맞춘다.
    for d in documents:
        d["meta"]["doc_id"] = tidy_doc_id(d["meta"]["doc_id"])
    if only_real:
        skip = ("csv-fallback", "missing", "failed")
        documents = [d for d in documents if d["extractor"] not in skip]
    return documents


def documents_table(documents=None):
    """문서 목록을 표로. 추출 품질을 눈으로 훑을 때."""
    documents = documents if documents is not None else load_documents()
    return pd.DataFrame([
        {
            "doc_id": d["meta"]["doc_id"],
            "사업명": d["meta"]["title"],
            "발주기관": d["meta"]["agency"],
            "예산": d["meta"]["budget"],
            "마감": d["meta"]["bid_close_at"],
            "형식": d["meta"]["file_type"],
            "추출방법": d["extractor"],
            "글자수": d["chars"],
            "경고": " / ".join(d["warnings"]),
        }
        for d in documents
    ])


# --- 명령줄로 돌리기 -------------------------------------------------------


def main():
    """명령줄에서 원본 hwp/pdf 를 뽑아 `data/processed/documents.jsonl` 을 만든다.

    문서별 추출 결과는 `outputs/reports/preprocessing_report.json` 에 남는다.
    CSV 의 `텍스트` 컬럼과 몇 배 차이 나는지도 같이 찍는다.
    """
    parser = argparse.ArgumentParser(
        description="원본 hwp/pdf → data/processed/documents.jsonl"
    )
    parser.add_argument(
        "--keep-noise",
        action="store_true",
        help="반복되는 머리말/꼬리말을 지우지 않는다 (비교용)",
    )
    args = parser.parse_args()

    print("원본 폴더:", settings.RAW)
    print("메타 CSV :", settings.META_CSV)
    if not settings.META_CSV.exists():
        raise SystemExit(f"\n{settings.META_CSV} 가 없습니다. 원본을 먼저 놓으세요.")
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

    print("=" * 60)
    print("추출 방법별 건수")
    print(table["추출방법"].value_counts().to_string())
    print()
    print("본문 길이 (글자)")
    print(table["글자수"].describe().round(0).to_string())

    weak = table[table["글자수"] < 1000].sort_values("글자수")
    if len(weak):
        print(f"\n[확인 필요] 1,000자 미만 {len(weak)}건")
        print(
            weak[["doc_id", "형식", "글자수", "추출방법", "사업명"]]
            .head(20)
            .to_string(index=False)
        )

    print(f"\n리포트 → {out}")
    print("다음:  python src/chunking.py")


if __name__ == "__main__":
    main()
