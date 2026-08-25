"""원본 파일 100건 → data/interim/documents.jsonl

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

import hashlib
import json

import pandas as pd

import paths
from ingest.clean import clean_text
from ingest.hwp import HwpParseError, extract_hwp_preview_text, extract_hwp_text
from ingest.hwp_table import extract_hwp_tables
from ingest.pdf import PdfParseError, extract_pdf_text, looks_scanned
from ingest.toc import drop_toc

# CSV 컬럼 이름을 코드에서 쓸 이름으로
COLUMNS = {
    "공고 번호": "notice_no",
    "공고 차수": "notice_seq",
    "사업명": "title",
    "사업 금액": "budget",
    "발주 기관": "agency",
    "공개 일자": "published_at",
    "입찰 참여 시작일": "bid_open_at",
    "입찰 참여 마감일": "bid_close_at",
    "사업 요약": "summary",
    "파일형식": "file_type",
    "파일명": "file_name",
}

# 이보다 짧으면 추출 실패로 본다 (표지만 나온 경우)
MIN_CHARS = 1000

# HWP 를 표 구조를 살려 뽑을지. False 면 예전 방식(문단만).
# 두 방식을 나란히 비교하려고 스위치로 뒀다. extractor 칸에 어느 쪽인지 남는다.
USE_TABLE_PARSER = True

# 문서 앞머리 목차를 걷어낼지
DROP_TOC = True


def make_doc_id(row):
    """공고번호가 있으면 그걸로, 없으면 파일명 해시로 아이디를 만든다."""
    notice = row.get("notice_no")
    seq = row.get("notice_seq")
    if pd.notna(notice) and str(notice).strip():
        suffix = f"-{int(seq)}" if pd.notna(seq) else ""
        return f"{str(notice).strip()}{suffix}"
    digest = hashlib.sha1(str(row["file_name"]).encode("utf-8")).hexdigest()[:12]
    return f"nofile-{digest}"


def load_metadata(csv_path=None):
    """data_list.csv 를 읽어 컬럼 이름을 정리하고 doc_id 를 붙인다."""
    df = pd.read_csv(csv_path or paths.META_CSV)
    df = df.rename(columns=COLUMNS)
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
    paths.make_dirs()
    raw_dir = raw_dir or paths.RAW
    out_path = out_path or paths.DOCUMENTS_JSONL

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


def load_documents(path=None, only_real=False):
    """documents.jsonl 을 읽는다.

    only_real=True 면 CSV 로 되돌아간 문서를 뺀다. 추출이 제대로 된 것만
    가지고 실험하고 싶을 때 쓴다.
    """
    path = path or paths.DOCUMENTS_JSONL
    if not path.exists():
        raise FileNotFoundError(
            f"추출 결과가 없습니다: {path}\n먼저 실행하세요:  python scripts/extract.py"
        )
    with open(path, encoding="utf-8") as f:
        documents = [json.loads(line) for line in f if line.strip()]
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
