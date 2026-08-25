"""문서를 청크로 자르기.

자르는 방법이 검색 성능을 크게 좌우한다. 정답이 두 조각으로 흩어지면
어떤 검색기도 못 붙인다.

세 가지를 넣어 뒀다. 노트북 2번에서 눈으로 비교하고 고른다.

    split_recursive   글자 수로 자르되 문단·문장 경계를 지킨다 (강의 기본)
    split_by_section  RFP 목차 구조(Ⅰ. / 1. / □)를 경계로 자른다
    split_semantic    임베딩으로 주제가 바뀌는 지점을 찾아 자른다 (느리다)

전부 langchain Document 리스트를 받아 Document 리스트를 돌려준다.
`chunk.page_content` 와 `chunk.metadata` 로 접근하는 건 강의와 같다.
"""

import json
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import paths

# RFP 목차 헤딩 패턴. 위에 있는 것부터 검사한다.
#   Ⅰ. 사업 안내 / 제1장 총칙 / 1. 사업개요 / □ 사 업 명
HEADING_PATTERNS = [
    (1, re.compile(r"^\s*([ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+)\s*[.．]\s*(.{1,40})$")),
    (1, re.compile(r"^\s*(제\s*\d+\s*[장절])\s*[.．]?\s*(.{0,40})$")),
    (2, re.compile(r"^\s*(\d{1,2})\s*[.．]\s*(.{1,40})$")),
    (3, re.compile(r"^\s*([□■◇◆])\s*(.{1,50})$")),
]


def find_headings(text, max_level=2):
    """(글자 위치, 단계, 제목) 목록을 돌려준다.

    새 문서 서식에서 헤딩이 잘 안 잡히면 HEADING_PATTERNS 에 줄을 추가한다.
    노트북 2번에서 이 함수 출력을 먼저 보는 게 순서다.
    """
    found = []
    offset = 0
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped:
            for level, pattern in HEADING_PATTERNS:
                if level > max_level:
                    continue
                match = pattern.match(stripped)
                if match:
                    title = " ".join(part for part in match.groups() if part).strip()
                    found.append((offset, level, title))
                    break
        offset += len(line) + 1
    return found


def to_documents(records):
    """documents.jsonl 레코드를 langchain Document 로 바꾼다.

    청킹 전에 한 번 거치는 단계다. 이후로는 전부 Document 로 다룬다.
    """
    return [
        Document(page_content=record["text"], metadata={**record["meta"]})
        for record in records
    ]


def _make_chunk(text, source_meta, order, section=None, add_header=False):
    title = source_meta.get("title", "")
    if add_header:
        head = f"[{title}]" + (f" [{section}]" if section else "")
        content = head + "\n" + text
    else:
        content = text
    meta = {
        **source_meta,
        "chunk_id": f"{source_meta.get('doc_id')}::{order:04d}",
        "order": order,
    }
    if section:
        meta["section"] = section
    return Document(page_content=content, metadata=meta)


# --- 방법 1. 글자 수로 자르기 --------------------------------------------


def split_recursive(records, size=1000, overlap=150, add_header=False):
    """글자 수로 자르되 문단(\\n\\n) → 줄(\\n) → 문장 순으로 경계를 지킨다.

    강의에서 쓴 RecursiveCharacterTextSplitter 와 같다. 강의는 150자였는데
    RFP 는 문서가 훨씬 길어서 1000자 정도가 출발점으로 낫다. 정답은 실험으로.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = []
    for record in records:
        meta = record["meta"]
        for order, piece in enumerate(splitter.split_text(record["text"])):
            chunks.append(_make_chunk(piece, meta, order, add_header=add_header))
    return chunks


# --- 방법 2. RFP 목차 구조로 자르기 ---------------------------------------


def split_by_section(records, size=1000, overlap=150, add_header=False, recursive=True):
    """헤딩을 경계로 자른다. 절이 너무 길면 그 안에서 다시 글자 수로 자른다.

    왜 이게 나을 수 있나 — 공공 RFP 는 서식이 거의 통일돼 있다.

        Ⅰ. 사업 안내
          □ 사업규모 : 87,000,000원 (VAT포함)
          □ 과업기간 : 계약체결일로부터 120일
        Ⅳ. 제안 요청내용
          요구사항 고유번호   SFR-002
          요구사항 명칭       하이브리드앱 환경 전환

    글자 수로만 자르면 "요구사항 고유번호"와 "명칭"이 다른 청크로 흩어진다.
    헤딩 경계로 자르면 한 질문의 답이 한 청크에 온전히 들어간다.

    다만 절 길이 편차가 크다(수십 자 ~ 수만 자). recursive=True 면
    긴 절만 한 번 더 자른다.

    ※ 이게 항상 이기는 건 아니다. 실제로 재 보면 문서 서식이 덜 정형화된
      경우 글자 수 방식이 더 나을 때도 있다. 노트북 4번에서 숫자로 확인할 것.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = []
    for record in records:
        text = record["text"]
        meta = record["meta"]
        headings = find_headings(text, max_level=2)

        # 문서 처음부터 첫 헤딩까지도 하나의 구역으로 잡는다
        bounds = sorted({(0, "본문"), *((off, title) for off, _, title in headings)})
        bounds.append((len(text), ""))

        order = 0
        for i in range(len(bounds) - 1):
            start, section = bounds[i]
            end = bounds[i + 1][0]
            body = text[start:end].strip()
            if len(body) < 30:
                continue

            pieces = (
                splitter.split_text(body) if recursive and len(body) > size else [body]
            )
            for piece in pieces:
                chunks.append(_make_chunk(piece, meta, order, section, add_header))
                order += 1
    return chunks


# --- 방법 3. 의미로 자르기 -------------------------------------------------


def split_semantic(
    records, embedder, threshold_type="percentile", threshold=95, add_header=False
):
    """임베딩으로 주제가 바뀌는 지점을 찾아 자른다. 강의 L05 의 SemanticChunker.

    문장마다 임베딩을 만들어 이웃과 비교하고, 확 달라지는 곳에서 끊는다.
    **문서 하나당 문장 수만큼 임베딩을 만들어야 해서 느리고 비용도 든다.**
    100건 전체에 돌리기 전에 5건으로 시간을 재 볼 것.
    """
    from langchain_experimental.text_splitter import SemanticChunker

    splitter = SemanticChunker(
        embedder,
        breakpoint_threshold_type=threshold_type,
        breakpoint_threshold_amount=threshold,
    )
    chunks = []
    for record in records:
        meta = record["meta"]
        for order, piece in enumerate(splitter.split_text(record["text"])):
            chunks.append(_make_chunk(piece, meta, order, add_header=add_header))
    return chunks


# --- 청크 앞에 사업명 붙이기 ------------------------------------------------
#
# 청크만 떼어 놓으면 어느 공고 이야기인지 알 수 없다. 그래서 앞에
# "[사업명] [절 제목]" 을 붙이고 싶어진다. 그런데 이게 검색기마다 반대로 작용한다.
#
#   임베딩(Dense)  도움이 된다. 청크에 맥락이 생겨 벡터가 더 정확해진다.
#   BM25          망친다. 한 공고의 모든 청크가 똑같은 사업명을 갖게 되어,
#                 그 공고 전체가 같은 점수로 올라오고 어느 청크가 맞는지
#                 구별하지 못한다.
#
# BM25 기준으로 세 번 쟀다 (적중률@5).
#
#   문서 3건 전문,      질문 3개    붙임 0.00  / 안붙임 0.67
#   문서 100건 잘린본문, 질문 65개   붙임 0.86  / 안붙임 0.85
#   문서 100건 전문,     질문 86개   붙임 0.605 / 안붙임 0.744    ← 실제 데이터
#
# 가운데 줄이 예외였다. 본문이 잘려 있으면 문서당 청크가 몇 개 안 되므로
# 사업명이 반복되는 횟수도 적어 희석이 안 일어난다. 전문으로 뽑으면 문서 하나가
# 청크 100개가 넘고, 그 전부가 같은 사업명을 달게 되어 BM25 가 그 공고 전체를
# 똑같이 올린다. 어느 청크가 맞는지 구별을 못 하는 것이다.
#
# MRR 은 거의 같다(0.391 vs 0.382). 즉 붙이면 **찾긴 찾는데 엉뚱한 청크를**
# 가져온다. 그래서 기본값은 안 붙이는 쪽(add_header=False)이다.
# 임베딩용으로 붙이고 싶으면 아래 함수를 따로 부른다.
#
#     plain  = split_by_section(docs)          # BM25 는 이걸로
#     headed = with_header(plain)              # 임베딩 인덱스는 이걸로
#
# 노트북 2번에서 직접 재 보고 정할 것.


def with_header(chunks, include_section=True):
    """청크 앞에 [사업명] [절 제목] 을 붙인 새 리스트를 만든다. 원본은 안 건드린다."""
    out = []
    for chunk in chunks:
        title = chunk.metadata.get("title", "")
        section = chunk.metadata.get("section", "") if include_section else ""
        head = f"[{title}]" + (f" [{section}]" if section else "")
        out.append(
            Document(
                page_content=head + "\n" + chunk.page_content,
                metadata=dict(chunk.metadata),
            )
        )
    return out


# --- 저장하고 불러오기 -----------------------------------------------------


def save_chunks(chunks, name):
    """청크를 파일로. 같은 설정을 다시 자르지 않게."""
    paths.make_dirs()
    path = paths.PROCESSED / f"chunks_{name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(
            json.dumps(
                {"text": chunk.page_content, "meta": chunk.metadata},
                ensure_ascii=False,
            )
            + "\n"
            for chunk in chunks
        )
    return path


def load_chunks(name):
    path = paths.PROCESSED / f"chunks_{name}.jsonl"
    with open(path, encoding="utf-8") as f:
        return [
            Document(page_content=row["text"], metadata=row["meta"])
            for row in (json.loads(line) for line in f if line.strip())
        ]


def chunk_stats(chunks):
    """청크 길이 분포. 표로 비교할 때 쓴다."""
    lengths = [len(c.page_content) for c in chunks]
    if not lengths:
        return {}
    lengths_sorted = sorted(lengths)
    return {
        "청크수": len(chunks),
        "평균": int(sum(lengths) / len(lengths)),
        "중앙": lengths_sorted[len(lengths) // 2],
        "최소": min(lengths),
        "최대": max(lengths),
        "50자미만": sum(1 for x in lengths if x < 50),
        "총글자": sum(lengths),
    }
