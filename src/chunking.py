"""문서를 청크로 자르고 저장한다.

자르는 방법이 검색 성능을 크게 좌우한다. 정답이 두 조각으로 흩어지면
어떤 검색기도 못 붙인다. 세 가지를 넣어 뒀다.

    split_recursive   글자 수로 자르되 문단·문장 경계를 지킨다 (강의 기본)
    split_by_section  RFP 목차 구조(Ⅰ. / 1. / □)를 경계로 자른다
    split_semantic    임베딩으로 주제가 바뀌는 지점을 찾아 자른다 (느리다)

전부 langchain Document 리스트를 받아 Document 리스트를 돌려준다.

## 목차를 먼저 지운다

목차에는 문서의 모든 절 제목이 한자리에 모여 있어서 **어떤 질문에도 조금씩
걸린다.** 실제로 "그럼 참가자격은?" 을 던졌더니 목차 청크가 1위로 올라왔다 —
"입찰 참가자격爜ȃ37" 이라는 글자는 있지만 답은 쪽번호뿐이다. 게다가 한글에서
새어나온 깨진 쪽번호(`爜ȃ`, 라틴확장B)까지 붙어 있다.

`cleaned_documents.jsonl` 은 팀원이 만든 것이라 우리 `drop_toc` 을 안 탔다.
그래서 자른 **뒤에** 목차 청크를 버린다 (`--raw` 로 끌 수 있다).
문서 단위로 지우면 안 된다 — `drop_toc_chunks` 주석 참고.

명령줄로 돌리면 `outputs/chunks/` 에 청크 파일이 떨어진다.

    python src/chunking.py
    python src/chunking.py --docs cleaned_documents --how recursive --size 1500
"""

import argparse
import json
import re
import sys
from pathlib import Path

# `python src/chunking.py` 로 직접 돌릴 때 config 를 찾게 한다.
# import 로 쓸 때는 이미 경로에 있어서 아무 일도 안 한다.
_ROOT = Path(__file__).resolve().parents[1]
for _folder in (_ROOT / "src", _ROOT):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings

# RFP 목차 헤딩 패턴. 위에 있는 것부터 검사한다.
#   Ⅰ. 사업 안내 / 제1장 총칙 / 1. 사업개요 / □ 사 업 명
HEADING_PATTERNS = [
    (1, re.compile(r"^\s*([ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+)\s*[.．]\s*(.{1,40})$")),
    (1, re.compile(r"^\s*(제\s*\d+\s*[장절])\s*[.．]?\s*(.{0,40})$")),
    (2, re.compile(r"^\s*(\d{1,2})\s*[.．]\s*(.{1,40})$")),
    (3, re.compile(r"^\s*([□■◇◆])\s*(.{1,50})$")),
]


def find_headings(text, max_level=2):
    """본문에서 목차 헤딩을 찾는다.

    새 문서 서식에서 헤딩이 잘 안 잡히면 `HEADING_PATTERNS` 에 줄을 추가한다.
    노트북 2번에서 이 함수 출력을 먼저 보는 게 순서다.

    Args:
        text: 문서 본문.
        max_level: 이 단계까지만 헤딩으로 본다. 1이면 `Ⅰ.` 만, 2면 `1.` 까지.

    Returns:
        `(글자 위치, 단계, 제목)` 튜플 리스트. 본문에 나온 순서 그대로다.
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
    """전처리 레코드를 langchain Document 로 바꾼다.

    청킹 전에 한 번 거치는 단계다. 이후로는 전부 Document 로 다룬다.

    Args:
        records: `{"text": ..., "meta": {...}}` 꼴 레코드 리스트.

    Returns:
        Document 리스트. `metadata` 는 레코드의 `meta` 를 그대로 옮긴다.
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


def split_recursive(records, size=1200, overlap=200, add_header=False):
    """글자 수로 자른다. 문단 → 줄 → 문장 순으로 경계를 지킨다.

    강의에서 쓴 `RecursiveCharacterTextSplitter` 와 같다. 강의는 150자였는데
    RFP 는 문서가 훨씬 길어서 1000자 정도가 출발점으로 낫다. 정답은 실험으로.

    Args:
        records: 전처리 레코드 리스트.
        size: 청크 최대 글자 수.
        overlap: 이웃 청크끼리 겹치는 글자 수.
        add_header: True 면 청크 앞에 `[사업명]` 을 붙인다. BM25 에는 해로우니
            기본값 False 를 권한다. 아래 `with_header` 주석 참고.

    Returns:
        Document 리스트.
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

    Args:
        records: 전처리 레코드 리스트.
        embedder: `embed_documents` 를 가진 임베딩 객체.
        threshold_type: 끊는 기준. `percentile` / `standard_deviation` 등.
        threshold: 기준값. percentile 이면 95 는 상위 5% 지점에서 끊는다는 뜻.
        add_header: 청크 앞에 `[사업명]` 을 붙일지.

    Returns:
        Document 리스트.
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
    """청크 앞에 `[사업명] [절 제목]` 을 붙인 새 리스트를 만든다.

    원본 리스트와 Document 는 건드리지 않는다.

    Args:
        chunks: 원본 청크 리스트.
        include_section: True 면 절 제목까지 붙인다.

    Returns:
        머리말이 붙은 새 Document 리스트.
    """
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


# --- 목차 청크 버리기 -------------------------------------------------------


def leader_count(text):
    """한글에서 새어나온 깨진 쪽번호 글자 수.

    v1 전처리본(`cleaned_documents`)에서만 나오는 신호다. 한글 목차의 점선과
    쪽번호가 필드 코드로 들어 있어 글자로 뽑으면 라틴확장B(U+0180~U+024F)로
    샜다. `4. 세부 작성지침爜ȃ35` 처럼 보인다.

    **팀원의 새 파서(v2 계열)는 이 문자를 안 만든다.** 그래서 이 신호만 쓰면
    목차를 하나도 못 잡는다 — 실제로 청크 8,381개 중 0개였고, 목차 청크가
    "예산이 얼마야?" 에 2위로 올라왔다. `toc_lines` 를 같이 쓴다.

    Args:
        text: 청크 본문.

    Returns:
        라틴확장B 글자 수.
    """
    return sum(1 for ch in text if "\u0180" <= ch <= "\u024f")


# 목차 표시. 이게 있어야 그 청크를 목차로 의심한다.
_TOC_MARK = re.compile(r"목\s*차|\[(별지|붙임|첨부)|[·.…⋯]{4,}")
# 점선 안내선. `제안서 효력 ·············· 242` 꼴이다.
# **띄어쓰기 없이 이어진 것만** 본다 — 표 셀 구분자 ` · ` 를 잡으면 안 된다.
_LEADER_DOTS = re.compile(r"[·.…⋯]{4,}")
# "제목 … 12" — 단위 없는 1~3자리 숫자로 끝나는 조각. 쪽번호다.
_PAGE_REF = re.compile(r"[가-힣A-Za-z\)\]][^\n]{0,60}?[\s\t]+(\d{1,3})(?=\s|$)")
# v3 목차는 쪽번호가 아예 없는 것이 많다. 대신 절 표시가 한 줄에 몰려 있다 —
# `Ⅰ. 사업 개요 … Ⅱ. 제안요청 내용 … Ⅲ. 입찰 안내`. 본문에서는 한 줄에
# 절 표시가 셋씩 나오지 않는다. 코퍼스 전체에서 88줄이 걸리고 전부 목차였다.
_SECTION_RUN = re.compile(r"[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]\s*[.=]")


def toc_lines(text, min_refs=1, max_chars=80, many_refs=3):
    """줄마다 목차인지 판정한다. **`toc_ish` 로 걸러진 청크 안에서만 쓴다.**

    목차 줄은 두 가지 꼴이다.

    - 단위 없는 쪽번호로 끝나는 짧은 줄 — `1. 사업일반\t1`, `[양식 2] 서약서\t73`.
      본문은 단위가 붙어서(`49,500천원`, `계약일로부터 150일`) 안 걸린다.
    - 쪽번호가 여럿 모인 긴 줄 — v3 는 목차를 한 줄로 이어 붙인다.
    - 점선 안내선 — `제안서 효력 ·············· 242`. 점이 너무 길어 쪽번호가
      안 잡히므로 점선만으로 판정한다. 다만 **띄어쓰기 없이 이어진 점**만 센다.
      표 셀 구분자 ` · ` 를 잡으면 표가 다 날아간다.

    쪽번호 하나만으로 판정하므로 느슨하다. 그래서 `toc_ish()` 가 목차 표시를
    확인한 청크에만 적용한다. 목차는 한 줄에 한 항목씩 오는 문서가 많아서
    (`제1장 사업 개요\t1`) 줄당 2개를 요구하면 통째로 놓친다.

    Args:
        text: 청크 본문.
        min_refs: 한 줄에 쪽번호가 이만큼 있어야 목차 줄로 본다.
        max_chars: 이보다 긴 줄은 본문으로 본다. 목차 항목은 짧다.
        many_refs: 쪽번호가 이만큼 모여 있으면 길이와 상관없이 목차 줄로 본다.

    Returns:
        줄마다 True/False 리스트.
    """
    out = []
    for line in text.split("\n"):
        stripped = line.strip()
        dotted = bool(_LEADER_DOTS.search(stripped))
        # v3 전처리본은 목차를 한 줄로 이어 붙인다 —
        # `- 추진개요\t3 - 추진방안\t5 - 추진일정\t7 …`. 길이로만 자르면
        # 이게 전부 본문으로 통과한다. 쪽번호가 여럿 모인 줄은 길어도 목차다.
        refs = len(_PAGE_REF.findall(stripped))
        numbered = refs >= many_refs or (
            refs >= min_refs and len(stripped) <= max_chars
        )
        sectioned = len(_SECTION_RUN.findall(stripped)) >= 3
        out.append(dotted or numbered or sectioned)
    return out


def toc_ish(text, min_refs=4):
    """이 청크에 목차가 들어 있나.

    목차 표시(`목차`·`[별지`·`[붙임`·`[첨부`·점선)가 있고, 쪽번호가 여러 개
    모여 있어야 한다. 둘 중 하나만으로는 안 된다 — 쪽번호만 보면
    `1 · 가천대학교 · 35 · 국립한국해양대학교` 같은 **진짜 표**가 걸린다.
    """
    if not _TOC_MARK.search(text):
        return False
    lines = text.split("\n")
    flags = toc_lines(text)
    if any(_LEADER_DOTS.search(line) for line in lines):
        return True  # 점선 안내선은 그것만으로 충분하다
    if any(len(_SECTION_RUN.findall(line)) >= 3 for line in lines):
        return True  # 절 표시가 한 줄에 셋 이상이면 목차다
    return (
        sum(
            len(_PAGE_REF.findall(line))
            for line, f in zip(lines, flags, strict=False)
            if f
        )
        >= min_refs
    )


def drop_toc_chunks(chunks, min_leaders=5, min_chars=80, verbose=False):
    r"""목차 **줄**을 지운다. 목차만 남는 청크는 버린다.

    목차에는 문서의 모든 절 제목이 한자리에 모여 있어서 **어떤 질문에도 조금씩
    걸린다.** 실제로 "그럼 참가자격은?" 을 던졌더니 목차 청크가 1위로 올라왔다 —
    "입찰 참가자격爜ȃ37" 이라는 글자는 있지만 답은 쪽번호뿐이다.

    **청크를 통째로 버리면 안 된다.** 문서 첫 청크는 보통
    `표지 + 사업개요 표 + 목차` 가 한 덩어리다. 통째로 버렸더니 의역 유형 정답
    17개(`49,500천원`, `900,000,000원` …)가 같이 사라져 적중률이
    0.550 → 0.175 로 무너졌다. 목차 줄만 골라 지우고 나머지는 남긴다.

    문서 단위로 `preprocessing.drop_toc` 을 돌리는 것도 안 된다. 팀원이 만든
    `cleaned_documents.jsonl` 은 칸마다 줄이 나뉜 형식이라 문서 전체를 목차로
    보고 **8,743,663자를 200자로 만들어 버린다.**

    신호를 두 개 쓴다 — v1 은 라틴확장B(`leader_count`), v2 계열은
    목차 표시 + 쪽번호 줄(`toc_ish`). 전처리본이 바뀌면 한쪽이 조용히 죽으므로
    **둘 다 본다.**

    Args:
        chunks: 자른 Document 리스트.
        min_leaders: 깨진 쪽번호 글자가 이만큼 있는 청크만 손댄다 (v1 신호).
        min_chars: 목차 줄을 뺀 뒤 이보다 짧으면 목차뿐인 청크로 보고 버린다.
        verbose: 몇 개를 손댔는지 찍을지.

    Returns:
        목차 줄을 걷어낸 새 리스트.
    """
    kept, trimmed, dropped = [], 0, 0
    for chunk in chunks:
        text = chunk.page_content
        by_leader = leader_count(text) >= min_leaders
        by_page = toc_ish(text)
        if not (by_leader or by_page):
            kept.append(chunk)
            continue

        flags = toc_lines(text)
        body = "\n".join(
            line
            for line, is_toc in zip(text.split("\n"), flags, strict=False)
            if not (is_toc and by_page) and leader_count(line) == 0
        ).strip()
        if len(body) < min_chars:
            dropped += 1
            continue
        chunk.page_content = body
        trimmed += 1
        kept.append(chunk)
    if verbose:
        print(
            f"목차 줄 정리: {trimmed}개 청크에서 목차만 걷어냄 · "
            f"{dropped}개는 목차뿐이라 버림"
        )
    return kept


# --- 저장하고 불러오기 -----------------------------------------------------


def save_chunks(chunks, name):
    """청크를 jsonl 로 저장한다. 같은 설정을 다시 자르지 않게.

    Args:
        chunks: 저장할 Document 리스트.
        name: 설정 이름. 파일은 `outputs/chunks/{name}.jsonl` 이 된다.

    Returns:
        저장한 파일 경로.
    """
    settings.make_dirs()
    path = settings.CHUNKS / f"{name}.jsonl"
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
    """저장해 둔 청크를 읽는다.

    Args:
        name: `save_chunks` 에 준 것과 같은 이름.

    Returns:
        Document 리스트.

    Raises:
        FileNotFoundError: 그 이름으로 저장된 청크가 없을 때.
    """
    path = settings.CHUNKS / f"{name}.jsonl"
    with open(path, encoding="utf-8") as f:
        return [
            _row_to_document(row)
            for row in (json.loads(line) for line in f if line.strip())
        ]


def _row_to_document(row):
    """jsonl 한 줄을 Document 로 만든다. 두 가지 스키마를 다 받는다.

    `save_chunks` 는 `{"text", "meta"}` 로 쓰는데, 전처리팀이 주는 청크
    파일은 LangChain 형식(`{"page_content", "metadata"}`)이고 생성용 본문이
    `page_content_for_generation` 으로 하나 더 붙어 있다. 그 본문은
    `metadata["gen"]` 에 실어 나른다 — `format_context(generation=True)` 가
    거기서 꺼내 쓴다.

    Args:
        row (dict): jsonl 한 줄을 파싱한 것.

    Returns:
        Document. 생성용 본문이 없으면 `metadata["gen"]` 도 없다.
    """
    if "text" in row:
        return Document(page_content=row["text"], metadata=row["meta"])

    # 문서 단위에서 쓰는 것과 **같은** 변환기를 쓴다. doc_id·title·agency 가
    # 여기서 만들어지는데, 이걸 건너뛰면 --scoped 도 1단계 평가도 조용히 깨진다.
    from preprocessing.run import from_langchain, tidy_doc_id

    record = from_langchain(row)
    meta = record["meta"]
    meta["doc_id"] = tidy_doc_id(meta["doc_id"])
    order = row["metadata"].get("chunk_index", 0)
    meta["chunk_id"] = f"{meta['doc_id']}::{order:04d}"
    meta["order"] = order
    if row.get("page_content_for_generation"):
        meta["gen"] = row["page_content_for_generation"]
    return Document(page_content=record["text"], metadata=meta)


def chunk_stats(chunks):
    """청크 길이 분포를 잰다. 설정끼리 표로 비교할 때 쓴다.

    Args:
        chunks: Document 리스트.

    Returns:
        청크수·평균·중앙·최소·최대·50자미만·총글자 를 담은 dict.
        빈 리스트를 주면 빈 dict.
    """
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


# --- 명령줄로 돌리기 -------------------------------------------------------


def main():
    """명령줄에서 청크 파일을 만든다.

    `outputs/chunks/{이름}.jsonl` 을 만들고 길이 분포를 찍는다.
    이름에 전처리본과 자르기 설정이 다 들어가므로, 어떤 설정으로 만든
    청크인지 파일 이름만 봐도 알 수 있다.
    """
    parser = argparse.ArgumentParser(
        description="문서를 잘라 outputs/chunks 에 저장한다."
    )
    parser.add_argument(
        "--docs",
        default="documents",
        help="data/processed 안의 jsonl 이름 (확장자 없이)",
    )
    parser.add_argument("--how", default="section", choices=["section", "recursive"])
    parser.add_argument("--size", type=int, default=1200)  # 실측으로 고른 값
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--name", help="저장 이름 (생략하면 설정에서 자동으로 만든다)")
    parser.add_argument(
        "--raw", action="store_true", help="목차·깨진 쪽번호를 안 지운다 (비교용)"
    )
    args = parser.parse_args()

    from preprocessing import load_documents  # 명령줄로 쓸 때만 필요하다

    documents = load_documents(args.docs)
    print(f"문서 {len(documents)}건 ({args.docs})")

    split = split_by_section if args.how == "section" else split_recursive
    chunks = split(documents, size=args.size, overlap=args.overlap)
    if not args.raw:
        chunks = drop_toc_chunks(chunks, verbose=True)

    name = args.name or f"{args.docs}__{args.how}_{args.size}_{args.overlap}"

    # 두 벌을 한 번에 낸다. 자르는 게 비싸고 머리말 붙이는 건 공짜라,
    # 머리말 때문에 전체를 다시 자를 이유가 없다.
    # 평범한 쪽은 BM25 가, 머리말 쪽은 임베딩 인덱스가 쓴다.
    plain_path = save_chunks(chunks, name)
    head_path = save_chunks(with_header(chunks), f"{name}__header")

    stats = chunk_stats(chunks)
    width = max(len(k) for k in stats)
    for key, value in stats.items():
        print(f"  {key:<{width}}  {value:>10,}")
    print(f"\n청크 저장 → {plain_path}")
    print(f"         → {head_path}")
    print("\n다음:")
    print(f"  python src/vectorstore.py --chunks {name}")
    print(f"  python src/vectorstore.py --chunks {name}__header")
    print(f"  python scripts/retrieval/compare_retrieval.py --chunks {name}")


if __name__ == "__main__":
    sys.exit(main())
