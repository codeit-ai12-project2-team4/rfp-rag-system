"""문서 앞머리의 목차를 걷어낸다.

목차에는 문서의 모든 제목 단어가 한자리에 모여 있다. 청크 하나에 그게
통째로 들어가면 거의 모든 질문에 조금씩 걸린다. BM25 에 특히 나쁘다.

찾는 방법 — **줄 모양**으로 본다. 목차 항목은 번호로 시작하고, 짧고,
문장이 아니고, 콜론이 없다. 끝나는 지점은 **항목이 되풀이되는 자리**다.
목차에 있던 'Ⅰ. 사업 안내' 가 다시 나오면 거기부터가 본문이다.

실제로 재 보면 목차는 문서의 0.3~1% 밖에 안 된다. 큰 효과를 기대할 건
아니고, 값싸고 확실한 정리 정도로 보면 된다.
"""

from __future__ import annotations

import re

# 목차 항목처럼 생긴 줄인가
_TOC_NUMBER = re.compile(
    r"^\s*(?:"
    r"[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩIVX]+\s*[.．]"  # Ⅰ.  IV.
    r"|제?\s*\d+\s*[장절]"  # 제1장
    r"|\d{1,2}\s*[.．]"  # 1.  12.
    r"|[가-하]\s*[.．]"  # 가.  나.
    r"|\[?\s*별\s*첨\s*\]?"  # [별첨]
    r")\s*\S"
)
# '목 차', '차 례', 'CONTENTS' 같은 표지 줄
_TOC_MARK = re.compile(
    r"^\s*[-–—<\[]*\s*(?:목\s*차|차\s*례|CONTENTS?)\s*[-–—>\]]*\s*$", re.IGNORECASE
)


def _key(line):
    """'1. 사 업 명' 과 '1. 사업명' 을 같은 것으로 본다."""
    return "".join(line.split())


def looks_like_toc_line(line):
    line = line.strip()
    if not (2 < len(line) <= 60):
        return False
    if ":" in line or "：" in line:  # '사업명: …' 은 본문이다
        return False
    if line.endswith(("다.", "함", "음", "임", "됨", "다")):
        return False
    return bool(_TOC_NUMBER.match(line))


def find_toc_lines(text, min_entries=8, window_chars=8000, max_stray=2):
    """목차 구간의 (첫 줄, 마지막 줄) 인덱스. 없으면 None."""
    lines = text.split("\n")

    # 문서 앞부분만 본다 — 뒤쪽의 짧은 절들을 잘라내지 않게
    limit, used = 0, 0
    for i, line in enumerate(lines):
        used += len(line) + 1
        limit = i
        if used > window_chars:
            break

    start = mark = None
    for i in range(limit + 1):
        if _TOC_MARK.match(lines[i]):
            mark, start = i, i + 1
            break
    if start is None:
        for i in range(limit + 1):
            if looks_like_toc_line(lines[i]):
                start = i
                break
    if start is None:
        return None

    seen = set()
    last_hit = None
    stray = 0
    for i in range(start, min(limit + 1, len(lines))):
        line = lines[i]
        if not line.strip():
            continue
        if looks_like_toc_line(line):
            if _key(line) in seen:  # 되풀이 → 여기부터 본문
                break
            seen.add(_key(line))
            last_hit = i
            stray = 0
        else:
            stray += 1
            if stray > max_stray:  # 본문 줄이 연달아 나오면 끝
                break

    if last_hit is None or len(seen) < min_entries:
        return None
    return (mark if mark is not None else start, last_hit)


def drop_toc(text, **kwargs):
    """(목차 뺀 본문, 잘라낸 목차) 를 돌려준다. 목차가 없으면 (원문, None)."""
    span = find_toc_lines(text, **kwargs)
    if not span:
        return text, None
    start, end = span
    lines = text.split("\n")
    removed = "\n".join(lines[start : end + 1])
    return "\n".join(lines[:start] + lines[end + 1 :]), removed
