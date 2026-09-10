"""추출 직후 텍스트 정제.

HWP/PDF에서 막 뽑아낸 텍스트에는 청킹과 임베딩을 망치는 잡음이 섞여 있다.
여기서 거르는 것들:

1. 목차 점선 리더(`개요 ......... 1`)와 그 자리에 끼어드는 깨진 글리프
2. 반복되는 머리말/꼬리말(페이지마다 똑같이 나오는 줄)
3. 사용 영역 밖의 사제 문자(HWP 표 구분자로 쓰인 PUA 영역 등)
4. 과도한 공백/빈 줄

정제 강도는 문서마다 다르니 함수를 잘게 쪼개 두었다. 청킹 실험에서
`aggressive` 토글로 A/B 비교할 수 있다.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

# 목차 점선: 마침표/중점/언더바가 4개 이상 이어지는 구간
_DOT_LEADER = re.compile(r"[.·․‥…_∙•]{4,}\s*\d*")
# 목차 페이지번호 필드가 깨진 자리. 한글 문서에서 목차 항목의 쪽번호는
# 필드로 들어가는데, 본문 스트림에서는 라틴 확장B 영역 글자로 새어 나온다.
#     "Ⅰ. 개요誙ȃ1"   "1. 추진 개요 盅ȃ1"
# 라틴 확장B(U+0180~U+02AF)는 한국어 RFP에 정상적으로 나올 일이 없으므로
# 그 앞의 한자 한 글자와 뒤따르는 쪽번호까지 묶어 지운다.
_BROKEN_FIELD = re.compile(r"[⺀-鿿]?[ƀ-ʯ]+\s*\d*")
# 사제 영역(PUA) + 결합용이 아닌 제어문자
_PUA = re.compile(r"[-￰-￿]")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# 줄 전체가 페이지 번호이거나 구분선인 경우
_PAGE_ONLY = re.compile(r"^\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$")
_RULE_ONLY = re.compile(r"^[\s\-–—=_~*]{3,}$")
_MULTI_BLANK = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r"[ \t　]{2,}")


def normalize(text: str) -> str:
    """한글 자모 분리(NFD)를 합치고 제어문자를 털어낸다."""
    text = unicodedata.normalize("NFC", text)
    text = _CTRL.sub("", text)
    text = _PUA.sub("", text)
    return text.replace("\r\n", "\n").replace("\r", "\n")


def strip_toc_leaders(text: str) -> str:
    text = _DOT_LEADER.sub(" ", text)
    return _BROKEN_FIELD.sub(" ", text)


def drop_repeated_lines(text: str, min_repeat: int = 5, max_len: int = 60) -> str:
    """페이지마다 반복되는 짧은 줄(머리말/꼬리말/기관명)을 제거한다.

    min_repeat 이상 등장하고 max_len 이하로 짧은 줄만 지운다.
    긴 문장이 반복되는 건 실제 요구사항 반복일 수 있어 건드리지 않는다.
    """
    lines = text.split("\n")
    counts = Counter(ln.strip() for ln in lines if ln.strip())
    noisy = {
        ln
        for ln, c in counts.items()
        if c >= min_repeat
        and len(ln) <= max_len
        and not ln.endswith(("다.", "함", "음"))
    }
    return "\n".join(ln for ln in lines if ln.strip() not in noisy)


def tidy_whitespace(text: str) -> str:
    lines = []
    for ln in text.split("\n"):
        ln = _MULTI_SPACE.sub(" ", ln).strip()
        if _PAGE_ONLY.match(ln) or _RULE_ONLY.match(ln):
            ln = ""
        lines.append(ln)
    return _MULTI_BLANK.sub("\n\n", "\n".join(lines)).strip()


def clean_text(text: str, aggressive: bool = True) -> str:
    text = normalize(text)
    text = strip_toc_leaders(text)
    if aggressive:
        text = drop_repeated_lines(text)
    return tidy_whitespace(text)
