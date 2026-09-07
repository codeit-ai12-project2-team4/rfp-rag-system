"""발췌가 원문 어디인지 말해 줄 수 있나. **붙이기 전에 되는지부터 잰다.**

    python scripts/retrieval/check_locate.py --demo    로직만 (몇 초)
    python scripts/retrieval/check_locate.py
    python scripts/retrieval/check_locate.py --show 3

`/ask` 뒤에 따라붙일 기능이다. "이 근거는 원문 어디에 있나."

**쪽수는 포기했다.** 이유가 데이터에 있다.

    폼피드(\\f)  0건        페이지 경계가 추출에서 사라졌다
    hwp 121건 · hwpx 22건 · pdf 10건

PDF 만 쪽이 파일 안에 있다. **HWP·HWPX 는 없다** — 쪽 나눔은 저장되는 값이 아니라
한글이 그릴 때 계산한다. 93%에서 쪽을 얻으려면 LibreOffice 로 렌더링해야 하는데,
그 쪽 나눔은 한글의 것과 다르다. **사용자는 원본을 한글로 열어 찾는다.** 다른
숫자를 주면 없느니만 못하다.

**대신 제목줄을 쓴다. 본문에 글자로 남아 있다.**

    Ⅳ. 입찰관련사항 > 2. 제안서 평가방법 > 가. 기술평가 배점

쪽수보다 낫다 — 이 제목을 원본에서 Ctrl+F 하면 바로 그 자리다. 쪽수는 스크롤해야
한다. 재전처리도 재색인도 LLM 도 필요 없다.

덤으로 목차가 파싱되면 쪽 범위도 붙는다. **목차의 쪽수는 저자가 한글에서 보고
적은 값이라 사용자가 열었을 때와 일치한다.** 렌더링 논쟁이 통째로 사라진다.

이 스크립트가 재는 것 — **문서의 아무 자리를 집었을 때 제목 경로가 나오나.**

    2단계 이상 90%+   붙인다
    1단계라도 90%+    붙이되 경로가 얕다고 알린다
    그 아래           접는다. 전처리에서 제목을 태그로 남겨야 한다
"""

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from config import retrieval as cfg  # noqa: E402
from config import settings  # noqa: E402

# 공문 제목 체계. 앞 숫자가 곧 깊이다 (Ⅰ. → 1. → 가. → 1) → (1)).
# **제목줄은 짧고 자기 줄을 통째로 쓴다.** 그 두 조건이 본문 문장을 걸러 낸다.
LEVELS = [
    (1, re.compile(r"^[ \t]*([ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ][.)]?[ \t]*\S[^\n]{0,38})[ \t]*$", re.M)),
    (1, re.compile(r"^[ \t]*(제[ \t]*\d{1,2}[ \t]*[장절][ \t]*\S[^\n]{0,38})[ \t]*$", re.M)),
    (2, re.compile(r"^[ \t]*(\d{1,2}[.)][ \t]*\S[^\n]{0,38})[ \t]*$", re.M)),
    (3, re.compile(r"^[ \t]*([가-힣][.)][ \t]*\S[^\n]{0,38})[ \t]*$", re.M)),
    (4, re.compile(r"^[ \t]*(\d{1,2}\)[ \t]*\S[^\n]{0,38})[ \t]*$", re.M)),
    (5, re.compile(r"^[ \t]*(\([0-9가-힣]{1,2}\)[ \t]*\S[^\n]{0,38})[ \t]*$", re.M)),
]


def headings(text):
    """제목줄을 위치 순으로 모은다.

    Args:
        text: 문서 전문.

    Returns:
        [(위치, 깊이, 제목)] — 위치 오름차순.
    """
    marks = []
    for depth, pattern in LEVELS:
        for m in pattern.finditer(text):
            marks.append((m.start(), depth, m.group(1).strip()))
    marks.sort()
    return marks


def path_at(marks, offset):
    """그 자리의 제목 경로.

    위에서부터 훑으며 각 깊이의 최신 제목만 남긴다. **상위가 바뀌면 하위는
    버린다** — `Ⅳ` 로 넘어갔는데 `Ⅲ` 아래의 `가.` 가 남아 있으면 거짓말이 된다.

    Args:
        marks: `headings()` 결과.
        offset: 문서 안 글자 위치.

    Returns:
        ["Ⅳ. 입찰관련사항", "2. 제안서 평가방법", ...] 얕은 것부터.
    """
    open_at = {}
    for pos, depth, title in marks:
        if pos > offset:
            break
        open_at = {d: t for d, t in open_at.items() if d < depth}
        open_at[depth] = title
    return [open_at[d] for d in sorted(open_at)]


# "- 제안요청내용\t 24" · "Ⅳ. 입찰관련사항 ......... 78"
TOC_LINE = re.compile(
    r"^[\s\-·•ㅁ□○◦]*"
    r"(?:[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ0-9]{1,4}[.)]\s*)?"
    r"(?P<title>[^\s.·][^\n]{1,38}?)"
    r"[\s.·…\t]{2,}"
    r"(?P<page>\d{1,3})\s*$",
    re.M,
)


def toc(text, head_chars=6000):
    """앞부분에서 `제목 → 시작쪽`. 목차는 맨 앞에 있고, 뒤까지 훑으면 표가 딸려 온다."""
    found = {}
    for m in TOC_LINE.finditer(text[:head_chars]):
        title = m.group("title").strip(" .·…\t")
        page = int(m.group("page"))
        if len(title) >= 2 and 0 < page < 500 and title not in found:
            found[title] = page
    items = sorted(found.items(), key=lambda kv: kv[1])
    return items if len(items) >= 3 else []


def page_of(text, offset, entries):
    """목차로 쪽 범위를 어림한다. 없으면 None.

    장 시작 쪽은 저자가 적은 값이라 정확하고, **장 안은 어림이다** (표가 많은
    장은 쪽당 글자가 적다). 그래서 한 쪽이 아니라 범위로 돌려준다.
    """
    marks = []
    for title, page in entries:
        i = text.find(title, 2000)  # 목차 자신은 건너뛴다
        if i > 0:
            marks.append((i, page))
    marks.sort()
    for k, (i, page) in enumerate(marks):
        end_i = marks[k + 1][0] if k + 1 < len(marks) else len(text)
        end_p = marks[k + 1][1] if k + 1 < len(marks) else page + 10
        if i <= offset < end_i:
            return page, end_p
    return None


def main():
    parser = argparse.ArgumentParser(description="발췌 → 원문 위치를 말할 수 있나")
    parser.add_argument("--docs", default=cfg.DOCS, help="data/processed 의 전처리본")
    parser.add_argument("--samples", type=int, default=20, help="문서당 찍어 볼 자리")
    parser.add_argument("--show", type=int, default=2, help="예시로 보여줄 문서 수")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    path = settings.PROCESSED / f"{args.docs}.jsonl"
    docs = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    print(f"{path.name} · 문서 {len(docs)}건\n")

    rng = random.Random(args.seed)
    depths = Counter()
    with_page = total = 0
    per_doc = []

    for d in docs:
        text = d["page_content"]
        marks = headings(text)
        entries = toc(text)
        deep = 0
        for _ in range(args.samples):
            offset = rng.randrange(len(text)) if len(text) > 1 else 0
            got = path_at(marks, offset)
            depths[min(len(got), 4)] += 1
            total += 1
            deep += len(got) >= 2
            if entries and page_of(text, offset, entries):
                with_page += 1
        per_doc.append((deep / max(1, args.samples), d, marks, entries))

    print("=" * 74)
    print(f"① 아무 자리를 집었을 때 제목 경로가 나오나 ({total:,}자리)")
    print("=" * 74)
    for n in sorted(depths):
        label = "못 찾음" if n == 0 else f"{n}단계" + ("+" if n == 4 else " ")
        share = depths[n] / total * 100
        print(f"  {label:<8} {depths[n]:>6}  {share:>5.1f}%  {'#' * round(share / 2)}")
    two_plus = sum(v for k, v in depths.items() if k >= 2) / total * 100
    one_plus = sum(v for k, v in depths.items() if k >= 1) / total * 100
    print(f"\n  1단계 이상  {one_plus:.1f}%")
    print(f"  2단계 이상  {two_plus:.1f}%   <- 이 값이 기능의 운명")
    print(f"  쪽 범위까지 {with_page / total * 100:.1f}%  (목차가 파싱된 문서)")
    print()
    if two_plus >= 90:
        print("판정: 붙인다. 경로가 충분히 깊다")
    elif one_plus >= 90:
        print("판정: 붙이되 경로가 얕다. 최상위 장만 말하는 문서가 많다")
    else:
        print("판정: 접는다. 전처리에서 제목을 태그로 남겨야 한다")

    print("\n" + "=" * 74)
    print("② 실제로 어떻게 나오나")
    print("=" * 74)
    per_doc.sort(key=lambda x: -x[0])
    for rate, d, marks, entries in per_doc[: args.show] + per_doc[-1:]:
        name = d["metadata"].get("사업명") or d["metadata"].get("source", "?")
        text = d["page_content"]
        print(f"\n[{str(name)[:46]}]  2단계 이상 {rate:.0%} · 제목 {len(marks)}개")
        for share in (0.3, 0.6, 0.85):
            offset = int(len(text) * share)
            got = path_at(marks, offset)
            span = page_of(text, offset, entries) if entries else None
            where = " > ".join(got) if got else "(못 찾음)"
            tail = f"   [{span[0]}~{span[1]}쪽]" if span else ""
            print(f"  {share:.0%} 지점 -> {where}{tail}")

    print("\n" + "=" * 74)
    print("붙인다면")
    print("=" * 74)
    print("제목 목록은 문서당 한 번만 뽑으면 된다. `prepare.py` 가 doc_id -> 제목")
    print("목록을 json 으로 떨궈 두고 API 가 그걸 읽는다. **요청마다 20MB 원문을")
    print("다시 파싱하지 않는다** — 9/10 에 1단계에서 겪었다(load_chunks).")
    print("발췌 위치는 부모 문서에서 `text.find(발췌 앞 40자)` 로 잡는다.")


def demo():
    """제목 경로와 쪽 범위가 뜻대로 도는지. 파일도 인덱스도 안 쓴다."""
    text = (
        "제안요청서\n\n- 추진개요\t 3\n- 제안요청내용\t 24\n- 입찰관련사항\t 78\n\n"
        + "머리말 " * 400
        + "\nⅠ. 추진개요\n" + "본문 " * 100
        + "\n1. 추진배경\n" + "본문 " * 100
        + "\n가. 필요성\n" + "본문 " * 100
        + "\nⅣ. 입찰관련사항\n" + "본문 " * 100
        + "\n2. 제안서 평가방법\n" + "본문 " * 100
    )
    marks = headings(text)
    titles = [t for _, _, t in marks]
    assert "Ⅰ. 추진개요" in titles and "가. 필요성" in titles, titles

    # 가. 필요성 안 -> 세 단계가 다 열려 있다
    got = path_at(marks, text.index("가. 필요성") + 50)
    assert got == ["Ⅰ. 추진개요", "1. 추진배경", "가. 필요성"], got

    # Ⅳ 로 넘어가면 Ⅰ 아래의 `1.`·`가.` 는 버려야 한다
    got = path_at(marks, text.index("2. 제안서 평가방법") + 50)
    assert got == ["Ⅳ. 입찰관련사항", "2. 제안서 평가방법"], got

    # 첫 제목 앞은 경로가 없다
    assert path_at(marks, 10) == [], path_at(marks, 10)

    entries = toc(text)
    assert [p for _, p in entries] == [3, 24, 78], entries
    span = page_of(text, text.index("Ⅳ. 입찰관련사항") + 50, entries)
    assert span == (78, 88), span
    print("demo ok:", got, "·", span)


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        main()
