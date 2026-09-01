"""v2 전처리본의 마크다운 표 문법을 걷어내 v3 를 만든다.

v2 는 표를 마크다운 파이프 표로 렌더한다. 그 결과 전체 글자의 14.8% 가
`|`, `|---|---|`, `<br>` 같은 마크업이다. 청크 예산도 컨텍스트 예산도
이걸 같이 세기 때문에 실제 답이 들어갈 자리가 줄어든다.

내용은 그대로 두고 표기만 v1 스타일로 되돌린 뒤, 기존 체인을 그대로 돌려
배점 적중률이 회복되는지 본다.

    python scripts/strip_markup.py
    python src/chunking.py --docs cleaned_documents_v3 --how recursive --size 1200
    python src/vectorstore.py --chunks cleaned_documents_v3__recursive_1200_200
    python src/vectorstore.py --chunks cleaned_documents_v3__recursive_1200_200__header
    python scripts/compare_retrieval.py --chunks cleaned_documents_v3__recursive_1200_200
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # scripts/retrieval/ 아래다
DEFAULT_SRC = "cleaned_documents_v2-1"
DEFAULT_DST = "cleaned_documents_v3"

_ESCAPED = re.compile(r"\\([|-])")  # 중첩 표의 \| \- 를 푼다
_DEBUG = re.compile(r"\[표 파싱[^\]]{0,200}\]")  # 본문에 새어 든 진단 문구
_SEPARATOR = re.compile(r"^\s*\|(\s*:?-{2,}:?\s*\|)+\s*$", re.MULTILINE)  # |---|---|
_EMPTY_ROW = re.compile(r"^\s*\|(\s*\|)*\s*$", re.MULTILINE)  # |||
_BLANKS = re.compile(r"\n{3,}")


def strip_markup(text):
    """마크다운 표 문법을 없애고 셀 구분만 남긴다.

    셀 경계(` · `)는 남긴다. 행/열 관계까지 지우면 배점표가 뭉개진다.
    `<br>` 은 표 안이면 칸 안의 줄바꿈이라 공백으로, 밖이면 줄바꿈으로 바꾼다.

    Args:
        text: v2 의 `page_content`.

    Returns:
        구분행·빈 행을 지우고 `|` 를 ` · ` 로 바꾼 문자열.
    """
    # 중첩 표는 안쪽 파이프가 이스케이프돼 온다. 먼저 풀어야 아래 규칙이 먹는다.
    text = _ESCAPED.sub(r"\1", text)
    text = _DEBUG.sub("", text)
    text = _SEPARATOR.sub("", text)
    text = _EMPTY_ROW.sub("", text)
    text = re.sub(r"^[\s|·-]*$", "", text, flags=re.MULTILINE)   # 표만 있던 줄이 비면 지운다

    lines = []
    for line in text.split("\n"):
        if "|" in line:
            line = line.replace("<br>", " ")          # 칸 안의 줄바꿈이다
            cells = [c.strip() for c in line.split("|")]
            # 구분 셀(---)은 표 문법이지 내용이 아니다. <br> 때문에 구분행이
            # 내용행과 한 줄로 붙어 오므로 줄이 아니라 칸 단위로 지운다.
            cells = [c for c in cells if c and not re.fullmatch(r":?-{2,}:?", c)]
            line = " · ".join(cells)
        else:
            line = line.replace("<br>", "\n")
        lines.append(line)

    return _BLANKS.sub("\n\n", "\n".join(lines)).strip()


def main():
    parser = argparse.ArgumentParser(description="마크다운 표 문법을 걷어낸다.")
    parser.add_argument("--src", default=DEFAULT_SRC, help="data/processed 의 이름")
    parser.add_argument("--dst", default=DEFAULT_DST)
    args = parser.parse_args()

    processed = ROOT / "data" / "processed"
    SRC, DST = processed / f"{args.src}.jsonl", processed / f"{args.dst}.jsonl"
    if not SRC.exists():
        sys.exit(f"없음: {SRC}")

    before = after = 0
    with open(SRC, encoding="utf-8") as src, open(DST, "w", encoding="utf-8") as dst:
        for line in src:
            row = json.loads(line)
            before += len(row["page_content"])
            row["page_content"] = strip_markup(row["page_content"])
            after += len(row["page_content"])
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"{before:,}자 → {after:,}자 ({after / before - 1:+.1%})")
    print(f"→ {DST}")
    print("\n다음:")
    print(
        "  python src/chunking.py --docs cleaned_documents_v3 --how recursive --size 1200"
    )


if __name__ == "__main__":
    main()
