"""CSV 의 사업금액이 제안요청서 본문에도 있는가.  python scripts/retrieval/check_budget.py

**머리에 금액을 넣기 전에 이걸 먼저 본다.** CSV 금액은 나라장터 API 값
(배정예산 또는 추정가격)이고, 본문이 말하는 사업비와 다를 수 있다. 다르면
모델은 머리를 읽고 답하는데 채점은 "컨텍스트 안에 있으니 맞다"고 센다 —
점수만 오르고 답은 틀린다.

    일치      본문 어딘가에 그 금액이 그대로 있다
    불일치    금액은 있는데 CSV 값과 다르다   ← 이게 많으면 머리에 넣으면 안 된다
    본문없음  본문에 금액 표기가 아예 없다     ← 이건 머리에 넣을 값이 있다는 뜻
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from config import retrieval as cfg
from config import settings

# "1,234,567원" · "1,234백만원" · "123,456천원"
MONEY = re.compile(r"([\d][\d,]{2,})\s*(원|천원|백만원|억원)")
SCALE = {"원": 1, "천원": 1_000, "백만원": 1_000_000, "억원": 100_000_000}


def amounts(text):
    """본문에 적힌 금액을 원 단위로."""
    got = set()
    for number, unit in MONEY.findall(text):
        try:
            got.add(int(number.replace(",", "")) * SCALE[unit])
        except ValueError:
            pass
    return got


def main():
    # 이름은 `.env` 의 DOCS 다. 하드코딩이 아니다 — 그런데 없다고만 하면
    # 이름이 틀린 건지 파일이 없는 건지 알 수가 없다. 옆에 뭐가 있는지 같이 찍는다.
    path = settings.PROCESSED / f"{cfg.DOCS}.jsonl"
    if not path.exists():
        print(f"X 전처리본이 없습니다: {path}")
        print(f"  (이름은 .env 의 DOCS={cfg.DOCS} 에서 옵니다)")
        near = (
            sorted(p.name for p in settings.PROCESSED.glob("*.jsonl"))
            if settings.PROCESSED.exists()
            else []
        )
        if near:
            print(f"  {settings.PROCESSED} 에 있는 것: {near}")
            print(
                "  → .env 의 DOCS 를 이 중 하나로 맞추거나, prepare.py --build 로 만드세요"
            )
        else:
            print(f"  {settings.PROCESSED} 가 비어 있습니다. prepare.py --build 부터.")
        return 1

    tally = Counter()
    mismatched = []
    ratios = []
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        meta = row["metadata"]
        raw = str(meta.get("사업금액") or "").strip()
        if not raw or raw.lower() == "nan":
            tally["CSV금액없음"] += 1
            continue
        try:
            want = int(float(raw.replace(",", "")))
        except ValueError:
            tally["CSV금액이상"] += 1
            continue

        found = amounts(row.get("page_content") or "")
        if not found:
            tally["본문없음"] += 1
        elif want in found:
            tally["일치"] += 1
        else:
            tally["불일치"] += 1
            near = sorted(found, key=lambda v: abs(v - want))[:3]
            mismatched.append((meta.get("사업명") or meta.get("source"), want, near))
            ratios.append(want / near[0] if near[0] else 0)

    total = sum(tally.values())
    if not total:
        # 0건인데 결론을 찍으면 "불일치 0건이니 넣어도 된다" 로 읽힌다.
        print(f"X 전처리본이 비어 있습니다: {path}")
        return 1
    print(f"{path.name} · 문서 {total}건\n")
    for key in ("일치", "불일치", "본문없음", "CSV금액없음", "CSV금액이상"):
        if tally[key]:
            print(f"  {key:<10} {tally[key]:>4}건  ({tally[key] / total:.0%})")

    if mismatched:
        print("\n불일치 예시 (앞 10건) — CSV 값 / 본문에서 가장 가까운 값들")
        for title, want, near in mismatched[:10]:
            print(f"  {str(title)[:38]:<38} {want:>15,} ← {[f'{v:,}' for v in near]}")

    # **어긋난 정도가 얼마인지가 판정을 가른다.** 배가 다르면 다른 값이고,
    # 1% 안쪽이면 기준이 다른 같은 값이다(부가세·부대비). 후자면 라벨을 붙여
    # 넣는 게 맞다 — 본문에 금액이 아예 없는 문서에는 머리가 유일한 출처다.
    if ratios:
        close = sum(1 for r in ratios if 0.99 <= r <= 1.11)
        print(f"\n불일치 {len(ratios)}건 중 CSV/본문 비가 0.99~1.11 인 것: {close}건")
        print("  (1.1 = 부가세, 1.00~1.01 = 부대비·반올림. 그 밖이면 다른 값이다)")

    print(
        "\n본문없음 비율이 높으면 머리에 넣는 게 맞다 — 그 문서엔 다른 출처가 없다."
        "\n불일치가 '기준이 다른 같은 값' 이면 라벨로 갈린다. 배가 다르면 빼야 한다."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
