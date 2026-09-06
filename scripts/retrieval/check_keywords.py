"""`AddKeywords` 사전이 실제로 무슨 일을 하는지 잰다. LLM 도 인덱스도 안 쓴다.

    python scripts/retrieval/check_keywords.py

**의심은 이거다 — 사전이 평가 세트에 맞춰진 것 아닌가.** 키가 11개뿐이고
그중 `평가`·`요구사항`·`얼마`·`기간` 은 세트의 유형 이름과 겹친다. 성적
이득(가중 MRR +0.008)이 노이즈 폭(±0.015)보다 작아 성적만으로는 못 가른다.

성적을 다시 재기 전에 훨씬 싼 것부터 본다.

1. **몇 문항에서 발동하나.** 15문항에서만 발동하면서 1~2문항을 구제했다면
   그 15개에 맞춘 사전이다. 150문항에서 발동하면서 1~2문항이면 그냥 노이즈다.
2. **붙이는 낱말이 정답 근거에 실제로 있나.** 없으면 후보를 엉뚱한 데로 끈다.
   평가 세트의 `keywords`(정답 문자열)와 대조한다.
3. **유형별로 쏠려 있나.** 한 유형에만 걸리면 그 유형에 맞춘 것이다.

이 셋으로 "빼자 / 두자 / 키우자" 가 갈린다.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from evaluation import load_evalset  # noqa: E402
from pieces.expand import AddKeywords  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="AddKeywords 사전이 하는 일을 잰다.")
    parser.add_argument("--evalset", default="eval_qa_both")
    args = parser.parse_args()

    synonyms = AddKeywords.SYNONYMS
    pairs = [p for p in load_evalset(args.evalset) if p.get("question")]
    print(f"{len(pairs)}문항 · 사전 {len(synonyms)}개\n")

    fired = []
    for pair in pairs:
        hit = [key for key in synonyms if key in pair["question"]]
        if hit:
            extra = " ".join(synonyms[key] for key in hit).split()
            fired.append((pair, hit, extra))

    rate = len(fired) / len(pairs)
    print(f"1. 발동  {len(fired)}/{len(pairs)}문항 ({rate:.0%})")
    if rate < 0.2:
        print("   → 드물게 발동한다. 이득이 있었다면 이 좁은 집합에서 나온 것이다")
    elif rate > 0.6:
        print("   → 대부분에서 발동한다. 그런데도 이득이 노이즈 안이면 안 듣는 것이다")

    total = Counter(p.get("type", "?") for p in pairs)
    hit_by = Counter(p.get("type", "?") for p, _, _ in fired)
    print("\n2. 유형별")
    for kind, n in total.most_common():
        got = hit_by.get(kind, 0)
        print(f"   {kind:6s} {got:3d}/{n:3d}  ({got / n:.0%})")

    # 근거에 없는 낱말을 붙이면 후보를 흩뜨린다. 이게 제일 중요하다.
    useful = wasted = 0
    per_key = Counter()
    for pair, hit, extra in fired:
        gold = f"{pair.get('keywords', '')} {pair.get('note', '')}"
        for word in set(extra):
            if word in gold:
                useful += 1
                per_key[word] += 1
            else:
                wasted += 1
    both = useful + wasted or 1
    print("\n3. 붙인 낱말이 정답 근거에 있나")
    print(f"   있음 {useful:4d} ({useful / both:.0%})   ← 도움이 될 수 있는 몫")
    print(f"   없음 {wasted:4d} ({wasted / both:.0%})   ← 후보를 흩뜨리는 몫")
    if useful / both < 0.25:
        print("   → 넷 중 셋이 근거에 없다. 사전이 어림짐작이라는 뜻이다")

    print("\n   근거에 실제로 있던 낱말 (많은 순)")
    for word, n in per_key.most_common(12):
        print(f"     {word:12s} {n:3d}")

    keys = Counter(key for _, hit, _ in fired for key in hit)
    print("\n4. 사전 키가 걸린 문항 수")
    for key in synonyms:
        n = keys.get(key, 0)
        print(f"   {key:8s} {n:3d}{'  ← 한 번도 안 걸림' if n == 0 else ''}")


if __name__ == "__main__":
    main()
