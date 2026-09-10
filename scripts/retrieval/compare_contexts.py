"""`--export` 로 뽑은 컨텍스트 두 벌을 비교한다. LLM 을 안 쓴다.

    python scripts/retrieval/compare_contexts.py \\
        outputs/eval_results/contexts_v8_gen.jsonl \\
        outputs/eval_results/contexts_v8_search.jsonl

검색은 두 벌이 똑같다 — 다른 건 프롬프트에 들어가는 본문뿐이다. 그래서 여기서
재는 건 성적이 아니라 **비용**이다. 표 마크업을 살리면 같은 6,000자 예산에
발췌가 몇 개 덜 들어가는지, 정답 문자열이 여전히 컨텍스트 안에 있는지.

답변 품질 A/B 는 이 파일들을 generation 에 넣어야 나온다. 이건 그 전에
"넣을 만한가" 를 보는 단계다.
"""

import json
import sys
from collections import defaultdict


def load(path):
    """qid → 레코드."""
    with open(path, encoding="utf-8") as f:
        return {r["qid"]: r for r in map(json.loads, f)}


def hit(record):
    """정답 문자열이 컨텍스트 안에 있나."""
    return any(k and k in record["context"] for k in (record.get("keywords") or []))


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    a, b = load(a_path), load(b_path)

    shared = sorted(set(a) & set(b))
    if len(shared) != len(a) or len(shared) != len(b):
        print(f"⚠ qid 가 다르다: {a_path} {len(a)}개 · {b_path} {len(b)}개 · 겹침 {len(shared)}개")
    print(f"{a_path.split('/')[-1]}  vs  {b_path.split('/')[-1]}   ({len(shared)}문항)\n")

    rows = defaultdict(lambda: defaultdict(list))
    for qid in shared:
        kind = a[qid].get("type", "?")
        for label, rec in (("A", a[qid]), ("B", b[qid])):
            rows[kind][f"청크_{label}"].append(rec["chunks"])
            rows[kind][f"글자_{label}"].append(rec["chars"])
            rows[kind][f"적중_{label}"].append(hit(rec))

    head = f"{'유형':<8}{'문항':>5}{'청크 A':>8}{'청크 B':>8}{'글자 A':>8}{'글자 B':>8}{'적중 A':>8}{'적중 B':>8}"
    print(head)
    print("-" * len(head))
    for kind in list(rows) + ["전체"]:
        if kind == "전체":
            cell = defaultdict(list)
            for k in rows:
                for key, values in rows[k].items():
                    cell[key].extend(values)
        else:
            cell = rows[kind]
        n = len(cell["청크_A"])
        avg = lambda key: sum(cell[key]) / n  # noqa: E731
        print(f"{kind:<8}{n:>5}{avg('청크_A'):>8.2f}{avg('청크_B'):>8.2f}"
              f"{avg('글자_A'):>8.0f}{avg('글자_B'):>8.0f}"
              f"{avg('적중_A'):>8.3f}{avg('적중_B'):>8.3f}")

    lost = [q for q in shared if hit(b[q]) and not hit(a[q])]
    gained = [q for q in shared if hit(a[q]) and not hit(b[q])]
    print(f"\nA 에서만 정답이 사라진 문항 {len(lost)}개  {lost[:8]}")
    print(f"A 에서만 정답이 생긴 문항   {len(gained)}개  {gained[:8]}")

    fewer = sum(a[q]["chunks"] < b[q]["chunks"] for q in shared)
    print(f"\nA 가 발췌를 더 적게 담은 문항 {fewer}/{len(shared)}개"
          f" — 예산은 같은데 본문이 길어지면 근거가 줄어든다")


if __name__ == "__main__":
    main()
