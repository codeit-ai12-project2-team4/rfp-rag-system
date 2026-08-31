"""두 실행 결과를 나란히 놓고 부호를 센다.

코퍼스나 설정을 바꿔 두 번 재면 표가 두 장 나오는데, 눈으로 대조하면 놓친다.
**칸마다 차이는 노이즈여도 부호가 한쪽으로 쏠리면 그건 신호다.** 189문항에서
18칸 중 17칸이 한쪽이면 우연일 확률이 0.0001 이다.

    python scripts/retrieval/compare_runs.py outputs/eval_results/v4.csv \\
                                             outputs/eval_results/v3.csv
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import pandas as pd  # noqa: E402

TYPES = ["배점", "요구사항", "의역"]


def load(path):
    """성적표 하나를 (유형, 설정) → MRR 로 읽는다.

    Args:
        path (str): csv 경로.

    Returns:
        tuple[pd.DataFrame, pd.Series]: MRR 피벗과 유형별 문항 수.
    """
    frame = pd.read_csv(path)
    pivot = frame.pivot_table(index="설정", columns="유형", values="MRR")
    counts = frame.groupby("유형")["질문수"].first()
    return pivot, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a", help="비교할 성적표 (왼쪽)")
    parser.add_argument("b", help="기준 성적표 (오른쪽)")
    args = parser.parse_args()

    left, counts = load(args.a)
    right, _ = load(args.b)

    kinds = [k for k in TYPES if k in left.columns and k in right.columns]
    setups = [s for s in left.index if s in right.index]
    if not setups:
        sys.exit("두 성적표에 공통인 설정이 없습니다")

    name_a, name_b = Path(args.a).stem, Path(args.b).stem
    print(f"\n{name_a} / {name_b}   (굵은 쪽이 이긴 칸)\n")
    header = "설정".ljust(28) + "".join(k.center(20) for k in kinds)
    print(header)

    wins = {name_a: 0, name_b: 0, "동률": 0}
    for setup in setups:
        cells = []
        for kind in kinds:
            x, y = left.loc[setup, kind], right.loc[setup, kind]
            mark = "<" if x > y else (">" if y > x else "=")
            wins[name_a if x > y else name_b if y > x else "동률"] += 1
            cells.append(f"{x:.3f} {mark} {y:.3f}".center(20))
        print(setup.ljust(28) + "".join(cells))

    total = sum(wins.values())
    print(f"\n칸 {total}개 — {name_a} {wins[name_a]} · {name_b} {wins[name_b]}"
          f" · 동률 {wins['동률']}")

    # 유형별 문항 수로 가중평균. 유형마다 표본이 달라서 단순평균은 왜곡된다
    weights = counts.reindex(kinds).fillna(0)
    print(f"\n가중평균 (문항 수 {dict(weights.astype(int))})")
    for setup in setups:
        avg_a = (left.loc[setup, kinds] * weights).sum() / weights.sum()
        avg_b = (right.loc[setup, kinds] * weights).sum() / weights.sum()
        print(f"  {setup:<28} {avg_a:.3f} / {avg_b:.3f}   {avg_b - avg_a:+.3f}")

    lopsided = max(wins[name_a], wins[name_b])
    if lopsided >= total * 0.8:
        print("\n한쪽으로 쏠렸습니다. 칸마다 차이가 노이즈여도 방향이 일정하면 신호입니다.")
    else:
        print("\n부호가 갈립니다. 두 실행에 실질 차이가 없다고 봅니다.")


if __name__ == "__main__":
    main()
