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


def label(path):
    """표에 쓸 이름. 파일명이 같으면 상위 폴더로 구분한다.

    실행마다 폴더를 만들고 안에는 늘 `scoped.csv` 로 두므로, 파일명만 쓰면
    두 열이 같은 이름이 된다.

    Args:
        path (str): 성적표 경로.

    Returns:
        str: 표에 쓸 이름.
    """
    p = Path(path)
    return f"{p.parent.name}/{p.stem}" if p.parent.name.startswith(("v", "run")) else p.stem


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


def grid(paths):
    """셋 이상을 한 표로. 부호 세기는 둘일 때만 뜻이 있다.

    Args:
        paths (list[str]): 성적표 경로들.
    """
    frames, counts = {}, None
    for path in paths:
        pivot, got = load(path)
        frames[label(path).replace("grid_", "")] = pivot
        counts = got if counts is None else counts

    kinds = [k for k in TYPES if all(k in f.columns for f in frames.values())]
    weights = counts.reindex(kinds).fillna(0)
    setups = sorted(set.intersection(*(set(f.index) for f in frames.values())))

    table = pd.DataFrame(
        {
            name: [
                (frame.loc[setup, kinds] * weights).sum() / weights.sum()
                for setup in setups
            ]
            for name, frame in frames.items()
        },
        index=setups,
    ).round(3)

    print(f"\n가중평균 MRR (문항 수 {dict(weights.astype(int))})\n")
    print(table.to_string())
    print(f"\n설정별 최고: {table.idxmax(axis=1).to_dict()}")
    best = table.max().idxmax()
    print(f"전체 최고: {best}  {table[best].max():.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a", help="비교할 성적표 (왼쪽)")
    parser.add_argument("b", help="기준 성적표 (오른쪽)")
    parser.add_argument("rest", nargs="*", help="셋 이상이면 가중평균 표만 낸다")
    args = parser.parse_args()

    if args.rest:
        grid([args.a, args.b, *args.rest])
        return

    left, counts = load(args.a)
    right, _ = load(args.b)

    kinds = [k for k in TYPES if k in left.columns and k in right.columns]
    setups = [s for s in left.index if s in right.index]
    if not setups:
        sys.exit("두 성적표에 공통인 설정이 없습니다")

    name_a, name_b = label(args.a), label(args.b)
    print(f"\n{name_a} / {name_b}   (부등호가 이긴 쪽을 가리킨다)\n")
    header = "설정".ljust(28) + "".join(k.center(20) for k in kinds)
    print(header)

    wins = {name_a: 0, name_b: 0, "동률": 0}
    for setup in setups:
        cells = []
        for kind in kinds:
            x, y = left.loc[setup, kind], right.loc[setup, kind]
            mark = ">" if x > y else ("<" if y > x else "=")
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

    # **동률은 분모에서 뺀다.** BM25 행은 두 실행에서 구조상 같고(같은 청크·같은
    # BM25), 임베더만 바꾸면 리랭크 행도 같다. 18칸 중 13칸이 동률로 고정이라
    # 동률을 분모에 넣으면 80% 문턱을 원리적으로 못 넘는다 — 승부 난 칸을 5:0 으로
    # 다 이겨도 "차이 없음" 이 찍혔다 (9/8).
    #
    # 그리고 이 도구는 "차이가 없다" 고 **단정하지 않는다.** 방향만 말하고
    # 크기 판단은 위의 가중평균으로 넘긴다.
    decided = wins[name_a] + wins[name_b]
    lopsided = max(wins[name_a], wins[name_b])
    if decided == 0:
        print("\n모든 칸이 동률입니다. 같은 실행이거나, 바꾼 것이 이 구간에 안 닿습니다.")
    elif lopsided >= decided * 0.8:
        winner = name_a if wins[name_a] > wins[name_b] else name_b
        print(f"\n승부 난 {decided}칸 중 {lopsided}칸이 {winner} 쪽입니다. 방향이 일정하면 신호입니다.")
    else:
        print(f"\n승부 난 {decided}칸이 갈립니다 ({wins[name_a]}:{wins[name_b]}). 가중평균으로 판단하세요.")


if __name__ == "__main__":
    main()
