"""답변 파일을 채점한다. 네 지표 중 셋은 LLM 없이 잰다.

    python scripts/retrieval/score_answers.py \\
        outputs/eval_results/answers_v8_gen.jsonl \\
        outputs/eval_results/answers_v8_search.jsonl

    # 충실성까지 (LLM 호출. 문항수 × 파일수 만큼 든다)
    python scripts/retrieval/score_answers.py ... --judge --model nano

지표:
    인용표시율    답변에 [n] 을 하나라도 달았나            공짜
    인용정확도    그 [n] 이 실제 발췌 번호 범위 안인가      공짜
    숫자근거율    답변의 숫자가 발췌 안에 있는 것인가       공짜
    물러섬        근거가 없을 때 모른다고 했나              공짜
    충실성        발췌에 있는 내용만 말했나 (YES/NO)        LLM

**Groundedness 는 따로 안 잰다.** 충실성과 사실상 같은 것을 묻는데 LLM 호출만
두 배가 된다. 문장 단위로 쪼개 재고 싶어지면 그때 붙인다.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

CITE = re.compile(r"\[(\d+)\]")
NUMBER = re.compile(r"[\d,]+")
BACKED_OFF = re.compile(r"확인되지\s*않|찾을\s*수\s*없|명시되어\s*있지\s*않|정보가\s*없")


def cite_marks(text):
    """`[1] [2] …` 에서 번호만 뽑는다."""
    return [int(n) for n in CITE.findall(text)]


def score(row):
    """한 문항의 공짜 지표. 값이 없으면 None — 평균에서 뺀다."""
    answer = row.get("answer") or ""
    context = row.get("context") or ""
    available = set(cite_marks(context))  # 발췌 머리의 [1] [2] …
    used = cite_marks(answer)

    numbers = {n.replace(",", "") for n in NUMBER.findall(answer) if len(n) > 1}
    # 인용 번호는 숫자 근거에서 뺀다 — 그건 본문 사실이 아니다
    numbers -= {str(n) for n in used}
    in_context = {n.replace(",", "") for n in NUMBER.findall(context)}

    return {
        "인용표시율": float(bool(used)),
        "인용정확도": (sum(n in available for n in used) / len(used)) if used else None,
        "숫자근거율": (len(numbers & in_context) / len(numbers)) if numbers else None,
        "물러섬": float(bool(BACKED_OFF.search(answer))),
        "정답포함": float(any(k and k in answer for k in (row.get("keywords") or []))),
    }


def judge_all(rows, model):
    """충실성을 LLM 으로 잰다. YES/NO/판정불가."""
    from evaluation.generation import judge_faithfulness
    from generation import AskableModel

    llm = AskableModel(model)
    verdicts = []
    for i, row in enumerate(rows, 1):
        verdicts.append(
            judge_faithfulness(llm, row["question"], row["context"], row.get("answer") or "")
        )
        if i % 20 == 0 or i == len(rows):
            print(f"    채점 {i}/{len(rows)}", flush=True)
    return verdicts


KEYS = ["인용표시율", "인용정확도", "숫자근거율", "물러섬", "정답포함"]


def main():
    parser = argparse.ArgumentParser(description="답변 채점")
    parser.add_argument("files", nargs="+", help="answers_*.jsonl")
    parser.add_argument("--judge", action="store_true", help="충실성까지 잰다 (LLM)")
    parser.add_argument("--model", default="nano")
    # 화면에 표를 찍는 것 말고 값도 남긴다. UI 가 이걸 읽는다 —
    # 찍힌 표를 파싱하는 건 서식이 한 칸만 바뀌어도 깨진다.
    parser.add_argument("--json", dest="json_out", help="지표를 JSON 으로 저장")
    args = parser.parse_args()

    table = {}
    for path in args.files:
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        scored = [score(r) for r in rows]

        by_type = defaultdict(lambda: defaultdict(list))
        for row, s in zip(rows, scored):
            for key in KEYS:
                if s[key] is not None:
                    by_type[row.get("type", "?")][key].append(s[key])
                    by_type["전체"][key].append(s[key])

        if args.judge:
            print(f"  {Path(path).name} 충실성 채점 중…")
            for row, verdict in zip(rows, judge_all(rows, args.model)):
                if verdict is not None:
                    by_type[row.get("type", "?")]["충실성"].append(float(verdict))
                    by_type["전체"]["충실성"].append(float(verdict))
            judged = len(by_type["전체"].get("충실성", []))
            if judged < len(rows):
                print(f"  ⚠ 판정불가 {len(rows) - judged}개 — 빈 응답이면 토큰 예산을 의심한다")

        table[Path(path).stem] = by_type

    keys = KEYS + (["충실성"] if args.judge else [])
    types = ["배점", "요구사항", "의역", "전체"]
    for kind in types:
        if not any(kind in t for t in table.values()):
            continue
        print(f"\n[{kind}]")
        print(f"  {'지표':<12}" + "".join(f"{name[:22]:>24}" for name in table))
        for key in keys:
            line = f"  {key:<12}"
            for by_type in table.values():
                values = by_type.get(kind, {}).get(key, [])
                line += f"{(sum(values) / len(values)):>24.3f}" if values else f"{'-':>24}"
            print(line)

    if args.json_out:
        out = {
            name: {
                kind: {
                    key: sum(values) / len(values)
                    for key, values in by_key.items()
                    if values
                }
                for kind, by_key in by_type.items()
            }
            for name, by_type in table.items()
        }
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"지표 저장 → {args.json_out}")


if __name__ == "__main__":
    main()
