"""이 실행이 **새 코드로** 돈 것인지, 발췌가 비지 않았는지 20초로 본다.

    python scripts/retrieval/check_run.py outputs/eval_results/contexts_xxx.jsonl

"결과가 똑같다" 는 보통 성능이 아니라 셋 중 하나다.

    1. API 를 재시작 안 했다      uvicorn 이 옛 모듈을 물고 있다
    2. answer.py 가 건너뛰었다    같은 --out 이면 이미 만든 qid 를 안 다시 만든다
    3. 발췌가 비었다             doc_id 가 코퍼스와 달라 컨텍스트가 빈 문자열이다

컨텍스트 머리만 보면 1번과 3번이 갈린다. 2026-09-03 에 머리를
`[1] 사업명 · 발주기관 · 공고번호 · 마감 · 금액` 으로 바꿨으므로,
공고번호가 안 보이면 옛 코드로 뽑은 것이다.
"""

import json
import sys
from pathlib import Path


def main(path):
    path = Path(path)
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    empty = [r for r in rows if not (r.get("context") or "").strip()]

    print(f"{path.name} · {len(rows)}문항\n")
    print(f"[1] 빈 발췌 {len(empty)}/{len(rows)}")
    if empty:
        print("    → doc_id 가 코퍼스와 다릅니다. 이대로 채점하면 전부 0점입니다.")
        print(f"       예: {[r.get('doc_ids') for r in empty[:2]]}")

    head = next(
        (r["context"].splitlines()[0] for r in rows if (r.get("context") or "").strip()),
        "",
    )
    print(f"\n[2] 컨텍스트 머리\n    {head[:120]}")
    marks = ["공고번호", "마감", "예산", "금액"]
    new = sum(1 for m in ("마감", "금액", "예산") if m in head)
    print(f"    → {'새 코드' if new else '옛 코드'}로 뽑은 발췌입니다"
          f" ({'마감·금액이 보인다' if new else '사업명·발주기관만 있다'})")
    if not new:
        print("       git pull 뒤 sudo systemctl restart bidmate-api 를 했는지 보세요.")

    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
