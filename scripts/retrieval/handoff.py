"""generation 팀에 넘길 질문 세트를 만든다.

평가 세트에는 **정답이 있는 문항만** 남는다. 그런데 생성에서 제일 중요한 건
"컨텍스트에 없으면 없다고 말하는가" 다. 답할 수 없는 문항이 하나도 없으면
그걸 못 잰다.

`build_evalset` 이 `코퍼스에없음` 으로 버린 문항이 정확히 그것이다 — 질문은
멀쩡한데 정답이 코퍼스 어디에도 없다. 버리지 말고 `answerable: false` 로
붙여서 물러섬 시험지로 쓴다.

    python scripts/retrieval/handoff.py
    python src/retriever.py --export --evalset eval_qa_handoff
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from config import settings  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="eval_qa_both.json")
    parser.add_argument(
        "--dropped",
        default="eval_qa_both_dropped.json",
        help="build_evalset 이 남긴 버린 문항 목록",
    )
    parser.add_argument("--out", default="eval_qa_handoff.json")
    args = parser.parse_args()

    rows = json.loads((settings.DATA / args.base).read_text())
    for row in rows:
        row["answerable"] = True

    path = settings.DATA / args.dropped
    if path.exists():
        for row in json.loads(path.read_text()):
            if row.get("_사유") != "코퍼스에없음":
                continue
            rows.append(
                {
                    "question": row["question"],
                    "doc_id": row.get("doc_id"),
                    "type": row.get("type"),
                    "keywords": [],  # 정답이 없다. 채점은 "모른다고 했나" 로 한다
                    "answerable": False,
                }
            )

    out = settings.DATA / args.out
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2))

    kinds = Counter(r.get("type") for r in rows)
    cannot = sum(1 for r in rows if not r["answerable"])
    print(f"{len(rows)}문항 → {out.name}")
    print(f"  답할 수 있음  {len(rows) - cannot}")
    print(f"  답할 수 없음  {cannot}   ← '문서에서 확인되지 않습니다' 가 정답")
    print("  유형별  " + " · ".join(f"{k} {n}" for k, n in kinds.most_common()))


if __name__ == "__main__":
    main()
