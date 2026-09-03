"""근거를 들고 있는 평가 세트를 발췌 파일로 바꾼다. **검색을 안 돌린다.**

    python scripts/retrieval/contexts_from_evidence.py generation_qaset_100.jsonl

`generator.py`/`sampler.py` 가 만든 세트는 `question`·`answer`·`evidence_text`
를 다 갖고 있다. 정답 근거가 파일 안에 있으므로 **검색할 이유가 없다** —
코퍼스도 색인도 TEI 도 필요 없다. 합성 문서라 코퍼스에 아예 없는 경우에도 잰다.

무엇을 재는가: **검색을 뺀 생성 상한.** 발췌가 완벽할 때 모델이

    - 근거 안에 있는 것만 말하는가 (충실성)
    - `[n]` 을 다는가, 그 번호가 맞는가 (인용표시율·인용정확도)
    - 근거에 없으면 없다고 하는가 (물러섬)

세 번째가 이 세트의 진짜 값이다. `question_type: 없음` 문항은 근거가 빈
섹션 머리뿐이라 답할 수 없다. 실제 서비스에서 제일 위험한 건 그때 지어내는 것이다.

**발췌를 하나만 주면 안 된다.** 그러면 `[1]` 밖에 못 달아 인용정확도가 늘 1.0 이
된다. 다른 문항의 근거를 섞어 넣어 고르게 만든다(기본 3개, 정답 자리는 무작위).

이 파일은 `retriever.py --export` 와 **같은 스키마**를 내므로 그다음은 똑같다.

    python scripts/retrieval/answer.py outputs/eval_results/contexts_xxx.jsonl --model mini
    python scripts/retrieval/score_answers.py outputs/eval_results/answers_xxx.jsonl --judge
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from config import settings  # noqa: E402


def load(path):
    """jsonl 이든 json 배열이든 받는다."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("evalset", help="근거를 들고 있는 jsonl / json")
    parser.add_argument("--out", help="생략하면 outputs/eval_results/contexts_<이름>.jsonl")
    parser.add_argument("--excerpts", type=int, default=3, help="발췌 개수 (정답 1 + 나머지)")
    parser.add_argument("--seed", type=int, default=0, help="섞는 순서를 고정한다")
    args = parser.parse_args()

    rows = load(args.evalset)
    out = Path(args.out) if args.out else (
        settings.EVAL_RESULTS / f"contexts_{Path(args.evalset).stem}.jsonl"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    pool = [str(r.get("evidence_text") or "").strip() for r in rows]
    pool = [p for p in pool if p]
    rng = random.Random(args.seed)
    written = 0

    with open(out, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows, 1):
            gold = str(row.get("evidence_text") or "").strip()
            if not gold:
                print(f"  건너뜀 {i}: evidence_text 가 없습니다")
                continue

            others = [p for p in rng.sample(pool, min(len(pool), args.excerpts * 3))
                      if p != gold][: max(0, args.excerpts - 1)]
            picked = others + [gold]
            rng.shuffle(picked)
            gold_at = picked.index(gold) + 1

            title = str(row.get("doc_id") or "").replace("[SYNTHETIC] ", "")
            context = "\n\n---\n\n".join(
                f"[{n}] {title if text == gold else '다른 공고'}\n{text}"
                for n, text in enumerate(picked, 1)
            )

            # `question_type: 없음` 은 답할 수 없는 문항이다. 물러섬을 재는 자리.
            kind = row.get("type") or row.get("question_type") or "?"
            f.write(json.dumps({
                "qid": f"{kind}-{i:03d}",
                "question": row["question"],
                "type": kind,
                "answerable": row.get("answerable", kind != "없음"),
                "doc_ids": [row.get("doc_id")],
                "keywords": row.get("keywords") or ([row["answer"]] if row.get("answer") else None),
                "context": context,
                "sources": [{"n": gold_at, "doc_id": row.get("doc_id")}],
                "chunks": len(picked),
                "chars": len(context),
            }, ensure_ascii=False) + "\n")
            written += 1

    kinds = {}
    for row in rows:
        kind = row.get("type") or row.get("question_type") or "?"
        kinds[kind] = kinds.get(kind, 0) + 1
    print(f"{written}/{len(rows)}문항 → {out}")
    print(f"  유형 {kinds}")
    print(f"  발췌 {args.excerpts}개 (정답 1 + 다른 문항 근거 {args.excerpts - 1})")
    print("\n검색을 뺀 생성 상한을 재는 것이다. 서비스 숫자와 나란히 두지 말 것.")


if __name__ == "__main__":
    main()
