"""컨텍스트 파일을 읽어 답변을 만든다. A/B 의 생성 쪽.

    python scripts/retrieval/answer.py outputs/eval_results/contexts_v8_search.jsonl
    python scripts/retrieval/answer.py outputs/eval_results/contexts_v8_gen.jsonl --model mini

`--out` 을 생략하면 `contexts_` 를 `answers_` 로 바꾼 이름으로 저장한다.

**중단해도 된다.** 이미 있는 결과는 건너뛰고 이어 쓴다. 191문항 × API 호출이라
한 번 끊기면 다시 다 부르는 게 아깝다. 한 줄 끝날 때마다 flush 한다.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from generation import generate_answer  # noqa: E402


def done_qids(path):
    """이미 만들어 둔 qid. 파일이 없으면 빈 집합."""
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)["qid"] for line in f if line.strip()}


def main():
    parser = argparse.ArgumentParser(description="컨텍스트 → 답변")
    parser.add_argument("contexts", help="--export 로 뽑은 jsonl")
    parser.add_argument("--model", default="mini", help="config.MODEL_CONFIGS 의 키")
    parser.add_argument("--out")
    parser.add_argument("--limit", type=int, help="앞에서 몇 개만 (연습용)")
    args = parser.parse_args()

    src = Path(args.contexts)
    out = Path(args.out) if args.out else src.with_name(
        src.name.replace("contexts_", "answers_")
    )

    with open(src, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        rows = rows[: args.limit]

    already = done_qids(out)
    todo = [r for r in rows if r["qid"] not in already]
    print(f"{src.name} → {out.name}")
    print(f"  {len(rows)}문항 중 {len(already)}개는 이미 있음 · {len(todo)}개 생성\n")

    started = time.time()
    failed = 0
    with open(out, "a", encoding="utf-8") as f:
        for i, row in enumerate(todo, 1):
            result = generate_answer(args.model, row["question"], row["context"])
            if not result["ok"] or not (result["answer"] or "").strip():
                failed += 1
                print(f"  !! {row['qid']}  {result.get('error') or '빈 답변'}")
            f.write(json.dumps({
                "qid": row["qid"],
                "type": row.get("type"),
                "question": row["question"],
                "keywords": row.get("keywords"),
                "answerable": row.get("answerable", True),
                "context": row["context"],   # 채점(충실성)이 이걸 본다
                "answer": result["answer"],
                "ok": result["ok"],
                "model": result["model"],
                "usage": result["usage"],
                "error": result["error"],
            }, ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  {time.time() - started:.0f}초")

    print(f"\n{out}  (실패·빈답변 {failed}개)")
    if failed:
        print("빈 답변이 많으면 max_completion_tokens 를 의심한다 (추론 토큰이 먼저 먹는다)")


if __name__ == "__main__":
    main()
