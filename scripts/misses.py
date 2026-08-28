"""틀린 문항을 눈으로 보게 펼친다.

적중률 0.675 는 "40개 중 13개를 못 찾았다"만 말한다. **왜** 못 찾았는지는
숫자로 안 나온다. 실패는 세 종류이고 대응이 전부 다르다.

    정답없음   정답 키워드가 코퍼스 어디에도 없다 → 질문이나 정답이 틀렸다
    후보밖     청크는 있는데 후보 30개에 못 든다  → 임베더가 못 잡는다
    순위밀림   후보엔 들었는데 예산 밖으로 밀렸다 → 리랭커/예산 문제

    python scripts/misses.py --chunks cleaned_documents_v3__recursive_1200_200
    python scripts/misses.py --chunks ... --type 배점 --full
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import chunking
from config import settings
from evaluation import fit_budget, load_evalset
from evaluation.evalset import matches
from models import load_embedder, load_reranker
from pieces import Dense, Pipeline, Rerank
from vectorstore import list_stores, load_store


def flat(text, width=200):
    """줄바꿈을 지우고 앞부분만. 한 줄로 봐야 눈으로 훑을 수 있다."""
    return " ".join(text.split())[:width]


def around(text, keyword, width=200):
    """정답 키워드 **주변**을 보여준다. 앞부분만 보면 매번 헛짚는다."""
    body = " ".join(text.split())
    at = body.find(keyword.strip()[:12])
    if at < 0:
        return body[:width]
    start = max(0, at - width // 3)
    return ("…" if start else "") + body[start : start + width]


def main():
    """유형별로 틀린 문항을 원인과 함께 펼친다."""
    parser = argparse.ArgumentParser(description="틀린 문항을 눈으로 본다.")
    parser.add_argument("--chunks", required=True, help="청크 이름 (__header 없이)")
    parser.add_argument("--type", default="의역", help="질문 유형. all 이면 전부")
    parser.add_argument("--evalset", default="eval_qa")
    parser.add_argument("--embed", default="tei", choices=["tei", "local", "fake"])
    parser.add_argument("--rerank", default="tei", choices=["tei", "local", "fake"])
    parser.add_argument("--pool", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--budget", type=int, default=settings.MAX_CONTEXT_CHARS)
    parser.add_argument("--full", action="store_true", help="맞힌 문항도 보여준다")
    args = parser.parse_args()

    index = f"{args.chunks}__header__{args.embed}"
    if index not in list_stores():
        sys.exit(f"인덱스가 없습니다: {index}\n"
                 f"  python src/vectorstore.py --chunks {args.chunks}__header")

    chunks = chunking.load_chunks(args.chunks)
    print(f"청크 {len(chunks):,}개")

    pairs = [p for p in load_evalset(args.evalset) if p.get("answerable", True)]
    if args.type != "all":
        pairs = [p for p in pairs if p.get("type") == args.type]

    store = load_store(index, load_embedder(args.embed))
    pipeline = Pipeline([
        Dense(store, k=args.pool),
        Rerank(load_reranker(args.rerank), k=args.top_k),
    ])

    reasons = Counter()
    for i, pair in enumerate(pairs, 1):
        found = pipeline(pair["question"]).chunks
        kept = fit_budget(found, args.budget)
        if any(matches(c.page_content, pair["keywords"]) for c in kept):
            reasons["맞음"] += 1
            if not args.full:
                continue
            verdict = "맞음"
        else:
            # 코퍼스 전체에서 정답이 든 청크를 찾는다. 이게 갈림길이다.
            gold = [c for c in chunks if matches(c.page_content, pair["keywords"])]
            if not gold:
                verdict = "정답없음"
            else:
                gold_ids = {c.metadata.get("chunk_id") for c in gold}
                rank = next(
                    (r for r, c in enumerate(found, 1)
                     if c.metadata.get("chunk_id") in gold_ids), None
                )
                verdict = f"순위밀림({rank}위)" if rank else "후보밖"
            reasons[verdict.split("(")[0]] += 1

        print(f"\n[{verdict}] {i}. {pair['question']}")
        print(f"    정답  {pair['keywords']}")
        if verdict.startswith("정답없음"):
            print("    → 코퍼스 어디에도 없다. 질문이나 정답 키워드를 고쳐야 한다.")
        elif verdict.startswith("후보밖") or verdict.startswith("순위밀림"):
            print(f"    정답 청크 {len(gold)}개 중 하나:")
            print(f"      {around(gold[0].page_content, pair['keywords'][0])}")
        print(f"    실제 1위: {flat(found[0].page_content) if found else '없음'}")

    total = sum(reasons.values())
    print("\n" + "=" * 70)
    print(f"{args.type} {total}문항")
    for reason, n in reasons.most_common():
        print(f"  {reason:<10} {n:>3}개  ({n / total:.0%})")
    print("\n정답없음 → 평가 세트를 고친다 / 후보밖 → 임베딩·청킹 / 순위밀림 → 리랭커·예산")


if __name__ == "__main__":
    main()
