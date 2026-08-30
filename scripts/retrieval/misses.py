"""틀린 문항을 눈으로 보게 펼친다.

적중률 0.675 는 "40개 중 13개를 못 찾았다"만 말한다. **왜** 못 찾았는지는
숫자로 안 나온다. 실패는 세 종류이고 대응이 전부 다르다.

    정답없음   정답 키워드가 코퍼스 어디에도 없다 → 질문이나 정답이 틀렸다
    후보밖     청크는 있는데 후보 30개에 못 든다  → 임베더가 못 잡는다
    순위밀림   후보엔 들었는데 예산 밖으로 밀렸다 → 리랭커/예산 문제

    python scripts/retrieval/misses.py --chunks cleaned_documents_v3__recursive_1200_200

`--b` 를 주면 **두 설정을 짝지어 비교한다.** 평균 MRR 이 0.04 벌어졌을 때
그게 실체인지 세 문항 우연인지는 평균으로 안 갈린다. 같은 문항을 두 설정에
똑같이 던져 **어느 문항이 뒤집혔는지** 세야 한다.

    python scripts/retrieval/misses.py --chunks ... --evalset eval_qa_merged \
      --type 의역 --a "Hybrid+Rerank" --b "Hybrid(BM25+Splade)+Rerank"
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import chunking
from config import settings
from evaluation import fit_budget, load_evalset
from evaluation.evalset import matches

sys.path.insert(0, str(Path(__file__).resolve().parent))  # compare_retrieval 재사용
from compare_retrieval import build_setups


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


def hit(search, pair, budget):
    """이 설정이 이 문항을 맞혔나.

    Args:
        search: 질문을 받아 청크 리스트를 돌려주는 함수.
        pair (dict): 평가 문항.
        budget (int): 컨텍스트 예산 글자 수.

    Returns:
        bool: 예산 안에 정답이 들어왔으면 True.
    """
    kept = fit_budget(search(pair["question"]), budget)
    return any(matches(c.page_content, pair["keywords"]) for c in kept)


def compare(a, b, args, pairs, chunks):
    """같은 문항을 두 설정에 던져 **뒤집힌 것만** 센다.

    평균이 0.04 벌어졌다는 말로는 아무것도 못 정한다. 141문항에서 두 설정이
    대부분 같은 답을 주므로, 실제로 갈린 문항은 보통 서너 개다. 그 서너 개를
    직접 읽어야 채택할지 말지가 정해진다.

    Args:
        a: 기준 설정의 검색 함수.
        b: 비교할 설정의 검색 함수.
        args: 명령줄 인자.
        pairs (list[dict]): 평가 문항들.
        chunks (list): 코퍼스 청크 (여기서는 안 쓰지만 호출부와 맞춘다).
    """
    only_a, only_b, both, neither = [], [], 0, 0
    for pair in pairs:
        got_a = hit(a, pair, args.budget)
        got_b = hit(b, pair, args.budget)
        if got_a and got_b:
            both += 1
        elif got_a:
            only_a.append(pair)
        elif got_b:
            only_b.append(pair)
        else:
            neither += 1

    total = len(pairs)
    print(f"\n{'=' * 70}\n{args.type} {total}문항")
    print(f"  둘 다 맞음   {both:>3}")
    print(f"  {args.a} 만   {len(only_a):>3}")
    print(f"  {args.b} 만   {len(only_b):>3}")
    print(f"  둘 다 틀림   {neither:>3}")

    moved = len(only_a) + len(only_b)
    print(f"\n실제로 갈린 문항 {moved}개 / {total}. ", end="")
    if moved < 5:
        print("이 정도면 우연과 구분이 안 됩니다 — 아래를 눈으로 보고 정하세요.")
    else:
        print(f"{args.b} 가 순증 {len(only_b) - len(only_a):+d}.")

    for label, group in ((args.b, only_b), (args.a, only_a)):
        for pair in group:
            print(f"\n[{label} 만 맞음] {pair['question']}")
            print(f"    정답  {pair['keywords']}")


def main():
    """유형별로 틀린 문항을 원인과 함께 펼친다."""
    parser = argparse.ArgumentParser(description="틀린 문항을 눈으로 본다.")
    parser.add_argument("--chunks", required=True, help="청크 이름 (__header 없이)")
    parser.add_argument("--type", default="의역", help="질문 유형. all 이면 전부")
    parser.add_argument("--evalset", default="eval_qa")
    parser.add_argument(
        "--embed", default="tei", choices=["tei", "local", "openai", "fake"]
    )
    parser.add_argument(
        "--rerank", default="tei", choices=["tei", "local", "cohere", "fake"]
    )
    parser.add_argument("--pool", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--budget", type=int, default=settings.MAX_CONTEXT_CHARS)
    parser.add_argument("--full", action="store_true", help="맞힌 문항도 보여준다")
    parser.add_argument("--a", default="Hybrid+Rerank", help="펼쳐 볼 설정")
    parser.add_argument("--b", help="주면 --a 와 짝지어 비교한다")
    parser.add_argument("--splade", action="store_true")
    parser.add_argument("--splade-model", default=None)
    parser.add_argument("--splade-batch", type=int, default=8)
    parser.add_argument("--splade-url", nargs="?", const="tei", default=None)
    parser.add_argument("--rerank-model", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--bm25-weights", default=[0.7])
    args = parser.parse_args()

    # --a/--b 에 Splade 가 들어 있으면 알아서 켠다. --splade 를 따로 요구하면
    # "그런 설정이 없습니다" 만 보고 왜 없는지 알 수가 없다.
    if any("Splade" in (name or "") for name in (args.a, args.b)):
        args.splade = True
    if args.splade_model is None:
        from pieces import SpladeModel

        args.splade_model = SpladeModel.PIXIE.value

    chunks = chunking.load_chunks(args.chunks)
    setups, _ = build_setups(args)
    for name in (args.a, args.b):
        if name and name not in setups:
            sys.exit(
                f"그런 설정이 없습니다: {name}\n"
                f"  있는 것: {', '.join(setups)}"
            )

    pairs = [p for p in load_evalset(args.evalset) if p.get("answerable", True)]
    if args.type != "all":
        pairs = [p for p in pairs if p.get("type") == args.type]

    # 정답이 안 붙은 문항은 뺀다. 넣어두면 무조건 오답이 되어 "정답없음" 을
    # 부풀린다. `compare_retrieval.py` 도 같은 기준으로 거른다.
    blank = [p for p in pairs if not p.get("keywords")]
    pairs = [p for p in pairs if p.get("keywords")]
    if blank:
        print(f"정답이 안 붙은 {len(blank)}문항은 뺍니다 (채점 불가)")

    if args.b:
        compare(setups[args.a], setups[args.b], args, pairs, chunks)
        return

    pipeline = setups[args.a]
    print(f"설정: {args.a}")
    reasons = Counter()
    for i, pair in enumerate(pairs, 1):
        found = pipeline(pair["question"])
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
