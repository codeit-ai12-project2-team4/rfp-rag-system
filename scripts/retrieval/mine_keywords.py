"""`AddKeywords` 사전의 **값(붙이는 낱말)** 을 코퍼스에서 뽑는다. 문항을 안 쓴다.

    python scripts/retrieval/mine_keywords.py --chunks $CHUNKS
    python scripts/retrieval/mine_keywords.py --chunks $CHUNKS --top 6 --write

9/10 에 용어추가를 되살렸지만 사전은 여전히 짐작으로 쓴 것이다. 남은 결함은
`check_keywords.py` 가 잰 이것 하나다 — **붙인 낱말의 84%가 정답 근거에 없다.**

고칠 때 부딪히는 것: 사전은 `구어 → 공문어` 짝인데 **구어는 문서에 없다.**
"얼마" 라고 쓴 RFP 는 없다. 그래서 코퍼스에서 뽑을 수 있는 건 값뿐이다.

    키(구어)   ← 문서에서 못 뽑는다. 사람이 쓰거나, 나중에 실사용 질의 로그에서
    값(공문어) ← **여기를 뽑는다.** 측정된 결함이 정확히 이쪽이다

문항을 한 개도 안 쓰는 게 요점이다. 평가 세트에서 뽑으면 그 세트로 잰 성적은
못 믿는다. 코퍼스만 쓰면 **177문항이 전부 시험지로 남는다.** 뽑을 문항과 잴
문항을 나눌 필요 자체가 없어진다.

이 스크립트는 사전을 고치지 않는다. 후보를 뽑아 보여주고 `--write` 로
json 을 떨어뜨린다. 채택은 A/B 를 보고 사람이 한다.

    python scripts/retrieval/compare_retrieval.py --chunks $CHUNKS --scoped \
        --keywords outputs/reports/keywords_mined.json
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from config import settings  # noqa: E402
from pieces.expand import AddKeywords  # noqa: E402
from pieces.search import cached_tokens, korean_tokens  # noqa: E402


def idf(df, total):
    """BM25Okapi 가 쓰는 IDF. 이 값이 0에 가까우면 붙여도 순위가 안 움직인다."""
    return math.log((total - df + 0.5) / (df + 0.5) + 1)


def diagnose(synonyms, doc_freq, total):
    """지금 사전의 낱말이 BM25 에서 실제로 힘을 갖는지 본다.

    함정 하나를 먼저 드러낸다. 사전에는 `과업기간` 이라고 적혀 있지만 질의도
    청크와 **같은 형태소 분석기를 지난다.** Kiwi 가 `과업`+`기간` 으로 쪼개면
    실제로 붙는 토큰은 그 둘이다. `check_keywords.py` 는 글자 그대로 대조했으니
    이 층을 못 봤다.
    """
    print("=" * 78)
    print("① 지금 사전 — 붙는 낱말이 BM25 에서 힘이 있나")
    print("=" * 78)
    print(f"{'키':<8} {'적은 말':<10} {'실제 토큰':<12} {'df':>6} {'df%':>6} {'IDF':>6}  판정")
    print("-" * 78)

    dead, risky, ok = 0, 0, 0
    for key, words in synonyms.items():
        for word in words.split():
            for token in korean_tokens(word) or [word]:
                df = doc_freq.get(token, 0)
                share = df / total * 100
                score = idf(df, total)
                if df == 0:
                    verdict, dead = "코퍼스에 없음", dead + 1
                elif share > 30:
                    verdict, dead = "죽음 — 너무 흔해 IDF≈0", dead + 1
                elif df < 5:
                    verdict, risky = "위험 — 걸리면 세게 흔든다", risky + 1
                else:
                    verdict, ok = "쓸모", ok + 1
                mark = "" if token == word else f"({word})"
                print(
                    f"{key:<8} {mark:<10} {token:<12} {df:>6} {share:>5.1f}% "
                    f"{score:>6.2f}  {verdict}"
                )
    print("-" * 78)
    print(f"쓸모 {ok} · 죽음 {dead} · 위험 {risky}\n")


def mine(chunk_tokens, synonyms, doc_freq, total, top, min_docs, lo, hi):
    """키마다 그 개념이 실제로 쓰인 청크를 모아, 거기서만 튀는 낱말을 뽑는다.

    점수는 c-TF-IDF 꼴이다 — **이 슬라이스 안 빈도 ÷ 코퍼스 전체 빈도.**
    슬라이스에서만 자주 나오는 낱말일수록 크다. 이름만 어렵지, "이 주제
    문단에서만 유독 자주 보이는 말" 을 세는 것이다.

    Args:
        chunk_tokens: 청크별 토큰 리스트.
        synonyms: 씨앗 사전. 슬라이스를 고르는 데만 쓴다.
        doc_freq: 코퍼스 전체 df.
        total: 청크 수.
        top: 키마다 남길 후보 수.
        min_docs: 슬라이스 안에서 이만큼은 나와야 후보로 본다.
        lo, hi: 전체 df 비율의 위아래 문. 밖은 버린다.

    Returns:
        {키: "낱말 낱말 …"} 새 사전.
    """
    print("=" * 78)
    print("② 코퍼스에서 뽑기 — 문항 0개")
    print("=" * 78)

    mined = {}
    for key, words in synonyms.items():
        seeds = set()
        for word in words.split():
            seeds.update(korean_tokens(word) or [word])

        slice_ids = [i for i, toks in enumerate(chunk_tokens) if seeds & set(toks)]
        if len(slice_ids) < min_docs:
            print(f"\n[{key}] 씨앗이 걸린 청크 {len(slice_ids)}개 — 너무 적어 건너뜀")
            mined[key] = words
            continue

        inside = Counter()
        for i in slice_ids:
            inside.update(set(chunk_tokens[i]))

        scored = []
        for token, count in inside.items():
            if count < min_docs:
                continue
            share = doc_freq[token] / total
            if not (lo <= share <= hi):
                continue  # 너무 흔하면 IDF≈0, 너무 드물면 잘못 걸릴 때 크게 흔든다
            lift = (count / len(slice_ids)) / share
            scored.append((lift, count, token))
        scored.sort(reverse=True)

        picked = [t for _, _, t in scored[:top]]
        mined[key] = " ".join(picked) if picked else words

        print(f"\n[{key}]  씨앗 {' '.join(sorted(seeds))}  · 슬라이스 {len(slice_ids)}청크")
        print(f"  전:  {words}")
        print(f"  후:  {mined[key]}")
        for lift, count, token in scored[:top]:
            print(
                f"       {token:<12} 슬라이스 {count:>5}/{len(slice_ids)} "
                f"· 전체 {doc_freq[token] / total * 100:>5.1f}% · 배수 {lift:>5.2f}"
            )
    return mined


def main():
    parser = argparse.ArgumentParser(description="AddKeywords 사전의 값을 코퍼스에서 뽑는다.")
    parser.add_argument("--chunks", required=True, help="outputs/chunks 의 청크 이름")
    parser.add_argument("--top", type=int, default=5, help="키마다 남길 낱말 수")
    parser.add_argument("--min-docs", type=int, default=20, help="후보의 최소 등장 청크 수")
    parser.add_argument("--lo", type=float, default=0.002, help="전체 df 비율 아래 문")
    parser.add_argument("--hi", type=float, default=0.25, help="전체 df 비율 위 문")
    parser.add_argument("--write", action="store_true", help="json 으로 떨어뜨린다")
    parser.add_argument(
        "--out", default=str(settings.OUTPUTS / "reports" / "keywords_mined.json")
    )
    args = parser.parse_args()

    import chunking  # 청크를 읽을 때만 필요하다

    chunks = chunking.load_chunks(args.chunks)
    texts = [c.page_content for c in chunks]
    print(f"청크 {len(texts)}개 · 형태소 분석 (캐시가 있으면 몇 초)\n")
    chunk_tokens = cached_tokens(texts, verbose=True)

    total = len(chunk_tokens)
    doc_freq = Counter()
    for toks in chunk_tokens:
        doc_freq.update(set(toks))

    synonyms = AddKeywords.SYNONYMS
    diagnose(synonyms, doc_freq, total)
    mined = mine(
        chunk_tokens, synonyms, doc_freq, total,
        args.top, args.min_docs, args.lo, args.hi,
    )

    print("\n" + "=" * 78)
    print("③ 남는 한계 — 이걸로 안 고쳐지는 것")
    print("=" * 78)
    print("발동률은 그대로다. 키(구어)가 질문에 있어야 붙으므로 44% 그대로다.")
    print("키를 늘리려면 실사용 질의 로그가 필요하다. 평가 세트에서 뽑으면 샌다.")

    if args.write:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(mined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\n→ {out}")
        print("  A/B:  python scripts/retrieval/compare_retrieval.py "
              f"--chunks {args.chunks} --scoped --keywords {out}")


def demo():
    """IDF 문턱과 채굴 점수가 뜻대로 도는지만 본다. 인덱스도 코퍼스도 안 쓴다."""
    assert idf(1, 10000) > idf(5000, 10000), "드문 낱말의 IDF 가 더 커야 한다"
    assert idf(9000, 10000) < 0.2, "9할에 나오는 낱말은 IDF 가 0 근처여야 한다"

    tokens = [["예산", "배정"], ["예산", "금액"], ["시스템"], ["시스템"], ["시스템", "예산"]]
    freq = Counter()
    for t in tokens:
        freq.update(set(t))
    mined = mine(
        tokens, {"얼마": "예산"}, freq, len(tokens),
        top=3, min_docs=1, lo=0.0, hi=1.0,
    )
    # `배정`·`금액` 은 예산 슬라이스 안에만 있고, `시스템` 은 주로 밖에 있다
    picked = mined["얼마"].split()
    assert "배정" in picked and "금액" in picked, picked
    assert "시스템" not in picked, picked
    print("\ndemo ok")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        main()
