"""`AddKeywords` 사전의 **값(붙이는 낱말)** 을 코퍼스에서 뽑는다. 문항을 안 쓴다.

    python scripts/retrieval/mine_keywords.py            # 진단 + 가지치기
    python scripts/retrieval/mine_keywords.py --write    # 가지친 사전을 json 으로
    python scripts/retrieval/mine_keywords.py --mine     # 채굴도 본다(참고용)

9/10 에 용어추가를 되살렸지만 사전은 여전히 짐작으로 쓴 것이다. 남은 결함은
`check_keywords.py` 가 잰 이것 하나다 — **붙인 낱말의 84%가 정답 근거에 없다.**

고칠 때 부딪히는 것: 사전은 `구어 → 공문어` 짝인데 **구어는 문서에 없다.**
"얼마" 라고 쓴 RFP 는 없다. 그래서 코퍼스에서 뽑을 수 있는 건 값뿐이다.

    키(구어)   ← 문서에서 못 뽑는다. 사람이 쓰거나, 나중에 실사용 질의 로그에서
    값(공문어) ← **여기를 뽑는다.** 측정된 결함이 정확히 이쪽이다

문항을 한 개도 안 쓰는 게 요점이다. 평가 세트에서 뽑으면 그 세트로 잰 성적은
못 믿는다. 코퍼스만 쓰면 **177문항이 전부 시험지로 남는다.** 뽑을 문항과 잴
문항을 나눌 필요 자체가 없어진다.

**채굴(②)은 해 봤고 안 됐다.** 1500자 청크는 주제 슬라이스로 못 쓴다 —
`배정예산` 이 든 청크의 나머지 95%는 딴 얘기라, "슬라이스에만 있는 낱말" 이
`상호출자제한기업집단`·`공매도` 같은 법령 상용구로 채워진다. `--mine` 으로
남겨는 뒀다. 근거는 배수 열이다. 값이 여러 후보에서 똑같으면(`*` 표시)
그건 `N ÷ 슬라이스` 로 포화한 것이고 순위가 아예 없다는 뜻이다.

**그래서 실제로 쓰는 건 가지치기(③)다.** 새 낱말을 지어내는 대신 ①에서
죽은 것으로 나온 낱말을 뺀다. `사업`(59%)·`사항`(64%) 같은 것들인데, 정답을
끌어오지도 못하면서 그 낱말이 든 엉뚱한 청크는 끌어온다. 잰 결함(84%)의
정체가 대부분 이거다.

이 스크립트는 사전을 고치지 않는다. `--write` 로 json 만 떨어뜨리고, 채택은
A/B 를 보고 사람이 한다. `--write` 가 마지막에 A/B 명령을 청크 이름까지
채워서 찍어 준다.
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from config import retrieval as cfg  # noqa: E402
from config import settings  # noqa: E402
from pieces.expand import AddKeywords  # noqa: E402
from pieces.search import cached_tokens, korean_tokens  # noqa: E402


def idf(df, total):
    """BM25Okapi 가 쓰는 IDF. 이 값이 0에 가까우면 붙여도 순위가 안 움직인다."""
    return math.log((total - df + 0.5) / (df + 0.5) + 1)


def diagnose(synonyms, doc_freq, total, dead_share):
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
                elif share > dead_share * 100:
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


def prune(synonyms, doc_freq, total, dead_share):
    """①번 표에서 **죽은 낱말만 뺀다.** 새 낱말을 지어내지 않는다.

    `check_keywords.py` 가 잰 결함은 "붙인 낱말의 84%가 정답 근거에 없다" 였다.
    그 84%의 상당 부분이 `사업`(59%)·`사항`(64%)·`계약`(42%) 처럼 **코퍼스의 절반에
    있는 낱말**이다. BM25 에서 IDF≈0 이라 정답을 끌어오지도 못하면서, 그 낱말이
    든 엉뚱한 청크는 끌어온다. 순이득의 분모만 키우는 것이다.

    사전에는 `배정예산` 이라고 적혀 있지만 질의도 청크와 같은 Kiwi 를 지나므로
    실제로 붙는 건 `배정`·`예산` 이다. **그러니 사전도 형태소로 적는다.** 뜻은
    같고, 무엇이 붙는지가 눈에 보인다.

    Returns:
        {키: "토큰 토큰 …"}. 남는 토큰이 없는 키는 아예 뺀다 — 죽은 키를
        남겨 둬도 붙는 게 없으니 사전만 길어진다.
    """
    print("=" * 78)
    print(f"③ 가지치기 — df {dead_share * 100:.0f}% 넘는 낱말을 뺀다")
    print("=" * 78)

    pruned = {}
    for key, words in synonyms.items():
        keep, drop = [], []
        for word in words.split():
            for token in korean_tokens(word) or [word]:
                df = doc_freq.get(token, 0)
                if df and df / total <= dead_share:
                    if token not in keep:
                        keep.append(token)
                elif token not in drop:
                    drop.append(token)
        print(f"\n[{key}]")
        print(f"  전:  {words}")
        print(f"  후:  {' '.join(keep) if keep else '(키를 통째로 뺀다)'}")
        if drop:
            print(f"  뺌:  {' '.join(drop)}")
        if keep:
            pruned[key] = " ".join(keep)
    return pruned


def mine(
    chunk_tokens, texts, synonyms, doc_freq, total,
    top, min_docs, lo, hi, seed_max, slice_max,
):
    """키마다 그 개념이 실제로 쓰인 청크를 모아, 거기서만 튀는 낱말을 뽑는다.

    점수는 c-TF-IDF 꼴이다 — **슬라이스 안 빈도 ÷ 코퍼스 전체 빈도.** 이 주제
    문단에서만 유독 자주 보이는 말일수록 크다.

    **1차 시도가 망가졌던 자리를 두 군데 막아 뒀다(9/10).**

    ① 씨앗에 문턱이 없었다. `기간` 의 씨앗이 형태소로 쪼개져 `사업`(59%)·
       `계약`(42%)·`기간`(30%)이 되니 슬라이스가 코퍼스의 74% 였다. 그건 주제가
       아니라 그냥 코퍼스다. **죽은 낱말은 후보에서만 걸렀지 앵커에서는 안
       걸렀다.** 그래서 `--seed-max` 를 넘는 씨앗은 슬라이스를 고르는 데 안 쓴다.
    ② 씨앗을 형태소로 쪼개 맞췄다. `사업기간` 이 `사업` 또는 `기간` 으로 걸리니
       안 넓어질 수가 없다. **씨앗은 글자 그대로 찾는다** — `사업기간` 이라고 쓴
       줄만 걸린다. 후보 쪽은 BM25 가 형태소를 보므로 그대로 형태소로 센다.

    슬라이스가 넓으면 슬라이스 밖에 안 나오는 토큰의 배수가 전부 `N ÷ 슬라이스`
    라는 **같은 상수**가 된다. 그러면 순위가 동점 포화라 아무 뜻이 없다. 그 상태를
    `--slice-max` 로 막고, 걸린 후보에는 `*` 를 붙여 눈에 보이게 둔다.

    Args:
        chunk_tokens: 청크별 형태소 토큰.
        texts: 청크 원문. 씨앗을 글자 그대로 찾는 데 쓴다.
        synonyms: 씨앗 사전.
        doc_freq: 코퍼스 전체 df(청크 단위).
        total: 청크 수.
        top: 키마다 남길 후보 수.
        min_docs: 슬라이스 안에서 이만큼은 나와야 후보로 본다.
        lo, hi: 후보의 전체 df 비율 위아래 문.
        seed_max: 이 비율을 넘는 씨앗은 앵커로 안 쓴다.
        slice_max: 슬라이스가 코퍼스의 이 비율을 넘으면 그 키는 못 뽑는다.

    Returns:
        {키: "낱말 낱말 …"}. 못 뽑은 키는 원래 값을 그대로 둔다.
    """
    print("=" * 78)
    print("② 코퍼스에서 뽑기 — 문항 0개")
    print("=" * 78)

    mined = {}
    for key, words in synonyms.items():
        # 씨앗은 글자 그대로. 너무 흔한 것은 앵커에서 뺀다.
        seeds, dropped = [], []
        for word in words.split():
            df = sum(1 for t in texts if word in t)
            (dropped if df / total > seed_max else seeds).append((word, df))

        if not seeds:
            print(f"\n[{key}] 씨앗이 전부 너무 흔하다 — 못 뽑음. 사전을 좁혀야 한다")
            print("       " + " · ".join(f"{w} {d / total * 100:.0f}%" for w, d in dropped))
            mined[key] = words
            continue

        keep = {w for w, _ in seeds}
        slice_ids = [i for i, t in enumerate(texts) if any(w in t for w in keep)]
        share = len(slice_ids) / total

        head = f"\n[{key}]  씨앗 {' '.join(sorted(keep))}"
        if dropped:
            head += f"  (뺀 씨앗 {' '.join(w for w, _ in dropped)})"
        print(f"{head}  · 슬라이스 {len(slice_ids)}청크 ({share * 100:.0f}%)")

        if len(slice_ids) < min_docs:
            print("       너무 적어 건너뜀")
            mined[key] = words
            continue
        if share > slice_max:
            print(f"       슬라이스가 코퍼스의 {share * 100:.0f}% — 주제가 아니다. 못 뽑음")
            mined[key] = words
            continue

        inside = Counter()
        for i in slice_ids:
            inside.update(set(chunk_tokens[i]))

        scored = []
        for token, count in inside.items():
            if count < min_docs:
                continue
            token_share = doc_freq[token] / total
            if not (lo <= token_share <= hi):
                continue  # 흔하면 IDF≈0, 드물면 잘못 걸릴 때 크게 흔든다
            lift = (count / len(slice_ids)) / token_share
            scored.append((lift, count, token))
        scored.sort(reverse=True)

        picked = [t for _, _, t in scored[:top]]
        mined[key] = " ".join(picked) if picked else words

        print(f"  전:  {words}")
        print(f"  후:  {mined[key]}")
        for lift, count, token in scored[:top]:
            # 슬라이스 밖에 하나도 없으면 배수가 상수로 포화한다. 표시해 둔다.
            flag = "*" if count == doc_freq[token] else " "
            print(
                f"       {token:<12}{flag} 슬라이스 {count:>5}/{len(slice_ids)} "
                f"· 전체 {doc_freq[token] / total * 100:>5.1f}% · 배수 {lift:>5.2f}"
            )
    return mined


def main():
    parser = argparse.ArgumentParser(description="AddKeywords 사전의 값을 코퍼스에서 뽑는다.")
    # 안 주면 서버가 쓰는 것과 **같은 청크**를 쓴다. `.env` 의 CHUNKS 다.
    # 이름을 손으로 적으면 서버는 v8 을 보는데 실험은 v7 을 보는 일이 생긴다.
    parser.add_argument(
        "--chunks", default=cfg.CHUNKS, help=f"청크 이름. 생략하면 config/retrieval.py 의 CHUNKS ({cfg.CHUNKS})"
    )
    parser.add_argument("--top", type=int, default=5, help="키마다 남길 낱말 수")
    parser.add_argument("--min-docs", type=int, default=20, help="후보의 최소 등장 청크 수")
    parser.add_argument("--lo", type=float, default=0.002, help="후보 df 비율 아래 문")
    parser.add_argument("--hi", type=float, default=0.25, help="후보 df 비율 위 문")
    parser.add_argument(
        "--seed-max", type=float, default=0.25, help="이보다 흔한 씨앗은 앵커로 안 쓴다"
    )
    parser.add_argument(
        "--slice-max", type=float, default=0.35, help="슬라이스가 이보다 크면 못 뽑는다"
    )
    parser.add_argument(
        "--dead-share", type=float, default=0.30, help="이보다 흔한 낱말은 죽은 것으로 본다"
    )
    parser.add_argument("--mine", action="store_true", help="②번 채굴도 돌린다(참고용)")
    parser.add_argument("--write", action="store_true", help="json 으로 떨어뜨린다")
    parser.add_argument(
        "--out", default=str(settings.OUTPUTS / "reports" / "keywords_mined.json")
    )
    args = parser.parse_args()

    import chunking  # 청크를 읽을 때만 필요하다

    chunks = chunking.load_chunks(args.chunks)
    texts = [c.page_content for c in chunks]
    print(f"{args.chunks} · 청크 {len(texts)}개 · 형태소 분석 (캐시가 있으면 몇 초)\n")
    chunk_tokens = cached_tokens(texts, verbose=True)

    total = len(chunk_tokens)
    doc_freq = Counter()
    for toks in chunk_tokens:
        doc_freq.update(set(toks))

    synonyms = AddKeywords.SYNONYMS
    diagnose(synonyms, doc_freq, total, args.dead_share)
    if args.mine:
        mine(
            chunk_tokens, texts, synonyms, doc_freq, total,
            args.top, args.min_docs, args.lo, args.hi,
            args.seed_max, args.slice_max,
        )

    pruned = prune(synonyms, doc_freq, total, args.dead_share)

    print("\n" + "=" * 78)
    print("남는 한계 — 이걸로 안 고쳐지는 것")
    print("=" * 78)
    print("발동률은 그대로다. 키(구어)가 질문에 있어야 붙으므로 44% 그대로다.")
    print("키를 늘리려면 실사용 질의 로그가 필요하다. 평가 세트에서 뽑으면 샌다.")
    print("새 낱말도 안 는다. 1500자 청크로는 주제 슬라이스를 못 만든다(②).")

    if args.write:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(pruned, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\n→ {out}")
        print("  A/B:  python scripts/retrieval/compare_retrieval.py "
              f"--chunks {args.chunks} --scoped --keywords {out}")
        # 청크 이름을 그대로 찍어 준다. compare_retrieval 은 --chunks 가 필수라
        # 여기서 복사해 붙이면 두 스크립트가 같은 코퍼스를 본다.


def demo():
    """IDF 문턱과 채굴 점수가 뜻대로 도는지만 본다. 인덱스도 코퍼스도 안 쓴다."""
    assert idf(1, 10000) > idf(5000, 10000), "드문 낱말의 IDF 가 더 커야 한다"
    assert idf(9000, 10000) < 0.2, "9할에 나오는 낱말은 IDF 가 0 근처여야 한다"

    tokens = [["예산", "배정"], ["예산", "금액"], ["시스템"], ["시스템"], ["시스템", "예산"]]
    texts = ["".join(t) for t in tokens]
    freq = Counter()
    for t in tokens:
        freq.update(set(t))
    mined = mine(
        tokens, texts, {"얼마": "예산"}, freq, len(tokens),
        top=3, min_docs=1, lo=0.0, hi=1.0, seed_max=1.0, slice_max=1.0,
    )
    # `배정`·`금액` 은 예산 슬라이스 안에만 있고, `시스템` 은 주로 밖에 있다
    picked = mined["얼마"].split()
    assert "배정" in picked and "금액" in picked, picked
    assert "시스템" not in picked, picked

    # 너무 흔한 씨앗은 앵커에서 빠지고, 그 키는 못 뽑는다고 말해야 한다
    kept = mine(
        tokens, texts, {"얼마": "예산"}, freq, len(tokens),
        top=3, min_docs=1, lo=0.0, hi=1.0, seed_max=0.1, slice_max=1.0,
    )
    assert kept["얼마"] == "예산", "씨앗이 전부 빠지면 원래 값을 그대로 둔다"

    # 가지치기: 흔한 낱말은 빠지고, 남는 게 없으면 키가 통째로 빠진다
    got = prune({"얼마": "예산 배정", "흔함": "시스템"}, freq, len(tokens), dead_share=0.5)
    assert got == {"얼마": "배정"}, got
    print("\ndemo ok")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        main()
