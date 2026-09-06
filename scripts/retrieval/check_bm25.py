"""BM25 의 진짜 비용을 쪼개 잰다. `python scripts/retrieval/check_bm25.py --chunks <이름>`

**"BM25 는 증분이 안 된다" 는 절반만 맞다.** 알고리즘상 막힌 게 아니라
`rank_bm25.BM25Okapi` 에 `add()` 가 없을 뿐이고, 145초의 대부분도 IDF 계산이
아니라 형태소 분석일 가능성이 크다. 그러면 답이 완전히 달라진다 —
"엔진을 바꾼다" 가 아니라 "토큰을 캐시한다" 다.

그래서 세 가지를 따로 잰다.

    1. 형태소 분석(Kiwi)      캐시하면 새 청크만 내면 되는 몫
    2. BM25Okapi() 색인 구축   토큰이 있어도 매번 내야 하는 몫
    3. RAM                     무중단 교체는 인덱스를 두 벌 들므로 이게 2배가 된다

`--add` 로 "매일 들어오는 N건" 을 흉내 내서, 캐시가 있을 때의 증분 비용도 잰다.
"""

import argparse
import gc
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import chunking  # noqa: E402


def rss_mb():
    """지금 프로세스가 쓰는 물리 메모리(MB). psutil 이 없으면 0."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e6
    except ImportError:
        return 0.0


def main():
    parser = argparse.ArgumentParser(description="BM25 의 시간·메모리를 단계별로 잰다.")
    parser.add_argument("--chunks", required=True, help="청크 이름")
    parser.add_argument("--add", type=int, default=300,
                        help="증분 흉내: 새로 들어온 청크 수 (기본 300 ≈ 하루치)")
    args = parser.parse_args()

    from pieces import search as engine
    from rank_bm25 import BM25Okapi

    base = rss_mb()
    chunks = chunking.load_chunks(args.chunks)
    texts = [c.page_content for c in chunks]
    after_load = rss_mb()
    print(f"청크 {len(chunks):,}개 · 본문 {sum(len(t) for t in texts) / 1e6:.1f}MB")
    print(f"  청크 적재 후 RSS +{after_load - base:.0f}MB")

    # --- 1. 형태소 분석 -------------------------------------------------
    # **`cached_tokens` 로 잰다.** 예전엔 `korean_tokens_batch` 를 직접 불러서
    # 캐시를 통째로 건너뛰었고, 두 번 돌려도 시간이 똑같이 나왔다.
    cache_path = engine._token_cache_path()
    had_cache = cache_path.exists()
    started = time.time()
    tokens = engine.cached_tokens(texts, verbose=True)
    tokenize_sec = time.time() - started
    after_tok = rss_mb()

    total_tokens = sum(len(t) for t in tokens)
    vocab = len(set(t for doc in tokens for t in doc))
    state = "캐시 있음" if had_cache else "캐시 비어 있음 — 이번에 채운다"
    print(f"\n1. 형태소 분석  {tokenize_sec:6.1f}초   RSS +{after_tok - after_load:.0f}MB"
          f"   ({state})")
    print(f"   토큰 {total_tokens:,}개 · 청크당 평균 {total_tokens / len(tokens):.0f}개"
          f" · 어휘 {vocab:,}개")
    if cache_path.exists():
        print(f"   캐시 파일 {cache_path.stat().st_size / 1e6:.1f}MB")

    # --- 2. 색인 구축 ---------------------------------------------------
    started = time.time()
    index = BM25Okapi(tokens)
    build_sec = time.time() - started
    after_build = rss_mb()
    print(f"2. 색인 구축    {build_sec:6.1f}초   RSS +{after_build - after_tok:.0f}MB")

    share = tokenize_sec / (tokenize_sec + build_sec) * 100
    print(f"\n   → 전체 {tokenize_sec + build_sec:.1f}초 중 형태소 분석이 {share:.0f}%")

    # --- 2b. 캐시가 실제로 얼마나 사는가 --------------------------------
    # 메모리 캐시(같은 프로세스)와 디스크 캐시(재시작 뒤)를 나눠 잰다.
    started = time.time()
    engine.cached_tokens(texts)
    warm_sec = time.time() - started

    engine._TOKEN_CACHE = None  # 프로세스를 새로 띄운 셈 친다
    started = time.time()
    engine.cached_tokens(texts)
    cold_sec = time.time() - started

    print(f"\n2b. 캐시 적중")
    print(f"   같은 프로세스 안       {warm_sec:6.1f}초")
    print(f"   재시작 뒤(디스크에서)  {cold_sec:6.1f}초"
          f"   ← 서버가 뜰 때 내는 실제 비용")
    print(f"   + 색인 구축            {build_sec:6.1f}초")
    # 캐시가 이미 있던 실행에서는 tokenize_sec 자체가 적중 시간이라
    # "캐시 없을 때" 로 쓰면 거짓말이 된다. 그때는 비교를 안 적는다.
    was_cold = "" if had_cache else f"  (캐시 없을 때 {tokenize_sec + build_sec:.1f}초)"
    print(f"   = 기동 {cold_sec + build_sec:.1f}초{was_cold}")

    # --- 3. 증분 흉내 ---------------------------------------------------
    # 캐시가 있다고 치면, 새로 들어온 것만 형태소 분석하고 색인만 다시 짓는다.
    # 캐시에 없는 본문이어야 하므로 뒤에 표시를 붙여 새 청크처럼 만든다.
    new = [t + f"\n[증분측정 {i}]" for i, t in enumerate(texts[: args.add])]
    started = time.time()
    engine.korean_tokens_batch(new)
    add_tok_sec = time.time() - started
    print(f"\n3. 증분 (새 청크 {args.add}건 가정)")
    print(f"   새 청크만 형태소 분석  {add_tok_sec:6.1f}초")
    print(f"   + 캐시에서 나머지      {cold_sec:6.1f}초")
    print(f"   + 색인 전체 재구축     {build_sec:6.1f}초"
          f"   (IDF 가 코퍼스 전역이라 이건 남는다)")
    print(f"   = {add_tok_sec + cold_sec + build_sec:.1f}초{was_cold}")
    print("   ※ 새 청크가 있으면 Kiwi 를 올려야 한다(+2~3초, RAM +약 460MB).")

    # --- 4. 무중단 교체 비용 --------------------------------------------
    # 옛 인덱스가 요청을 받는 동안 새 인덱스를 만들어야 하므로 한동안 두 벌이다.
    index_mb = after_build - after_load
    print(f"\n4. 무중단 교체")
    print(f"   인덱스 한 벌 ≈ {index_mb:.0f}MB (청크 본문 포함)")
    print(f"   교체 순간 두 벌  ≈ {index_mb * 2:.0f}MB 피크")
    try:
        import psutil

        free = psutil.virtual_memory().available / 1e6
        print(f"   지금 여유 {free:.0f}MB → {'여유 있음' if free > index_mb * 2 + 1000 else '빠듯하다'}")
    except ImportError:
        print("   (psutil 이 없어 여유 메모리를 못 봤다)")

    # 코드에 박힌 가정과 대조한다. 안 맞으면 need_memory 가 헛것을 막고 있다.
    assumed = len(chunks) / 10000 * 0.5 * 1000
    print(f"\n   코드의 가정(need_memory): 1만 청크당 500MB → {assumed:.0f}MB")
    print(f"   실측:                                        {index_mb:.0f}MB"
          f"  ({'가정이 과하다' if index_mb < assumed * 0.7 else '가정이 모자란다' if index_mb > assumed else '얼추 맞다'})")

    del index, tokens
    gc.collect()


if __name__ == "__main__":
    main()
