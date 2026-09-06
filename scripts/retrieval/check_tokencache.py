"""토큰 캐시 자체 점검. `python scripts/retrieval/check_tokencache.py`

두 가지만 본다. 이게 깨지면 BM25 가 조용히 틀린 토큰으로 검색한다.

1. 두 번째 호출이 형태소 분석을 **안 한다** (캐시 적중)
2. 토크나이저가 바뀌면 캐시를 **버린다** (옛 토큰을 계속 쓰면 안 된다)
"""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # scripts/retrieval/ 아래다
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]


def main():
    from config import settings

    with tempfile.TemporaryDirectory() as tmp:
        settings.VECTORSTORE = Path(tmp)
        from pieces import search

        search._TOKEN_CACHE = None
        texts = ["제안서 제출 마감일은 언제인가", "가격 평가 배점은 몇 점인가"]

        calls = []
        real = search.korean_tokens_batch

        def counted(items):
            calls.append(len(items))
            return real(items)

        search.korean_tokens_batch = counted
        try:
            first = search.cached_tokens(texts)
            assert calls == [2], f"첫 호출이 2건을 안 잘랐다: {calls}"
            assert all(first), f"토큰이 비었다: {first}"

            # 같은 본문 + 새 본문 하나. 새 것만 잘라야 한다.
            second = search.cached_tokens(texts + ["사업 기간은 12개월이다"])
            assert calls == [2, 1], f"캐시가 안 먹었다: {calls}"
            assert second[:2] == first, "같은 본문인데 토큰이 달라졌다"

            # 프로세스를 새로 띄운 셈 치고 디스크에서 읽는다
            search._TOKEN_CACHE = None
            third = search.cached_tokens(texts)
            assert calls == [2, 1], f"디스크 캐시를 안 읽었다: {calls}"
            assert third == first, "디스크에서 읽은 토큰이 다르다"

            # 토크나이저가 바뀌면 버린다
            search._TOKEN_CACHE = None
            version = search._kiwi_version
            search._kiwi_version = lambda: "다른-토크나이저"
            search.cached_tokens(texts)
            assert calls == [2, 1, 2], f"토크나이저가 바뀌었는데 옛 캐시를 썼다: {calls}"
            search._kiwi_version = version
        finally:
            search.korean_tokens_batch = real
    print("토큰 캐시 OK")


if __name__ == "__main__":
    main()
