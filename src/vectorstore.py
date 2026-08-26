"""FAISS 인덱스 만들기 / 불러오기.

강의에서는 `FAISS.from_documents(chunks, embedding_model)` 한 줄로 끝냈고
저장은 안 했다. 문서가 4개짜리 카페 안내라 매번 다시 만들어도 1초였다.

RFP 100건이면 청크가 수천 개다. 매번 다시 임베딩하면 몇 분씩 걸리고
OpenAI 를 쓰면 돈이 나간다. 그래서 이름을 붙여 저장하고 다음부터 불러온다.

    store = build_store(chunks, embedder, name="section_1000__bgem3ko")
    store = load_store("section_1000__bgem3ko", embedder)

이름 규칙 — 청킹 설정과 임베딩 모델을 둘 다 넣는다. 둘 중 하나만 바뀌어도
다른 인덱스여야 하기 때문이다. 이름이 겹치면 엉뚱한 벡터로 검색하게 된다.
"""

import shutil

from langchain_community.vectorstores import FAISS

from _config import settings


def index_path(name):
    return settings.VECTORSTORE / name


def build_store(chunks, embedder, name=None, force=False, verbose=True):
    """청크를 임베딩해 FAISS 인덱스를 만든다.

    name 을 주면 디스크에 저장하고, 다음에 같은 이름으로 부르면 다시
    안 만들고 불러온다. name 없이 부르면 메모리에만 만든다.
    force=True 면 저장된 게 있어도 다시 만든다.
    """
    if name:
        settings.make_dirs()
        path = index_path(name)
        if path.exists() and not force:
            if verbose:
                print(f"저장된 인덱스를 불러옵니다: {path}")
            return FAISS.load_local(
                str(path), embedder, allow_dangerous_deserialization=True
            )

    if verbose:
        total = sum(len(c.page_content) for c in chunks)
        print(f"임베딩 시작 — 청크 {len(chunks):,}개 / 총 {total:,}자")

    store = FAISS.from_documents(chunks, embedder)

    if name:
        path = index_path(name)
        if path.exists():
            shutil.rmtree(path)
        store.save_local(str(path))
        if verbose:
            print(f"인덱스 저장: {path}")
    return store


def load_store(name, embedder):
    path = index_path(name)
    if not path.exists():
        available = [p.name for p in settings.VECTORSTORE.glob("*") if p.is_dir()]
        raise FileNotFoundError(
            f"인덱스가 없습니다: {path}\n"
            f"저장된 인덱스: {available or '(없음)'}\n"
            "먼저 build_store(chunks, embedder, name=...) 를 부르세요."
        )
    return FAISS.load_local(str(path), embedder, allow_dangerous_deserialization=True)


def list_stores():
    """만들어 둔 인덱스 목록."""
    if not settings.VECTORSTORE.exists():
        return []
    return sorted(p.name for p in settings.VECTORSTORE.glob("*") if p.is_dir())


def add_chunks(store, chunks, name=None):
    """이미 있는 인덱스에 청크를 더한다.

    나라장터에서 매일 새 공고가 들어오는 상황을 위한 것이다. 전체를 다시
    임베딩하지 않고 새 것만 붙인다. 자세한 건 docs/나중에_규모_키우기.md 참고.
    """
    store.add_documents(chunks)
    if name:
        path = index_path(name)
        if path.exists():
            shutil.rmtree(path)
        store.save_local(str(path))
    return store


def estimate_cost(chunks, price_per_1m=0.02):
    """OpenAI 임베딩을 쓸 때 대략 얼마나 나올지.

    한국어는 대략 2.2자당 1토큰. text-embedding-3-small 이 1M 토큰당 $0.02.
    TEI 를 쓰면 0원이므로 이 함수는 OpenAI 쓸 때만 의미가 있다.
    """
    total_chars = sum(len(c.page_content) for c in chunks)
    tokens = total_chars / 2.2
    return {
        "글자수": total_chars,
        "예상토큰": int(tokens),
        "예상비용(달러)": round(tokens / 1_000_000 * price_per_1m, 4),
    }
