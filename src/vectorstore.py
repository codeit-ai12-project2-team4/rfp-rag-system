"""청크를 임베딩해 FAISS 인덱스를 만들고 불러온다.

강의에서는 `FAISS.from_documents(chunks, embedding_model)` 한 줄로 끝냈고
저장은 안 했다. 문서가 4개짜리 카페 안내라 매번 다시 만들어도 1초였다.

RFP 100건이면 청크가 수천 개다. 매번 다시 임베딩하면 몇 분씩 걸리고
OpenAI 를 쓰면 돈이 나간다. 그래서 이름을 붙여 저장하고 다음부터 불러온다.

이름 규칙 — 청킹 설정과 임베딩 모델을 둘 다 넣는다. 둘 중 하나만 바뀌어도
다른 인덱스여야 하기 때문이다. 이름이 겹치면 엉뚱한 벡터로 검색하게 된다.

명령줄로 돌리면 `outputs/vectorstore/{이름}/` 에 `index.faiss` 와
`index.pkl` 이 떨어진다.

    python src/vectorstore.py --chunks documents__section_1000_150
    python src/vectorstore.py --chunks ... --embed fake     서버 없이 배관만 확인
"""

import argparse
import shutil
import sys
from pathlib import Path

# `python src/vectorstore.py` 로 직접 돌릴 때 config 를 찾게 한다.
# import 로 쓸 때는 이미 경로에 있어서 아무 일도 안 한다.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from langchain_community.vectorstores import FAISS  # noqa: E402

from config import settings  # noqa: E402


def index_path(name):
    """인덱스 이름을 폴더 경로로 바꾼다.

    Args:
        name: 인덱스 이름.

    Returns:
        `outputs/vectorstore/{name}` 경로.
    """
    return settings.VECTORSTORE / name


def build_store(chunks, embedder, name=None, force=False, verbose=True):
    """청크를 임베딩해 FAISS 인덱스를 만든다.

    Args:
        chunks: 임베딩할 Document 리스트.
        embedder: langchain `Embeddings` 객체.
        name: 저장 이름. 주면 디스크에 저장하고, 다음에 같은 이름으로 부르면
            다시 안 만들고 불러온다. 없으면 메모리에만 만든다.
        force: True 면 저장된 게 있어도 다시 만든다.
        verbose: 진행 상황을 찍을지.

    Returns:
        FAISS 인덱스.
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
    """저장해 둔 인덱스를 불러온다.

    Args:
        name: `build_store` 에 준 것과 같은 이름.
        embedder: 만들 때와 **같은** 임베딩 객체. 다른 모델을 주면 차원이
            안 맞거나, 맞더라도 엉뚱한 결과가 나온다.

    Returns:
        FAISS 인덱스.

    Raises:
        FileNotFoundError: 그 이름으로 저장된 인덱스가 없을 때.
    """
    path = index_path(name)
    if not path.exists():
        available = [p.name for p in settings.VECTORSTORE.glob("*") if p.is_dir()]
        raise FileNotFoundError(
            f"인덱스가 없습니다: {path}\n"
            f"저장된 인덱스: {available or '(없음)'}\n"
            "먼저 만드세요:  python src/vectorstore.py --chunks <청크이름>"
        )
    return FAISS.load_local(str(path), embedder, allow_dangerous_deserialization=True)


def list_stores():
    """만들어 둔 인덱스 목록.

    Returns:
        인덱스 이름 리스트. 하나도 없으면 빈 리스트.
    """
    if not settings.VECTORSTORE.exists():
        return []
    return sorted(p.name for p in settings.VECTORSTORE.glob("*") if p.is_dir())


def add_chunks(store, chunks, name=None):
    """이미 있는 인덱스에 청크를 더한다.

    나라장터에서 매일 새 공고가 들어오는 상황을 위한 것이다. 전체를 다시
    임베딩하지 않고 새 것만 붙인다.

    Args:
        store: 이미 만들어 둔 FAISS 인덱스.
        chunks: 더할 Document 리스트.
        name: 주면 그 이름으로 다시 저장한다.

    Returns:
        청크가 더해진 FAISS 인덱스 (같은 객체).
    """
    store.add_documents(chunks)
    if name:
        path = index_path(name)
        if path.exists():
            shutil.rmtree(path)
        store.save_local(str(path))
    return store


def estimate_cost(chunks, price_per_1m=0.02):
    """OpenAI 임베딩을 쓸 때 대략 얼마나 나올지 어림한다.

    한국어는 대략 2.2자당 1토큰. TEI 를 쓰면 0원이므로 이 함수는 OpenAI 를
    쓸 때만 의미가 있다.

    Args:
        chunks: 임베딩할 Document 리스트.
        price_per_1m: 100만 토큰당 달러. text-embedding-3-small 이 0.02.

    Returns:
        글자수·예상토큰·예상비용(달러) 를 담은 dict.
    """
    total_chars = sum(len(c.page_content) for c in chunks)
    tokens = total_chars / 2.2
    return {
        "글자수": total_chars,
        "예상토큰": int(tokens),
        "예상비용(달러)": round(tokens / 1_000_000 * price_per_1m, 4),
    }


# --- 명령줄로 돌리기 -------------------------------------------------------


def main():
    """명령줄에서 FAISS 인덱스를 만든다.

    청크 파일 이름에 임베딩 종류를 붙여 인덱스 이름을 만든다. 그래서
    `documents__section_1000_150__tei` 처럼 **전처리본·자르기·임베딩이
    이름 하나에 다 남는다.**
    """
    parser = argparse.ArgumentParser(
        description="청크를 임베딩해 outputs/vectorstore 에 저장한다."
    )
    parser.add_argument("--chunks", required=True,
                        help="outputs/chunks 의 청크 이름 (python src/chunking.py 가 찍어 준다)")
    parser.add_argument("--embed", default="tei", choices=["tei", "local", "fake"])
    parser.add_argument("--name", help="인덱스 이름 (생략하면 청크이름__임베딩)")
    parser.add_argument("--force", action="store_true", help="이미 있어도 다시 만든다")
    parser.add_argument("--dry-run", action="store_true",
                        help="만들지 않고 개수와 예상 비용만 본다")
    args = parser.parse_args()

    import chunking  # 명령줄로 쓸 때만 필요하다
    from models import load_embedder

    chunks = chunking.load_chunks(args.chunks)
    cost = estimate_cost(chunks)
    print(f"청크 {len(chunks):,}개 · {cost['글자수']:,}자 "
          f"· OpenAI 로 하면 약 ${cost['예상비용(달러)']}")

    if args.dry_run:
        print("--dry-run 이므로 여기서 멈춥니다.")
        return

    name = args.name or f"{args.chunks}__{args.embed}"
    build_store(chunks, load_embedder(args.embed), name=name, force=args.force)

    print(f"\n인덱스 이름: {name}")
    print(f"노트북에서:  store = load_store({name!r}, embedder)")


if __name__ == "__main__":
    main()
