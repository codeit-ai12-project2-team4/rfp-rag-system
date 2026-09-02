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
import hashlib
import json
import shutil
import sys
import warnings
from pathlib import Path

# `python src/vectorstore.py` 로 직접 돌릴 때 config 를 찾게 한다.
# import 로 쓸 때는 이미 경로에 있어서 아무 일도 안 한다.
_ROOT = Path(__file__).resolve().parents[1]
for _folder in (_ROOT / "src", _ROOT):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

# langchain-community 가 sunset 경고를 띄운다. FAISS 자체는 멀쩡하고,
# 대안이라던 langchain-faiss(PyPI)는 빈 껍데기라 드롭인 교체가 없다.
# 청크 9천 개면 FAISS 메모리로 충분하다 — 수십만 개가 되면 그때 옮긴다.
warnings.filterwarnings("ignore", message=".*langchain-community.*")

from langchain_community.vectorstores.faiss import FAISS

from config import settings


def index_path(name):
    """인덱스 이름을 폴더 경로로 바꾼다.

    Args:
        name: 인덱스 이름.

    Returns:
        `outputs/vectorstore/{name}` 경로.
    """
    return settings.VECTORSTORE / name


def fingerprint(embedder):
    """이 임베더가 어떤 모델인지 한 줄로.

    Args:
        embedder: langchain `Embeddings` 객체.

    Returns:
        str: 모델 이름. 알 수 없으면 클래스 이름.
    """
    for attr in ("model_id", "model", "model_name"):
        got = getattr(embedder, attr, None)
        if isinstance(got, str) and got:
            return got
    return type(embedder).__name__


def stamp(path, embedder, store, chunks=None):
    """어떤 모델·어떤 청크로 만든 인덱스인지 옆에 적어 둔다.

    청크 **개수**만으로는 부족하다. 전처리 파이프라인이 바뀌어도 개수가 같을 수
    있고, 그러면 조용히 옛 인덱스를 쓰게 된다 (8/29 에 그렇게 하루를 버렸다).
    본문 해시까지 찍어 둔다.

    Args:
        path (Path): 인덱스 폴더.
        embedder: 만들 때 쓴 임베딩 객체.
        store: 만들어진 FAISS 인덱스.
        chunks: 만들 때 쓴 Document 리스트. 주면 본문 지문을 같이 찍는다.
    """
    (path / "meta.json").write_text(
        json.dumps(
            {
                "model": fingerprint(embedder),
                "dim": store.index.d,
                "chunks": store.index.ntotal,
                "signature": chunk_signature(chunks) if chunks else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def chunk_signature(chunks):
    """청크 본문의 md5 앞 12자. 같은 내용이면 같은 값이 나온다."""
    digest = hashlib.md5()
    for chunk in chunks:
        digest.update(chunk.page_content.encode())
    return digest.hexdigest()[:12]


def verify(path, name, embedder, store, chunks=None):
    """불러온 인덱스가 지금 임베더로 만든 게 맞는지 본다.

    안 맞으면 faiss 가 `assert d == self.d` 로 죽는데, 그건 검색 첫 질문에서야
    터지고 어디가 잘못됐는지 한 글자도 안 알려준다. **차원이 같은데 모델만 다른
    경우는 죽지도 않는다** — BGE-m3 와 arctic 은 둘 다 1024 라 조용히 엉뚱한
    결과가 나온다. 실제로 그렇게 하루를 버렸다.

    Args:
        path (Path): 인덱스 폴더.
        name (str): 인덱스 이름 (오류 메시지에 쓴다).
        embedder: 지금 쓰려는 임베딩 객체.
        store: 불러온 FAISS 인덱스.
        chunks: 지금 쓰려는 청크. 주면 본문 지문까지 대조한다. **전처리
            파이프라인이 바뀌면 이것만 잡아낸다** — 이름도 개수도 그대로일 수 있다.

    Returns:
        불러온 인덱스 그대로.

    Raises:
        RuntimeError: 만들 때와 다른 모델이거나 다른 청크일 때.
    """
    now = fingerprint(embedder)
    meta = path / "meta.json"
    rebuild = f"  python src/vectorstore.py --chunks {name.rsplit('__', 1)[0]} --force"

    if meta.exists():
        was = json.loads(meta.read_text())
        if was.get("model") != now:
            raise RuntimeError(
                f"인덱스를 만든 모델과 다릅니다: {name}\n"
                f"  만들 때  {was.get('model')}\n"
                f"  지금     {now}\n"
                f"다시 만드세요:\n{rebuild}"
            )
        if chunks and was.get("signature") and was["signature"] != chunk_signature(chunks):
            raise RuntimeError(
                f"인덱스를 만든 청크와 다릅니다: {name}\n"
                f"  만들 때  {was['signature']}  ({was.get('chunks')}개)\n"
                f"  지금     {chunk_signature(chunks)}  ({len(chunks)}개)\n"
                f"전처리본이 바뀌었습니다. 다시 만드세요:\n{rebuild}"
            )
        return store

    # 도장이 없는 옛 인덱스는 차원만이라도 본다
    dim = len(embedder.embed_query("차원 확인"))
    if store.index.d != dim:
        raise RuntimeError(
            f"인덱스 차원이 안 맞습니다: {name}\n"
            f"  인덱스 {store.index.d}차원 / 지금 임베더({now}) {dim}차원\n"
            f"다시 만드세요:\n{rebuild}"
        )
    return store


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
            return verify(
                path,
                name,
                embedder,
                FAISS.load_local(
                    str(path), embedder, allow_dangerous_deserialization=True
                ),
                chunks,
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
        stamp(path, embedder, store, chunks)
        if verbose:
            print(f"인덱스 저장: {path} ({fingerprint(embedder)}, {store.index.d}차원)")
    return store


def load_store(name, embedder, chunks=None):
    """저장해 둔 인덱스를 불러온다.

    Args:
        name: `build_store` 에 준 것과 같은 이름.
        embedder: 만들 때와 **같은** 임베딩 객체. 다른 모델을 주면 차원이
            안 맞거나, 맞더라도 엉뚱한 결과가 나온다.
        chunks: 지금 쓰려는 청크. 주면 인덱스가 이 청크로 만들어진 게 맞는지
            대조한다. 청크를 이미 읽어 둔 곳(BM25 를 같이 쓰는 곳)은 넘길 것.

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
    return verify(
        path,
        name,
        embedder,
        FAISS.load_local(str(path), embedder, allow_dangerous_deserialization=True),
        chunks,
    )


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
    parser.add_argument(
        "--chunks",
        required=True,
        help="outputs/chunks 의 청크 이름 (python src/chunking.py 가 찍어 준다)",
    )
    parser.add_argument(
        "--embed", default="tei", choices=["tei", "local", "openai", "fake"]
    )
    parser.add_argument("--name", help="인덱스 이름 (생략하면 청크이름__임베딩)")
    parser.add_argument("--force", action="store_true", help="이미 있어도 다시 만든다")
    parser.add_argument(
        "--dry-run", action="store_true", help="만들지 않고 개수와 예상 비용만 본다"
    )
    args = parser.parse_args()

    import chunking  # 명령줄로 쓸 때만 필요하다
    from models import load_embedder

    chunks = chunking.load_chunks(args.chunks)
    cost = estimate_cost(chunks)
    print(
        f"청크 {len(chunks):,}개 · {cost['글자수']:,}자 "
        f"· OpenAI 로 하면 약 ${cost['예상비용(달러)']}"
    )

    if args.dry_run:
        print("--dry-run 이므로 여기서 멈춥니다.")
        return

    name = args.name or f"{args.chunks}__{args.embed}"
    build_store(chunks, load_embedder(args.embed), name=name, force=args.force)

    print(f"\n인덱스 이름: {name}")
    print(f"노트북에서:  store = load_store({name!r}, embedder)")


if __name__ == "__main__":
    main()


def _demo():
    """지문 대조가 실제로 막는지 본다. 임베딩 없이 meta.json 만 다룬다.

        python -c "import sys; sys.path[:0]=['src','.']; import vectorstore; vectorstore._demo()"
    """
    import tempfile
    from types import SimpleNamespace

    from langchain_core.documents import Document

    chunks = [Document(page_content="가나다"), Document(page_content="라마바")]
    other = [Document(page_content="가나다"), Document(page_content="사아자")]  # 개수는 같다
    assert chunk_signature(chunks) != chunk_signature(other)

    embedder = SimpleNamespace(model_id="테스트모델")
    store = SimpleNamespace(index=SimpleNamespace(d=8, ntotal=2))

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)
        stamp(path, embedder, store, chunks)

        verify(path, "테스트", embedder, store, chunks)          # 같은 청크 — 통과
        verify(path, "테스트", embedder, store)                  # 청크 없음 — 옛 동작 그대로

        try:
            verify(path, "테스트", embedder, store, other)
        except RuntimeError as error:
            assert "청크와 다릅니다" in str(error), error
        else:
            raise AssertionError("개수가 같은 다른 청크를 못 잡았다")

    print("통과 — 개수가 같아도 내용이 다르면 잡는다")
