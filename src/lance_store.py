"""FAISS 대신 LanceDB. `vectorstore.py` 와 **같은 자리에 끼운다.**

바꾸는 건 환경변수 하나다. 청크·임베더·리랭커·평가세트는 그대로 쓴다.

    python src/lance_store.py --chunks cleaned_documents_v7__recursive_1500_250
    STORE=lance python scripts/retrieval/compare_retrieval.py ...
    STORE=lance uvicorn src.api:app --port 8010

**왜 옮기나** — FAISS 는 인덱스를 통째로 메모리에 올리고(청크 9,500개에 수백MB),
새 공고가 들어올 때마다 pickle 을 다시 쓴다. LanceDB 는 파일에 바로 붙이고
읽을 때 필요한 만큼만 본다. 나라장터에서 매일 공고가 들어오는 게 전제라
그쪽이 맞다. **다만 지금 규모에서는 둘 다 빠르다.** 그래서 지우지 않고 나란히
두고 재 본다 — 느려지면 되돌리면 된다.

**인덱스(IVF_PQ)를 안 만든다.** 9,500행이면 전수 스캔이 수십 ms 고, 그게
FAISS 의 IndexFlat 과 같은 조건이라 A/B 가 공정하다. PQ 는 압축이라
정확도가 떨어진다 — 수십만 행이 되면 그때 `tbl.create_index()` 를 넣는다.

거리 척도는 L2 다. LanceDB 기본값이고 langchain FAISS 기본값도 L2 라 같다.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

# `python src/lance_store.py` 로 직접 돌릴 때 config 를 찾게 한다.
_ROOT = Path(__file__).resolve().parents[1]
for _folder in (_ROOT / "src", _ROOT):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import lancedb
from langchain_core.documents import Document

from config import settings


def _db():
    """`outputs/lancedb/` 를 연다. 없으면 만든다."""
    settings.LANCEDB.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(settings.LANCEDB))


def _stamp(name):
    """어떤 모델로 만든 테이블인지 적어 두는 파일 경로."""
    return settings.LANCEDB / f"{name}.json"


def _fingerprint(embedder):
    """이 임베더가 어떤 모델인지 한 줄로. vectorstore.fingerprint 와 같은 규칙."""
    for attr in ("model_id", "model", "model_name"):
        got = getattr(embedder, attr, None)
        if isinstance(got, str) and got:
            return got
    return type(embedder).__name__


class LanceStore:
    """`Dense` 부품이 부르는 `similarity_search` 하나만 FAISS 와 맞춘다.

    MMR 은 구현하지 않았다. `retriever.py` 가 안 쓰고, 쓰는 순간
    `Dense(mmr=True)` 가 AttributeError 로 시끄럽게 죽는다 — 조용히 다른
    결과를 내는 것보다 낫다.
    """

    def __init__(self, table, embedder, name=""):
        self.table = table
        self.embedder = embedder
        self.name = name

    def similarity_search(self, query, k=5):
        """질문에 가까운 청크 k개.

        Args:
            query: 질문 문자열.
            k: 가져올 개수.

        Returns:
            Document 리스트. metadata 는 만들 때 넣은 그대로.
        """
        rows = self.table.search(self.embedder.embed_query(query)).limit(k).to_list()
        return [
            Document(page_content=row["text"], metadata=json.loads(row["meta"]))
            for row in rows
        ]

    def __len__(self):
        return self.table.count_rows()

    def __repr__(self):
        return f"LanceStore({self.name!r}, {len(self)}행)"


def build_store(chunks, embedder, name=None, force=False, verbose=True):
    """청크를 임베딩해 LanceDB 테이블을 만든다. `vectorstore.build_store` 와 같은 서명.

    Args:
        chunks: 임베딩할 Document 리스트.
        embedder: langchain `Embeddings` 객체.
        name: 테이블 이름. 없으면 임시 이름으로 만든다(디스크에는 남는다).
        force: True 면 이미 있어도 다시 만든다.
        verbose: 진행 상황을 찍을지.

    Returns:
        LanceStore.
    """
    db = _db()
    name = name or "tmp"
    if name in db.table_names() and not force:
        if verbose:
            print(f"저장된 테이블을 불러옵니다: {settings.LANCEDB / (name + '.lance')}")
        return load_store(name, embedder)

    if verbose:
        total = sum(len(c.page_content) for c in chunks)
        print(f"임베딩 시작 — 청크 {len(chunks):,}개 / 총 {total:,}자")

    vectors = embedder.embed_documents([c.page_content for c in chunks])
    rows = [
        {
            "vector": vector,
            "text": chunk.page_content,
            # 메타데이터는 문서마다 키가 달라서 컬럼으로 펼치면 스키마가 깨진다.
            # 통째로 JSON 문자열 한 칸에 넣는다.
            # ponytail: 이러면 doc_id 로 prefilter 를 못 한다. 필요해지면
            # doc_id 만 별도 컬럼으로 빼고 .where("doc_id IN (...)") 를 쓴다.
            "meta": json.dumps(chunk.metadata, ensure_ascii=False),
        }
        for chunk, vector in zip(chunks, vectors)
    ]
    table = db.create_table(name, data=rows, mode="overwrite")
    _stamp(name).write_text(
        json.dumps(
            {"model": _fingerprint(embedder), "dim": len(rows[0]["vector"]), "chunks": len(rows)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if verbose:
        print(f"테이블 저장: {name} ({_fingerprint(embedder)}, {len(rows[0]['vector'])}차원)")
    return LanceStore(table, embedder, name)


def load_store(name, embedder):
    """저장해 둔 테이블을 연다.

    만든 임베더와 다른 모델을 주면 여기서 막는다. **차원이 같으면 안 죽고
    조용히 엉뚱한 결과가 나오기 때문이다** — BGE-m3 와 arctic 이 둘 다
    1024차원이라 실제로 하루를 버린 적이 있다.

    Args:
        name: `build_store` 에 준 것과 같은 이름.
        embedder: 만들 때와 **같은** 임베딩 객체.

    Returns:
        LanceStore.

    Raises:
        FileNotFoundError: 그 이름의 테이블이 없을 때.
        RuntimeError: 만들 때와 다른 모델일 때.
    """
    db = _db()
    if name not in db.table_names():
        raise FileNotFoundError(
            f"테이블이 없습니다: {name}\n"
            f"저장된 테이블: {sorted(db.table_names()) or '(없음)'}\n"
            f"먼저 만드세요:  python src/lance_store.py --chunks <청크이름>"
        )
    now = _fingerprint(embedder)
    stamp = _stamp(name)
    if stamp.exists():
        was = json.loads(stamp.read_text(encoding="utf-8")).get("model")
        if was != now:
            raise RuntimeError(
                f"테이블을 만든 모델과 다릅니다: {name}\n"
                f"  만들 때  {was}\n"
                f"  지금     {now}\n"
                f"다시 만드세요:\n"
                f"  python src/lance_store.py --chunks {name.rsplit('__', 1)[0]} --force"
            )
    return LanceStore(db.open_table(name), embedder, name)


def list_stores():
    """만들어 둔 테이블 목록."""
    return sorted(_db().table_names()) if settings.LANCEDB.exists() else []


def drop_store(name):
    """테이블과 도장 파일을 지운다."""
    _db().drop_table(name, ignore_missing=True)
    _stamp(name).unlink(missing_ok=True)


def main():
    """명령줄에서 LanceDB 테이블을 만든다. 이름 규칙은 FAISS 쪽과 같다."""
    parser = argparse.ArgumentParser(
        description="청크를 임베딩해 outputs/lancedb 에 저장한다."
    )
    parser.add_argument("--chunks", required=True, help="outputs/chunks 의 청크 이름")
    parser.add_argument("--embed", default="tei", choices=["tei", "local", "openai", "fake"])
    parser.add_argument("--name", help="테이블 이름 (생략하면 청크이름__임베딩)")
    parser.add_argument("--force", action="store_true", help="이미 있어도 다시 만든다")
    args = parser.parse_args()

    import chunking
    from models import load_embedder

    chunks = chunking.load_chunks(args.chunks)
    name = args.name or f"{args.chunks}__{args.embed}"
    store = build_store(chunks, load_embedder(args.embed), name=name, force=args.force)

    print(f"\n테이블 이름: {name}  ({len(store):,}행)")
    print(f"검색에 쓰려면:  STORE=lance uvicorn src.api:app --port 8010")


if __name__ == "__main__":
    main()
