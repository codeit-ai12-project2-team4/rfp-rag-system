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
import hashlib
import json
import sys
from pathlib import Path

# `python src/lance_store.py` 로 직접 돌릴 때 config 를 찾게 한다.
_ROOT = Path(__file__).resolve().parents[1]
for _folder in (_ROOT / "src", _ROOT):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import lancedb
from langchain_core.documents import Document

from config import retrieval as cfg
from config import settings


def _db():
    """`outputs/lancedb/` 를 연다. 없으면 만든다."""
    settings.LANCEDB.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(settings.LANCEDB))


def _names(db):
    """테이블 이름 목록. **버전·메서드마다 반환 모양이 다르다.**

    `list_tables()` 는 `context` / `page_token` / `tables` 를 가진 **페이지 응답
    객체**를 주고, 옛 `table_names()` 는 `list[str]` 을 준다. 페이지 객체를
    그대로 `in` 으로 비교하면 **방금 만든 테이블도 없다고 나온다** —
    8,920행을 다 임베딩해 놓고 API 가 그렇게 죽었다.

    `table_names()` 가 deprecated 라 `list_tables()` 를 먼저 쓴다. 옛 버전에는
    `list_tables()` 가 없어서 그때만 되돌아간다.

    Args:
        db: `lancedb.connect()` 결과.

    Returns:
        list[str]: 테이블 이름.
    """
    got = db.list_tables() if hasattr(db, "list_tables") else db.table_names()
    if isinstance(got, dict):
        return list(got.get("tables") or [])
    tables = getattr(got, "tables", None)
    return list(got if tables is None else tables)


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


def _write_stamp(name, model, dim, chunks):
    """테이블 옆에 도장을 찍는다. `prepare.py` 가 이걸 보고 다시 만들지 정한다.

    Args:
        name: 테이블 이름.
        model: 임베딩 모델 이름.
        dim: 벡터 차원.
        chunks: 지금 테이블에 들어 있는 청크 전체.
    """
    from pieces.search import chunk_signature

    _stamp(name).write_text(
        json.dumps(
            {
                "model": model,
                "dim": dim,
                "chunks": len(chunks),
                "signature": chunk_signature(chunks) if chunks else None,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


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
    if name in _names(db) and not force:
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
            # doc_id 는 **컬럼으로 뺀다.** 나머지 메타데이터는 문서마다 키가
            # 달라서 펼치면 스키마가 깨지므로 JSON 한 칸에 통째로 넣는다.
            #
            # doc_id 만 컬럼인 이유는 지우기 위해서다. 마감 지난 공고를
            # 아카이브하면 벡터도 같이 빠져야 하는데, FAISS 는 인덱스를 통째로
            # 메모리에 올려 다시 써야 하고 LanceDB 는 `delete()` 한 줄이다.
            # 그게 이 저장소를 재는 진짜 이유가 됐다.
            "doc_id": str(chunk.metadata.get("doc_id") or ""),
            "meta": json.dumps(chunk.metadata, ensure_ascii=False),
        }
        for chunk, vector in zip(chunks, vectors)
    ]
    table = db.create_table(name, data=rows, mode="overwrite")
    # **지문까지 찍는다.** 이름도 개수도 같은데 내용만 바뀌는 일이 잦다 —
    # 목차 제거는 줄을 지우므로 청크 개수가 안 변한다(실제로 9,189개 그대로였다).
    # FAISS 인덱스가 이걸로 잡으니 여기도 같아야 prepare.py 가 둘을 같게 본다.
    _write_stamp(name, _fingerprint(embedder), len(rows[0]["vector"]), chunks)
    if verbose:
        print(
            f"테이블 저장: {name} ({_fingerprint(embedder)}, {len(rows[0]['vector'])}차원)"
        )
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
    if name not in _names(db):
        raise FileNotFoundError(
            f"테이블이 없습니다: {name}\n"
            f"저장된 테이블: {sorted(_names(db)) or '(없음)'}\n"
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


def add_chunks(name, chunks, embedder, verbose=True):
    """이미 있는 테이블에 청크를 **덧붙인다.** 매일 들어오는 새 공고용.

    전체를 다시 임베딩하지 않는다. 하루 32건이면 청크 3천 개인데, 코퍼스가
    한 달만 쌓여도 전체 재임베딩은 10만 개가 넘는다. 그 차이가 몇 분과
    몇십 분을 가른다.

    **테이블이 없으면 만들지 않고 에러다.** 조용히 새로 만들면 옛 인덱스가
    사라진 걸 아무도 모른다 — 첫 인덱스는 `build_store` 로 명시적으로 만든다.

    Args:
        name: 테이블 이름.
        chunks: 더할 Document 리스트.
        embedder: **만들 때와 같은** 임베딩 객체. 도장으로 확인한다.
        verbose: 진행 상황을 찍을지.

    Returns:
        (더하기 전 행 수, 더한 뒤 행 수).
    """
    store = load_store(name, embedder)  # 여기서 모델 도장을 검사한다
    if not chunks:
        return (len(store), len(store))
    before = len(store)
    vectors = embedder.embed_documents([c.page_content for c in chunks])
    store.table.add([
        {
            "vector": vector,
            "text": chunk.page_content,
            "doc_id": str(chunk.metadata.get("doc_id") or ""),
            "meta": json.dumps(chunk.metadata, ensure_ascii=False),
        }
        for chunk, vector in zip(chunks, vectors)
    ])
    after = store.table.count_rows()
    if verbose:
        print(f"{name}: {before:,} → {after:,}행 ({after - before:,}개 추가)")
    return before, after


def delete_docs(name, doc_ids, embedder=None):
    """공고 몇 건의 청크를 테이블에서 지운다. 아카이브할 때 쓴다.

    FAISS 로는 이게 인덱스를 통째로 다시 쓰는 일이라 사실상 못 한다.

    Args:
        name: 테이블 이름.
        doc_ids: 지울 doc_id 목록.
        embedder: 안 줘도 된다. 지우기만 할 거면 모델 확인이 필요 없다.

    Returns:
        (지우기 전 행 수, 지운 뒤 행 수).
    """
    doc_ids = [str(d) for d in doc_ids if str(d)]
    if not doc_ids:
        return (0, 0)
    table = _db().open_table(name)
    before = table.count_rows()
    quoted = ", ".join("'" + d.replace("'", "''") + "'" for d in doc_ids)
    table.delete(f"doc_id IN ({quoted})")
    after = table.count_rows()
    print(f"{name}: {before:,} → {after:,}행 ({before - after:,}개 지움)")
    return before, after


def _digest(texts):
    """본문 여러 개를 순서와 상관없이 한 지문으로. 정렬하고 해싱한다.

    정렬하는 이유는 테이블에서 읽은 행 순서를 믿을 수 없기 때문이다.
    지우고 다시 넣으면 순서가 바뀌는데, 그때마다 "바뀐 문서" 로 잡히면
    증분이 아니라 전체 재임베딩이 된다.
    """
    digest = hashlib.md5()
    for text in sorted(texts):
        digest.update(text.encode())
    return digest.hexdigest()[:12]


def _doc_hashes(table):
    """테이블에 든 문서별 본문 지문. **벡터 컬럼은 안 읽는다.**

    벡터가 용량의 99% 다. 1,024차원 float 이 청크마다 4KB 라 10만 청크면
    400MB 를 읽게 된다. 벡터만 빼면 십수 MB 다.

    **본문뿐 아니라 메타데이터도 지문에 넣는다.** 본문만 보면, CSV 를 고쳐
    제목이 비로소 채워진 문서가 "안 바뀐 문서" 로 잡혀 그냥 지나간다.
    화면에는 제목이 계속 안 나오고, 다시 돌려도 매번 0건 변경이다.

    `to_lance()` 를 안 쓰는 이유는 그게 pylance 를 따로 깔아야 하기 때문이다.
    `limit(None)` 이 없으면 기본 10행만 온다 — 그러면 나머지가 전부
    "빠진 공고" 로 잡혀 통째로 지워진다.
    """
    got = (
        table.search()
        .select(["doc_id", "text", "meta"])
        .limit(None)
        .to_arrow()
        .to_pydict()
    )
    by = {}
    for doc_id, text, meta in zip(got["doc_id"], got["text"], got["meta"]):
        by.setdefault(doc_id, []).append((text or "") + "\x00" + (meta or ""))
    return {doc_id: _digest(parts) for doc_id, parts in by.items()}


def sync_docs(name, chunks, embedder, verbose=True):
    """테이블을 청크 파일과 맞춘다. **새 공고·바뀐 공고만 임베딩한다.**

    크론이 하루 여러 번 크롤링하면 코퍼스는 계속 자라는데 새로 들어오는 건
    그중 일부다. 매번 전체를 다시 임베딩하면 코퍼스에 비례해 느려지고,
    그 시간 동안 API 는 옛 인덱스를 물고 있다. 들어온 것만 넣으면 시간이
    **하루치에 비례**한다.

    바뀐 공고(차수가 올라 본문이 갈린 경우)는 doc_id 가 같으므로 지우고
    다시 넣는다. 지우지 않고 더하면 같은 공고의 옛 청크와 새 청크가 함께
    검색된다.

    Args:
        name: 테이블 이름.
        chunks: 지금 청크 파일 전체의 Document 리스트.
        embedder: **만들 때와 같은** 임베딩 객체. 도장으로 확인한다.
        verbose: 진행 상황을 찍을지.

    Returns:
        (더한 문서 수, 지운 문서 수, 그대로 둔 문서 수).
    """
    store = load_store(name, embedder)  # 여기서 모델 도장을 검사한다
    old = _doc_hashes(store.table)

    parts = {}
    for chunk in chunks:
        parts.setdefault(str(chunk.metadata.get("doc_id") or ""), []).append(
            chunk.page_content
            + "\x00"
            + json.dumps(chunk.metadata, ensure_ascii=False)
        )
    new = {doc_id: _digest(p) for doc_id, p in parts.items()}

    gone = [d for d in old if d not in new]
    changed = [d for d in new if d in old and old[d] != new[d]]
    added = [d for d in new if d not in old]
    kept = len(new) - len(changed) - len(added)

    if verbose:
        print(f"{name}: 새 {len(added)}건 · 바뀜 {len(changed)}건 · "
              f"빠짐 {len(gone)}건 · 그대로 {kept}건")

    if gone or changed:
        delete_docs(name, gone + changed)
    todo = set(added) | set(changed)
    if todo:
        add_chunks(
            name,
            [c for c in chunks if str(c.metadata.get("doc_id") or "") in todo],
            embedder,
            verbose=verbose,
        )

    # 도장은 항상 다시 찍는다. 더하고 지운 게 없어도 청크 지문은 바뀔 수 있다.
    was = json.loads(_stamp(name).read_text(encoding="utf-8"))
    _write_stamp(name, was["model"], was["dim"], chunks)
    return len(added), len(gone), kept


def list_stores():
    """만들어 둔 테이블 목록."""
    return sorted(_names(_db())) if settings.LANCEDB.exists() else []


def drop_store(name):
    """테이블과 도장 파일을 지운다."""
    _db().drop_table(name, ignore_missing=True)
    _stamp(name).unlink(missing_ok=True)


def main():
    """명령줄에서 LanceDB 테이블을 만든다. 이름 규칙은 FAISS 쪽과 같다."""
    parser = argparse.ArgumentParser(
        description="청크를 임베딩해 outputs/lancedb 에 저장한다."
    )
    # 생략하면 config/retrieval.py 가 쓰는 것과 **같은** 이름을 쓴다.
    # 손으로 적다 틀리면 API 가 "테이블이 없습니다" 로 죽는다 (실제로 겪었다).
    parser.add_argument(
        "--chunks",
        default=cfg.chunk_name(),
        help="outputs/chunks 의 청크 이름 (생략하면 config 기본값)",
    )
    parser.add_argument(
        "--embed", default="tei", choices=["tei", "local", "openai", "fake"]
    )
    parser.add_argument("--name", help="테이블 이름 (생략하면 청크이름__임베딩)")
    parser.add_argument("--force", action="store_true", help="이미 있어도 다시 만든다")
    parser.add_argument(
        "--sync",
        action="store_true",
        help="있는 테이블에 새/바뀐 공고만 반영 (전체 재임베딩 안 함)",
    )
    args = parser.parse_args()

    import chunking
    from models import load_embedder

    chunks = chunking.load_chunks(args.chunks)
    name = args.name or f"{args.chunks}__{args.embed}"
    embedder = load_embedder(args.embed)
    if args.sync and name in _names(_db()):
        sync_docs(name, chunks, embedder)
        store = load_store(name, embedder)
    else:
        store = build_store(chunks, embedder, name=name, force=args.force)

    print(f"\n테이블 이름: {name}  ({len(store):,}행)")
    if name != cfg.index_name():
        print(
            f"주의: config 가 찾는 이름은 {cfg.index_name()} 입니다. API 가 이걸 못 찾습니다."
        )
    print("검색에 쓰려면:  STORE=lance uvicorn src.api:app --port 8010")


if __name__ == "__main__":
    main()
