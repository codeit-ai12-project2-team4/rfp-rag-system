"""증분 경로 자체 점검.  python scripts/retrieval/check_incremental.py

두 가지를 확인한다. 여기가 깨지면 크론이 매일 코퍼스 전체를 다시 처리한다.

1. `sync_docs` 가 새 공고만 임베딩하고, 바뀐 공고는 지우고 다시 넣고,
   빠진 공고는 지우는가. fake 임베더로 재므로 TEI 가 없어도 돈다.
2. `_cached_rows` 가 전처리본을 되살릴 때 CSV 컬럼을 떼어 내는가.
   안 떼면 병합이 한 번 더 붙어 `사업명_원본` 같은 컬럼이 생긴다.
"""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]


def doc(text, doc_id):
    from langchain_core.documents import Document

    return Document(page_content=text, metadata={"doc_id": doc_id, "사업명": "무엇"})


def test_cached_rows():
    from preprocessing.rfp.build import _cached_rows

    with tempfile.NamedTemporaryFile(
        "w", suffix=".jsonl", encoding="utf-8", delete=False
    ) as f:
        f.write(
            json.dumps(
                {
                    "page_content": "본문",
                    "page_content_for_generation": "표 있는 본문",
                    "metadata": {
                        "source": "가.hwp",
                        "extractor": "hwp5-table",
                        "사업기간": "12개월",
                        "사업명": "CSV 에서 온 것",
                        "공고번호": "123",
                        "메타매칭방식": "exact",
                    },
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        path = Path(f.name)

    row = _cached_rows(path)["가.hwp"]
    assert row["clean_text"] == "본문", row
    assert row["clean_text_for_generation"] == "표 있는 본문", row
    assert row["extractor"] == "hwp5-table", row
    assert row["사업기간"] == "12개월", row  # 본문에서 뽑은 건 남는다
    for gone in ("사업명", "공고번호", "메타매칭방식", "source"):
        assert gone not in row, f"{gone} 가 남았다 — 병합이 한 번 더 붙는다"
    path.unlink()
    print("_cached_rows OK")


def test_sync_docs():
    import lance_store
    from models import load_embedder

    with tempfile.TemporaryDirectory() as tmp:
        lance_store.settings.LANCEDB = Path(tmp)
        embedder = load_embedder("fake")

        first = [doc("가나다 첫째", "A"), doc("라마바 둘째", "A"), doc("사아자", "B")]
        lance_store.build_store(first, embedder, name="t", verbose=False)

        # B 는 그대로, A 는 본문이 바뀌고, C 가 새로 들어온다
        second = [doc("완전히 다른 본문", "A"), doc("사아자", "B"), doc("새 공고", "C")]
        got = lance_store.sync_docs("t", second, embedder, verbose=False)
        assert got == (1, 0, 1), got  # 새 1(C) · 빠짐 0 · 그대로 1(B), A 는 바뀜

        table = lance_store._db().open_table("t")
        assert table.count_rows() == 3, table.count_rows()
        assert set(lance_store._doc_hashes(table)) == {"A", "B", "C"}

        # 같은 걸 또 넣으면 아무것도 안 바뀐다.
        # **여기가 이 파일의 핵심이다** — 깨지면 매일 전체 재임베딩이 된다.
        got = lance_store.sync_docs("t", second, embedder, verbose=False)
        assert got == (0, 0, 3), got

        # 빠진 공고는 지운다
        left = [d for d in second if d.metadata["doc_id"] != "B"]
        got = lance_store.sync_docs("t", left, embedder, verbose=False)
        assert got == (0, 1, 2), got
        assert lance_store._db().open_table("t").count_rows() == 2

        stamp = json.loads((Path(tmp) / "t.json").read_text(encoding="utf-8"))
        assert stamp["chunks"] == 2, stamp
        print("sync_docs OK")


if __name__ == "__main__":
    test_cached_rows()
    test_sync_docs()
    print("\n전부 통과")
