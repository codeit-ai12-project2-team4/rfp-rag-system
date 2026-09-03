"""Dense 검색이 왜 안 되는지 20초로 좁힌다.  python scripts/retrieval/check_dense.py

2026-09-03 eval_notices 에서 Dense MRR 0.003 · Top10 0.006 이 나왔다.
공고 195건에서 아무거나 10개 찍어도 Top10 은 0.05 쯤이다. **찍는 것보다 나쁘다** —
성능이 낮은 게 아니라 **매번 같은 엉뚱한 것**을 준다는 뜻이다. 그리고 그건
8/31 에 "질의 셋을 눌러도 같은 목록이 나온다" 던 그 증상과 같은 것이다.

가능한 원인은 네 가지뿐이고, 아래 다섯 항목이 그중 하나를 가리킨다.

    [2] 테이블에 문서가 몇 건뿐          → 색인이 청크 파일 전체로 안 만들어졌다
    [3] 저장된 벡터 ≠ 지금 임베딩한 것    → 다른 모델·다른 접두어로 만든 색인이다
    [4] 질의 벡터끼리 코사인이 1에 가깝다 → 임베더가 질의를 구분 못 한다
    [5] 질의마다 top5 가 같다            → 검색이 벡터를 안 보고 있다
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import chunking
import lance_store
from config import retrieval as cfg
from config import settings
from models import load_embedder

QUERIES = ["클라우드 전환 사업", "장애인 접근성 개선", "이러닝 시스템 운영"]


def cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def main():
    index, chunk_name = cfg.index_name(), cfg.chunk_name()
    print(f"인덱스 {index}\n청크   {chunk_name}\n")

    stamp = settings.LANCEDB / f"{index}.json"
    print("[1] 도장", json.loads(stamp.read_text(encoding="utf-8")) if stamp.exists()
          else "없음")

    embedder = load_embedder("tei")
    store = lance_store.load_store(index, embedder)
    table = store.table
    got = table.search().select(["doc_id", "text"]).limit(None).to_arrow().to_pydict()
    docs = set(got["doc_id"])
    chunks = chunking.load_chunks(chunk_name)
    want = {str(c.metadata.get("doc_id") or "") for c in chunks}
    print(f"\n[2] 테이블 {len(got['doc_id']):,}행 · 문서 {len(docs)}건")
    print(f"    청크파일 {len(chunks):,}행 · 문서 {len(want)}건")
    missing = want - docs
    if missing:
        print(f"    X 색인에 없는 문서 {len(missing)}건: {sorted(missing)[:5]}")
    else:
        print("    O 문서 집합이 같다")

    # 저장된 벡터가 정말 그 본문의 벡터인가. 여기가 어긋나면 나머지는 볼 것도 없다.
    sample = table.search().select(["text", "vector"]).limit(3).to_arrow().to_pylist()
    fresh = embedder.embed_documents([r["text"] for r in sample])
    print("\n[3] 저장된 벡터 vs 지금 임베딩 (1.0 이어야 한다)")
    for row, vec in zip(sample, fresh):
        stored = list(row["vector"])
        print(f"    코사인 {cos(stored, vec):+.3f}   노름 {sum(x*x for x in stored)**0.5:.3f}"
              f"   «{row['text'][:30]}…»")

    qv = [embedder.embed_query(q) for q in QUERIES]
    print("\n[4] 질의 벡터끼리 (낮아야 한다)")
    for i in range(len(qv)):
        for j in range(i + 1, len(qv)):
            print(f"    {cos(qv[i], qv[j]):+.3f}  {QUERIES[i]} ↔ {QUERIES[j]}")

    print("\n[5] 질의별 top5 doc_id (달라야 한다)")
    for q in QUERIES:
        hits = store.similarity_search(q, k=5)
        print(f"    {q:<16} {[h.metadata.get('doc_id') for h in hits]}")


if __name__ == "__main__":
    main()
