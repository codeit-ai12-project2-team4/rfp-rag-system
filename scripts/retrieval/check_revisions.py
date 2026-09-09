"""공고 차수가 여럿인 건을 다루기 전에, **다룰 값이 있는지** 먼저 본다.

    python scripts/retrieval/check_revisions.py

세 가지를 잰다. 하나라도 답이 "없다" 면 계획을 접거나 바꿔야 한다.

    ① 차수가 여럿인 공고가 몇 건이고, **본문이 실제로 다른가**
       차수만 오르고 첨부가 같으면 비교할 게 없다. 그러면 목록에서
       접어 버리는 게 맞고, 비교 기능은 지을 이유가 없다.

    ② 평가 세트가 그 공고들을 건드리나
       건드리면 목록 접기가 지표를 움직인다. 앞의 표들과 비교가 깨진다.

    ③ `Dense` 후필터가 실제로 굶는가
       `Dense.search` 는 전역 상위 `k*10` 을 뽑고 **그 다음에** 거른다.
       그 공고 청크가 그 안에 없으면 0개가 남는다. 차수 둘을 주면 같은
       자리를 나눠 쓴다. 굶으면 LanceDB 사전필터로 바꿔야 한다.
"""

import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import chunking  # noqa: E402
from config import retrieval as cfg  # noqa: E402
from evaluation import evalset as ev  # noqa: E402


def split_id(doc_id):
    """`20240330003-2` → `("20240330003", 2)`. 못 쪼개면 (doc_id, 0)."""
    no, _, order = str(doc_id).rpartition("-")
    if not no or not order.isdigit():
        return str(doc_id), 0
    return no, int(order)


def main():
    chunks = chunking.load_chunks(cfg.CHUNKS)
    print(f"청크 {len(chunks):,}개 · {cfg.CHUNKS}\n")

    # doc_id → 본문 전체(순서대로 이어 붙인 것)
    body = defaultdict(list)
    for c in chunks:
        body[str(c.metadata.get("doc_id") or "")].append(c.page_content)

    families = defaultdict(dict)
    for doc_id, parts in body.items():
        no, order = split_id(doc_id)
        families[no][order] = "".join(parts)

    dup = {no: v for no, v in families.items() if len(v) > 1}
    print(f"① 공고 {len(families)}건 중 차수가 여럿인 것 {len(dup)}건")
    if not dup:
        print("   → 다룰 값이 없다. 여기서 멈춘다.")
        return

    same = 0
    for no, byorder in sorted(dup.items()):
        orders = sorted(byorder)
        old, new = byorder[orders[0]], byorder[orders[-1]]
        # 글자 단위 유사도. 1.0 이면 본문이 똑같다는 뜻이다.
        ratio = SequenceMatcher(None, old, new, autojunk=False).quick_ratio()
        mark = "  같음" if ratio > 0.99 else ""
        if ratio > 0.99:
            same += 1
        print(f"   {no}  차수 {orders}  {len(old):,}자 → {len(new):,}자"
              f"  유사도 {ratio:.3f}{mark}")
    print(f"   본문이 사실상 같은 것 {same}/{len(dup)}건")
    if same == len(dup):
        print("   → 전부 같다. **비교 기능을 지을 이유가 없다.** 접기만 한다.")

    # ② 평가 세트가 이 공고들을 건드리나
    print(f"\n② 평가 세트 {cfg.EVALSET}")
    try:
        pairs = ev.load_evalset(cfg.EVALSET)
    except Exception as error:  # noqa: BLE001 - 세트가 없어도 나머지는 봐야 한다
        print(f"   못 읽었다: {error}")
    else:
        hit = [p for p in pairs if split_id(p.get("doc_id", ""))[0] in dup]
        print(f"   {len(pairs)}문항 중 이 공고들을 가리키는 문항 {len(hit)}개")
        for p in hit[:10]:
            print(f"     {p.get('doc_id')}  {str(p.get('question'))[:50]}")
        if hit:
            print("   → 목록 접기가 지표를 움직인다. 앞의 표와 비교가 깨진다.")

    # ③ Dense 후필터가 굶는가
    print("\n③ Dense 후필터 — 차수 둘을 주면 각각 몇 개가 살아남나")
    from pieces.search import Dense  # noqa: PLC0415 - 여기서만 쓴다
    import retriever  # noqa: PLC0415

    store = retriever.open_index(retriever.INDEX, "tei", cfg.CHUNKS)
    if store is None:
        print("   인덱스를 못 열었다 — 건너뛴다")
        return
    pool = cfg.POOL
    for no, byorder in list(sorted(dup.items()))[:5]:
        ids = [f"{no}-{o}" for o in sorted(byorder)]
        # 그 공고 제목을 질문 대신 쓴다. 실제 질문에 가장 가까운 대리물이다.
        title = next(
            (c.metadata.get("title") for c in chunks
             if str(c.metadata.get("doc_id") or "") == ids[-1]), no)
        hits = Dense(store, k=pool, doc_ids=ids).search(str(title), pool)
        got = defaultdict(int)
        for h in hits:
            got[str(h.metadata.get("doc_id"))] += 1
        line = " · ".join(f"{i}:{got.get(i, 0)}" for i in ids)
        starved = " ← 굶음" if any(got.get(i, 0) == 0 for i in ids) else ""
        print(f"   {no}  pool {pool} 요청 → {len(hits)}개  [{line}]{starved}")

    print("\n굶는 게 있으면 LanceDB 사전필터(.where(doc_id IN …, prefilter=True))로 바꾼다.")
    print("delete_docs() 가 그 절을 만드는 코드를 이미 갖고 있다.")


if __name__ == "__main__":
    main()
