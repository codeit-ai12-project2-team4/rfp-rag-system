"""공고 차수 처리를 확인한다 — 규칙 자체와, 코퍼스의 실제 상태.

    python scripts/retrieval/check_revisions.py

`chunking.drop_stale_revisions` 자체 검사를 먼저 돌리고(코퍼스 없이 도는
어서션 몇 줄), 그 다음 코퍼스에 대고 세 가지를 잰다. 9/9 에 처음 돌렸을 때
답이 이랬고, 그 답이 지금 구현을 정했다.

    ① 차수가 여럿인 공고가 몇 건이고, **본문이 실제로 다른가**
       9/9: 공고 320건 중 10건. 그중 **9건이 유사도 1.000** — 마감만 미루고
       차수를 올린 것이다. 그래서 "최신만 남긴다" 가 아니라 **본문 해시로
       가른다**. 내용이 갈린 1건(R26BK01719775)은 남겨야 "1차와 뭐가
       달라졌나" 를 답할 수 있다.

    ② 평가 세트가 그 공고들을 건드리나
       9/9: 0문항. 그래서 거르기가 지표를 안 움직인다. 걸리면 접기 전후를
       둘 다 재서 실어야 한다.

    ③ `Dense` 후필터에서 한쪽 차수가 통째로 빠지지 않는가
       `Dense.search` 는 전역 상위 `k*10` 을 뽑고 **그 다음에** 거른다.
       그 공고 청크가 그 안에 없으면 **후보에 하나도 안 남는다.**
       9/9: 양쪽 다 남았다. 대신 두 차수가 정확히 반반씩 나왔다(6:6·15:15·11:11)
       — 빠진 게 아니라 **낭비**였고, 그게 거르기의 근거다. 코퍼스가 커지면
       300 천장에 걸릴 수 있으니 이 칸은 계속 본다. 한쪽이 0개로 떨어지기 시작하면 LanceDB
       사전필터로 바꾼다 — `lance_store.delete_docs` 가 `doc_id IN (…)` 절을
       만드는 코드를 이미 갖고 있다.
"""

import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import chunking  # noqa: E402
from config import retrieval as cfg  # noqa: E402
from config import settings  # noqa: E402
from evaluation import evalset as ev  # noqa: E402


def split_id(doc_id):
    """`20240330003-2` → `("20240330003", 2)`. 못 쪼개면 (doc_id, 0)."""
    no, _, order = str(doc_id).rpartition("-")
    if not no or not order.isdigit():
        return str(doc_id), 0
    return no, int(order)


class _Doc:
    """Document 흉내. 이 규칙이 보는 건 본문과 doc_id 둘뿐이다."""

    def __init__(self, text, doc_id):
        self.page_content = text
        self.metadata = {"doc_id": doc_id}


def selftest():
    """`drop_stale_revisions` 규칙을 코퍼스 없이 확인한다. **본문으로만 판단하나.**

    틀리면 조용히 나빠진다. 본문이 다른 차수를 지우면 "1차와 뭐가 달라졌나" 가
    답할 수 없는 질문이 되고, 같은 차수를 안 지우면 pool 절반이 같은 내용으로
    채워진다. 어느 쪽도 화면에 오류로 안 보인다. 그래서 코퍼스를 보기 전에
    이것부터 돌린다.
    """
    def ids(chunks):
        return [c.metadata["doc_id"] for c in chunks]

    drop = chunking.drop_stale_revisions

    assert chunking.split_doc_id("R26BK01719775-1") == ("R26BK01719775", 1)
    # 처음 받은 100건은 `{기관}_{사업명}` 이라 규칙 밖이다. 건드리면 안 된다.
    assert chunking.split_doc_id("행안부_클라우드전환")[1] is None

    # 실측 9건 — 마감만 미루고 차수를 올린 경우
    got = drop([_Doc("가나", "A-0"), _Doc("다라", "A-0"),
                _Doc("가나", "A-1"), _Doc("다라", "A-1")])
    assert ids(got) == ["A-1", "A-1"], ids(got)

    # 실측 나머지 1건 — 본문이 실제로 갈렸다. **비교 재료다.**
    assert ids(drop([_Doc("옛", "B-0"), _Doc("새", "B-1")])) == ["B-0", "B-1"]

    # 차수 셋 — 최신과 같은 것만 뺀다
    got = drop([_Doc("옛", "E-0"), _Doc("새", "E-1"), _Doc("새", "E-2")])
    assert ids(got) == ["E-0", "E-2"], ids(got)

    # 청크 순서가 다르면 다른 문서다 (낱말을 재배열한 개정일 수 있다)
    got = drop([_Doc("가", "F-0"), _Doc("나", "F-0"),
                _Doc("나", "F-1"), _Doc("가", "F-1")])
    assert len(got) == 4, ids(got)

    # 뺄 게 없으면 **받은 객체를 그대로** 돌려준다. `load_chunks` 가 lru_cache 로
    # 같은 리스트를 돌려쓰므로 새 리스트를 만들면 공짜로 복사가 는다.
    one = [_Doc("가", "C-0")]
    assert drop(one) is one
    odd = [_Doc("가", "행안부_사업"), _Doc("가", "행안부_사업")]
    assert drop(odd) is odd

    print("자체 검사 통과\n")


def main():
    selftest()
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
        # **유사도는 사람이 읽는 눈금이다. 거르는 기준이 아니다.**
        # `drop_stale_revisions` 는 해시를 정확히 비교한다. 0.999 여도 한 글자가
        # 다르면 남긴다 — 금액 한 줄, 날짜 한 줄이 바뀐 진짜 개정일 수 있다.
        mark = "  ← 글자까지 같음 (걸러졌어야 한다)" if old == new else ""
        if old == new:
            same += 1
        print(f"   {no}  차수 {orders}  {len(old):,}자 → {len(new):,}자"
              f"  유사도 {ratio:.3f}{mark}")
    print(f"\n   글자까지 같은 것 {same}/{len(dup)}건"
          f"  (0이어야 정상 — `load_chunks` 가 이미 걸렀다)")
    if same:
        print("   → 걸러졌어야 할 게 남아 있다. drop_stale_revisions 를 보라.")
    else:
        print("   → 남은 것은 전부 **본문이 실제로 다른** 차수다. 비교 재료다.")

    # 무엇이 달라졌는지 실제로 본다. **문항을 지어내지 않으려면 이걸 봐야 한다.**
    #
    # **화면에 쏟지 않는다.** 전처리가 표를 평탄화해서 같은 문구가 수십 번
    # 반복되는데, 그대로 찍으면 터미널 스크롤백이 통째로 날아간다(9/9).
    # 파일로 쓰고 여기서는 몇 줄만 미리 보여준다.
    if "--diff" in sys.argv:
        from difflib import unified_diff  # noqa: PLC0415 - 이 갈래에서만 쓴다

        out = settings.EVAL_RESULTS / "revisions_diff.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for no, byorder in sorted(dup.items()):
                orders = sorted(byorder)
                lines = [
                    line for line in unified_diff(
                        byorder[orders[0]].splitlines(),
                        byorder[orders[-1]].splitlines(),
                        lineterm="", n=0,
                    )
                    if line.startswith(("+", "-")) and line[1:].strip()
                    and not line.startswith(("+++", "---"))
                ]
                f.write(f"\n===== {no}  {orders[0]}차 → {orders[-1]}차  "
                        f"({len(lines)}줄) =====\n")
                f.write("\n".join(lines) + "\n")
                print(f"   {no}  {orders[0]}차 → {orders[-1]}차  바뀐 줄 {len(lines)}개")
                for line in lines[:4]:
                    print(f"     {line[:90]}")
                if len(lines) > 4:
                    print(f"     … 나머지는 파일에")
        print(f"   → {out}")

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

    # ③ Dense 후필터에서 한쪽 차수가 통째로 빠지지 않는가
    print("\n③ Dense 후필터 — 차수 둘을 주면 각각 몇 개가 후보에 남나")
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
        starved = " ← 한쪽이 후보에 없다" if any(got.get(i, 0) == 0 for i in ids) else ""
        print(f"   {no}  pool {pool} 요청 → {len(hits)}개  [{line}]{starved}")

    print("\n한쪽이 0개면 LanceDB 사전필터(.where(doc_id IN …, prefilter=True))로 바꾼다.")
    print("delete_docs() 가 그 절을 만드는 코드를 이미 갖고 있다.")


if __name__ == "__main__":
    main()
