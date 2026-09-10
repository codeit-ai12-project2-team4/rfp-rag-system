"""이번 변경분을 **직접 돌려서** 확인하는 스크립트. 셋 다 따로 돌아간다.

    python scripts/check_gen_store.py --selftest
        GPU·서버·네트워크 없이 LanceDB 배관만 본다. 30초. 여기부터.

    python scripts/check_gen_store.py --gen qwen
        SGLang 컨테이너를 그 모델로 갈아끼우고 실제로 답을 받는다.
        처음 받는 모델이면 몇 분. 모델 키는 config/model_config.py 참고.

    python scripts/check_gen_store.py --compare
        같은 질문을 FAISS 와 LanceDB 에 넣어 겹치는 정도와 걸린 시간을 잰다.
        먼저 python src/lance_store.py --chunks <청크이름> 으로 테이블을 만들 것.
"""

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _folder in (_ROOT / "src", _ROOT):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

QUERIES = [
    "이 사업의 배정예산은 얼마인가",
    "제안서 제출 방식과 마감 일시",
    "참가 자격에 어떤 실적이 필요한가",
    "클라우드 전환 관련 요구사항",
    "평가 배점은 어떻게 되는가",
]


def selftest():
    """서버 없이 LanceDB 배관을 확인한다. 깨지면 여기서 AssertionError 가 난다."""
    from langchain_core.documents import Document

    import lance_store
    from models import FakeEmbeddings

    name = "_selftest"
    chunks = [
        Document(page_content="배정예산은 3억 5천만원이다", metadata={"doc_id": "a", "n": 1}),
        Document(page_content="제안서는 나라장터로 제출한다", metadata={"doc_id": "b", "n": 2}),
        Document(page_content="참가자격은 소프트웨어사업자 신고 업체", metadata={"doc_id": "c", "n": 3}),
    ]
    embedder = FakeEmbeddings()
    lance_store.drop_store(name)
    store = lance_store.build_store(chunks, embedder, name=name, force=True, verbose=False)

    assert len(store) == 3, len(store)
    hits = store.similarity_search("배정예산", k=2)
    assert len(hits) == 2, hits
    # 메타데이터가 JSON 왕복을 견디는지 — 여기가 제일 잘 깨진다
    assert set(hits[0].metadata) == {"doc_id", "n"}, hits[0].metadata
    assert isinstance(hits[0].metadata["n"], int), hits[0].metadata
    assert hits[0].page_content in {c.page_content for c in chunks}

    # 다시 열어도 같은지
    reopened = lance_store.load_store(name, embedder)
    assert len(reopened) == 3

    # 다른 모델로 열면 막아야 한다 (조용히 엉뚱한 결과가 나오는 게 최악이다)
    other = FakeEmbeddings(dim=128)
    other.model_name = "다른모델"
    try:
        lance_store.load_store(name, other)
    except RuntimeError as e:
        assert "만들 때" in str(e), e
    else:
        raise AssertionError("모델이 달라도 안 막혔습니다")

    lance_store.drop_store(name)
    print("selftest 통과 — LanceDB 저장·검색·모델도장 확인")


def check_gen(keys):
    """실제로 SGLang 을 태워 답을 받는다. 모델 교체 시간도 같이 잰다."""
    from config import MODEL_CONFIGS
    from generation import generate_answer
    from models.sglang import current

    context = "\n".join([
        "[1] 본 사업의 총 사업비는 금 350,000,000원(부가세 포함)이다.",
        "[2] 제안서는 나라장터를 통해 2024년 12월 23일 10시까지 제출한다.",
    ])
    print(f"지금 올라와 있는 모델: {current() or '(없음)'}\n")
    for key in keys:
        if key not in MODEL_CONFIGS:
            print(f"{key}: 모르는 키. 가능한 값 {list(MODEL_CONFIGS)}")
            continue
        started = time.time()
        result = generate_answer(key, "이 사업의 배정예산은?", context)
        mark = "OK " if result["ok"] else "실패"
        print(f"{mark} {key:9} {result['model']}")
        print(f"     {time.time() - started:6.1f}초 (교체 포함) / 생성 {result['latency_sec']:.1f}초")
        print(f"     {result['answer'] or result['error']}\n")


def compare():
    """같은 질문을 FAISS 와 LanceDB 에 넣어 결과가 얼마나 겹치는지 본다."""
    import lance_store
    import vectorstore
    from config import retrieval as cfg
    from models import load_embedder

    embedder = load_embedder(cfg.EMBED)
    name = cfg.index_name()
    print(f"인덱스: {name}\n")
    faiss = vectorstore.load_store(name, embedder)
    lance = lance_store.load_store(name, embedder)

    total = 0.0
    for query in QUERIES:
        t0 = time.time()
        a = [d.page_content for d in faiss.similarity_search(query, k=cfg.TOP_K)]
        t1 = time.time()
        b = [d.page_content for d in lance.similarity_search(query, k=cfg.TOP_K)]
        t2 = time.time()
        overlap = len(set(a) & set(b)) / max(len(a), 1)
        same_top = "같음" if a[:1] == b[:1] else "다름"
        total += overlap
        print(
            f"  겹침 {overlap:5.0%}  1위 {same_top}  "
            f"faiss {(t1 - t0) * 1000:5.0f}ms  lance {(t2 - t1) * 1000:5.0f}ms   {query}"
        )
    print(f"\n평균 겹침 {total / len(QUERIES):.0%}")
    print("겹침이 100%가 아니어도 정상이다 — 동점 처리 순서가 다르다.")
    print("50% 아래면 인덱스를 다른 청크·다른 임베더로 만든 것이니 다시 만들 것.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true", help="서버 없이 LanceDB 배관 확인")
    parser.add_argument("--gen", nargs="*", metavar="KEY", help="SGLang 으로 실제 생성 (모델 키)")
    parser.add_argument("--compare", action="store_true", help="FAISS vs LanceDB 검색 비교")
    args = parser.parse_args()

    if args.selftest:
        selftest()
    if args.gen is not None:
        check_gen(args.gen or ["qwen"])
    if args.compare:
        compare()
    if not (args.selftest or args.gen is not None or args.compare):
        parser.print_help()


if __name__ == "__main__":
    main()
