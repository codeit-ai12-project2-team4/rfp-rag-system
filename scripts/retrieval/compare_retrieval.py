#!/usr/bin/env python
"""검색 방법을 비교한다. BM25 · Dense · Hybrid · Rerank 를 한 표로.

    python scripts/retrieval/compare_retrieval.py --chunks cleaned_documents_v3__recursive_1200_200
    python scripts/retrieval/compare_retrieval.py --chunks ... --splade
    python scripts/retrieval/compare_retrieval.py --chunks ... --embed fake   서버 없이 배관 확인

청킹은 고정하고 **검색 방법만** 바꾼다.

## 무엇을 비교하나

    BM25                키워드. 형태소를 갈라 단어가 겹치는 걸 찾는다
    Dense               임베딩. 뜻이 비슷한 걸 찾는다
    Splade              어휘 확장. 문서에 안 적힌 낱말까지 가중치를 붙인다
    Hybrid              BM25 + Dense 를 RRF(순위 융합)로 합친다
    Hybrid(BM25+Splade) Dense 를 안 쓴다. 인덱스도 서버도 없이 1.4초
    …+Rerank            합친 뒤 크로스 인코더로 등수를 다시 매긴다
    용어추가+…            질의에 공문 용어를 덧붙인다 (사전만, LLM 없이)

## 무엇을 보나

**리랭커는 적중률이 아니라 등수를 올리는 부품이다.** 적중률이 그대로여도 MRR 이
오르면 일한 것이다. 반대로 MRR 만 보면 재현율이 떨어지는 걸 놓친다. 둘 다 본다.

**리랭커가 후보를 다 지우는 구간이 있다.** 쉬운 평가 세트에서는 상류를 뭘 바꿔도
최종 성적이 바이트 단위로 같았다. pool 30 안에 정답이 늘 있으면 리랭커가
처음부터 다시 줄을 세우기 때문이다. 상류 실험이 의미를 가지려면 평가 세트가
충분히 어려워야 한다.

## 여기서 뺀 것들 (결론이 났다)

    머리말 계열     내 평가 세트가 질문의 100%를 사업명으로 시작해서 생긴 착시였다
    Widen          뒤 청크 붙이기. 세 유형 모두에서 졌다
    Dense+Splade   BM25+Splade 에 전부 지고 3배 느리다
    BM25 0.3       0.5 와 구분이 안 됐다. 0.7 만 남긴다

전 과정은 `docs/시행착오.md`.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))  # 평평한 import: chunking, preprocessing …
sys.path.insert(0, str(ROOT))  # config.settings

import pandas as pd

import chunking
import evaluation as ev
from config import settings
from models import load_embedder, load_reranker
from pieces import (
    BM25,
    AddKeywords,
    Dense,
    Hybrid,
    Pipeline,
    Rerank,
    Splade,
    SpladeModel,
)
from preprocessing import load_documents
from vectorstore import list_stores, load_store


def build_setups(args):
    """비교할 검색기들을 미리 만든다.

    **검색기는 반드시 여기서 한 번만 만든다.** lambda 안에서 만들면 질문마다
    인덱스를 새로 만들게 되어 수십 배 느려진다.

    Args:
        args: 명령줄 인자.

    Returns:
        `({설정 이름: 검색 함수}, [검색기 객체])`. 두 번째는 `--scoped` 에서
        공고 범위를 걸 때 쓴다.
    """
    embedder = load_embedder(args.embed)
    searchers = []
    chunks = chunking.load_chunks(args.chunks)
    print(f"청크 {len(chunks):,}개 ({args.chunks})")

    # BM25 를 짓기 전에 본다. 뒤에서 터지면 몇 분을 버린다.
    plain_index = f"{args.chunks}__{args.embed}"
    if plain_index not in list_stores():
        sys.exit(
            f"인덱스가 없습니다: {plain_index}\n"
            f"  python src/vectorstore.py --chunks {args.chunks}"
        )

    setups = {}

    t = time.time()
    bm25 = BM25(chunks, k=args.pool)
    searchers.append(bm25)
    print(f"  BM25 인덱스 준비 {time.time() - t:.0f}초")
    setups["BM25"] = lambda q: bm25.search(q, args.pool)

    dense = Dense(load_store(plain_index, embedder), k=args.pool)
    searchers.append(dense)
    setups["Dense"] = lambda q: dense.search(q, args.pool)

    hybrid = Hybrid([dense, bm25], k=args.pool, pool=args.pool)
    setups["Hybrid"] = lambda q: hybrid.search(q, args.pool)

    # Splade 는 코퍼스를 마스크 언어모델로 통째로 인코딩한다. 캐시가 없으면
    # 맥이 몇 분간 뜨겁다. 그래서 기본은 꺼두고 --splade 로만 켠다.
    splade = None
    if args.splade:
        splade = Splade(
            chunks,
            model=args.splade_model,
            k=args.pool,
            batch_size=args.splade_batch,
            url=args.splade_url,
            cache=args.chunks,
            verbose=True,
        )
        searchers.append(splade)
        setups["Splade"] = lambda q: splade.search(q, args.pool)

    # Splade 를 섞는 두 가지. 방법 A 는 어휘 매칭 + 어휘 확장, 방법 B 는
    # 뜻(Dense) + 어휘 확장이다.
    bm25_splade = None
    if splade is not None:
        bm25_splade = Hybrid([bm25, splade], weights=[0.4, 0.6], k=args.pool, pool=args.pool)
        setups["Hybrid(BM25+Splade)"] = lambda q: bm25_splade.search(q, args.pool)

    if not args.no_rerank:
        # Rerank 는 부품이라 Pipeline 에 끼워서 쓴다. 직접 구현하지 않는다.
        kwargs = {}
        if args.rerank == "local" and args.rerank_model:
            kwargs = {
                "model": args.rerank_model,
                "trust_remote_code": args.trust_remote_code,
            }
        reranker = load_reranker(args.rerank, **kwargs)
        rr = Pipeline([hybrid, Rerank(reranker, k=args.pool)])
        setups["Hybrid+Rerank"] = lambda q: rr(q).chunks
        # BM25:Dense 비중. RRF 는 순위 융합이라 가중치가 결과를 바꾼다.
        # 지금까지 50:50 고정이었고 한 번도 안 쟀다. 재인덱싱이 필요 없다.
        for weight in args.bm25_weights:
            if abs(weight - 0.5) < 1e-6:
                continue  # 기본값은 위 Hybrid+Rerank 와 같다
            mixed = Hybrid(
                [dense, bm25], weights=[1 - weight, weight], k=args.pool, pool=args.pool
            )
            pipe = Pipeline([mixed, Rerank(reranker, k=args.pool)])
            setups[f"Hybrid(BM25 {weight:.1f})+Rerank"] = (
                lambda p: lambda q: p(q).chunks
            )(pipe)

        # 질의에 공문 용어를 덧붙인다. LLM 없이 사전만 쓴다.
        # 의역 유형("돈이 얼마나 드나" → "사업예산")을 겨냥한 것이다.
        keyword_pipe = Pipeline([AddKeywords(), hybrid, Rerank(reranker, k=args.pool)])
        setups["용어추가+Hybrid+Rerank"] = lambda q: keyword_pipe(q).chunks

        # Splade 조합을 리랭커에 태운다. 후보를 만드는 쪽이 바뀌면 리랭커가
        # 볼 30개가 바뀌므로, 리랭커 없이 잰 순위는 그대로 가지 않는다.
        if bm25_splade is not None:
            pipe = Pipeline([bm25_splade, Rerank(reranker, k=args.pool)])
            setups["Hybrid(BM25+Splade)+Rerank"] = lambda q: pipe(q).chunks


    return setups, searchers


def scope_to_doc(search, searchers, where):
    """질문이 가리키는 공고 안에서만 찾게 만든다.

    실제 제품에서 요약 카드를 만들 때의 흐름이다 — 공고를 먼저 고르고 그 안에서
    항목을 찾는다. 머리말의 이득이 "검색을 그 공고에 몰아주는" 효과였다면,
    공고를 직접 고정하는 순간 그 이득은 사라진다.

    Args:
        search: 원래 검색 함수.
        searchers: 범위를 걸 검색기 객체들 (Dense·BM25). Hybrid 는 이들을 쓴다.
        where: `{질문: [doc_id]}`.

    Returns:
        공고를 좁혀 검색하는 함수.
    """

    def run(question):
        for searcher in searchers:
            searcher.doc_ids = where.get(question)
        return search(question)

    return run


def check_answers_exist(pairs, chunks):
    """정답이 청크에 실제로 있는지 먼저 센다.

    **지표를 재기 전에 이걸 봐야 한다.** 청킹이나 정제를 건드리면 정답이 통째로
    사라질 수 있는데, 그러면 모든 설정이 같이 떨어져서 "검색이 나빠졌다"로
    잘못 읽힌다. 실제로 목차 청크를 통째로 버렸다가 의역 정답 17개가 같이
    사라져 적중률이 0.550 → 0.175 로 무너졌고, 원인을 찾는 데 두 번 헛짚었다.

    Args:
        pairs: 평가 질문 리스트.
        chunks: 검색 대상 청크.

    Returns:
        `{유형: (살아있음, 전체)}`.
    """
    from evaluation import matches

    by_doc = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.metadata.get("doc_id"), []).append(chunk.page_content)

    out = {}
    for pair in pairs:
        kind = pair.get("type", "전체")
        alive, total = out.get(kind, (0, 0))
        body = "".join(by_doc.get(pair["doc_id"], []))
        out[kind] = (alive + bool(matches(body, pair["keywords"])), total + 1)

    broken = {k: v for k, v in out.items() if v[0] < v[1]}
    if broken:
        print("  ⚠ 정답이 청크에 없는 질문이 있다 — 검색 성능이 아니라 데이터 문제다")
        for kind, (alive, total) in broken.items():
            print(f"      {kind} {alive}/{total}")
    else:
        print(
            "  정답은 전부 청크 안에 있다 "
            + " · ".join(f"{k} {v[0]}/{v[1]}" for k, v in out.items())
        )
    return out


def main():
    """검색 방법을 비교해 outputs/eval_results/retrieval.csv 를 만든다."""
    parser = argparse.ArgumentParser(description="검색 방법 비교")
    parser.add_argument("--chunks", required=True, help="outputs/chunks 의 청크 이름")
    parser.add_argument(
        "--docs",
        default="cleaned_documents",
        help="질문을 즉석에서 뽑을 전처리본 (--evalset 없을 때만)",
    )
    parser.add_argument(
        "--evalset",
        default="eval_qa",
        help="data/ 의 평가 세트 이름. 없으면 --docs 에서 즉석 생성",
    )
    parser.add_argument(
        "--embed", default="tei", choices=["tei", "local", "openai", "fake"]
    )
    parser.add_argument(
        "--rerank", default="tei", choices=["tei", "local", "cohere", "fake"]
    )
    parser.add_argument(
        "--rerank-model",
        help="--rerank local 일 때 쓸 HuggingFace 모델. TEI 가 못 받는 걸 재볼 때.",
    )
    parser.add_argument(
        "--trust-remote-code", action="store_true", help="jina 처럼 커스텀 구조일 때"
    )
    parser.add_argument(
        "--bm25-weights",
        default="0.7",
        type=lambda s: [float(x) for x in s.split(",")],
        help="Hybrid 의 BM25 비중들. 쉼표로 구분",
    )
    parser.add_argument("--no-rerank", action="store_true", help="리랭커를 빼고 잰다")
    parser.add_argument(
        "--splade",
        action="store_true",
        help="Splade 희소 검색을 같이 잰다 (코퍼스 인코딩에 시간이 든다)",
    )
    parser.add_argument(
        "--splade-model",
        default=SpladeModel.PIXIE.value,
        choices=[m.value for m in SpladeModel],
        help="쓸 Splade 모델",
    )
    parser.add_argument(
        "--splade-batch",
        type=int,
        default=16,
        help="Splade 인코딩 배치 크기. 맥이 뜨거우면 8 이나 4 로 줄인다",
    )
    parser.add_argument(
        "--splade-url",
        nargs="?",
        const="tei",
        default=None,
        help="TEI 에 인코딩을 맡긴다. 그냥 주면 8084, 주소를 주면 그 주소",
    )
    parser.add_argument(
        "--pool",
        type=int,
        default=30,
        help="검색기가 돌려주는 개수. 예산으로 자르므로 넉넉히 준다",
    )
    parser.add_argument("--budget", type=int, default=settings.MAX_CONTEXT_CHARS)
    parser.add_argument(
        "--scoped",
        action="store_true",
        help="질문이 가리키는 공고 안에서만 찾는다. "
        "요약 카드를 만들 때의 실제 흐름이다",
    )
    parser.add_argument("--out", default=str(settings.EVAL_RESULTS / "retrieval.csv"))
    args = parser.parse_args()

    try:
        pairs = ev.load_evalset(args.evalset)
        print(f"평가 세트 {args.evalset}.json · 질문 {len(pairs)}개")
    except FileNotFoundError:
        pairs = ev.make_pairs_from_documents(load_documents(args.docs))
        print(f"평가 세트가 없어 즉석 생성 · 질문 {len(pairs)}개")

    # 답이 없는 질문(물러섬)은 검색 지표 대상이 아니다. 생성 평가에서 쓴다.
    skipped = [p for p in pairs if not p.get("keywords")]
    pairs = [p for p in pairs if p.get("keywords")]
    if skipped:
        print(f"  답이 없는 질문 {len(skipped)}개는 검색 지표에서 뺀다 (생성 평가용)")

    kinds = sorted({p.get("type", "전체") for p in pairs})
    counts = {k: sum(1 for p in pairs if p.get("type", "전체") == k) for k in kinds}
    print("  유형: " + " · ".join(f"{k} {n}문항" for k, n in counts.items()) + "\n")

    check_answers_exist(pairs, chunking.load_chunks(args.chunks))

    setups, searchers = build_setups(args)
    if args.scoped:
        where = {p["question"]: [p["doc_id"]] for p in pairs}
        setups = {n: scope_to_doc(f, searchers, where) for n, f in setups.items()}
        print("  공고를 질문마다 하나로 좁혀서 잰다 (--scoped)")
    metric = f"적중률@{args.budget}자"

    # 유형은 질문을 나눠 갖는다. 유형별로 돌려도 검색 횟수는 한 번 도는 것과 같다.
    frames = []
    for kind in kinds:
        subset = [p for p in pairs if p.get("type", "전체") == kind]
        print(f"\n[{kind}] {len(subset)}문항")
        part = ev.compare(setups, subset, budget=args.budget)
        part.insert(0, "유형", kind)
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)

    print("\n" + "=" * 78)
    print("유형 × 설정 — 적중률")
    print(df.pivot(index="설정", columns="유형", values=metric).to_string())
    print("\n유형 × 설정 — MRR")
    print(df.pivot(index="설정", columns="유형", values="MRR").to_string())

    # 머리말을 붙이면 청크가 길어져 예산에 덜 들어간다. 발췌 글자 수가 설정마다
    # 크게 다르면 그건 성능 차이가 아니라 예산을 덜 쓴 것이다.
    print("\n유형 × 설정 — 발췌 글자 (예산 %d자를 얼마나 썼나)" % args.budget)
    print(df.pivot(index="설정", columns="유형", values="평균글자").to_string())

    print("\n유형별 1위")
    for kind in kinds:
        part = df[df["유형"] == kind]
        hit = part.nlargest(1, metric).iloc[0]
        mrr = part.nlargest(1, "MRR").iloc[0]
        print(
            f"  {kind:<8} 적중률 {hit['설정']:<14} {hit[metric]:.3f}   "
            f"MRR {mrr['설정']:<14} {mrr['MRR']:.3f}"
        )
    print("\n리랭커는 적중률이 아니라 등수를 올리는 부품이다. MRR 이 올랐는지 보라.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
