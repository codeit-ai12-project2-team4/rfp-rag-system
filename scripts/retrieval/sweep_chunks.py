"""청크 크기를 바꿔가며 확정 설정(Hybrid+Rerank)을 잰다.

1200/200 은 **옛 전처리본(v1)** 에서 고른 값이다. 코퍼스가 v3 로 바뀌었으니
다시 재야 한다. 표 문법을 걷어내면서 글자가 줄었으므로 최적 크기도 옮겼을 수 있다.

크기마다 청킹 → 인덱스 2개 → 측정을 돈다. **한 크기에 20~30분** 걸리니
밤에 걸어 두고 자는 용도다. 중간 결과를 매번 CSV 에 덮어쓰므로 도중에 끊겨도
거기까지는 남는다.

    python scripts/retrieval/sweep_chunks.py --docs cleaned_documents_v3
    python scripts/retrieval/sweep_chunks.py --docs ... --sizes 800,1200,2000 --overlap-ratio 0.17

이미 만들어 둔 청크·인덱스가 있으면 건너뛴다. 다시 만들려면 --force.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import pandas as pd

import chunking
import evaluation as ev
from config import settings
from models import load_embedder, load_reranker
from pieces import BM25, Dense, Hybrid, Pipeline, Rerank
from vectorstore import list_stores, load_store


def run(command):
    """하위 명령을 돌린다. 실패하면 멈춘다."""
    print(f"  $ {' '.join(command)}")
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        sys.exit(f"실패: {' '.join(command)}")


def ensure(docs, size, overlap, embed, force):
    """그 크기의 청크와 인덱스를 준비한다. 이미 있으면 건너뛴다.

    Returns:
        청크 이름.
    """
    name = f"{docs}__recursive_{size}_{overlap}"
    chunk_file = settings.CHUNKS / f"chunks_{name}.jsonl"

    if force or not chunk_file.exists():
        run([sys.executable, "src/chunking.py", "--docs", docs,
             "--how", "recursive", "--size", str(size), "--overlap", str(overlap)])

    for suffix in ("", "__header"):
        if force or f"{name}{suffix}__{embed}" not in list_stores():
            run([sys.executable, "src/vectorstore.py",
                 "--chunks", f"{name}{suffix}"] + (["--force"] if force else []))
    return name


def score(name, pairs, embed, rerank, pool, top_k, budget):
    """확정 설정 하나로 유형별 지표를 낸다."""
    embedder = load_embedder(embed)
    chunks = chunking.load_chunks(name)
    pipeline = Pipeline([
        Hybrid(
            [Dense(load_store(f"{name}__{embed}", embedder), k=pool),
             BM25(chunks, k=pool)],
            k=pool, pool=pool,
        ),
        Rerank(load_reranker(rerank), k=top_k),
    ])

    by_type = {}
    for pair in pairs:
        by_type.setdefault(pair.get("type", "기타"), []).append(pair)

    rows = []
    for kind, group in by_type.items():
        frame = ev.compare({name: lambda q: pipeline(q).chunks}, group,
                           budget=budget, verbose=False)
        row = frame.iloc[0].to_dict()
        row["유형"] = kind
        rows.append(row)
    return rows


def main():
    """크기별로 청킹·인덱싱·측정을 돌고 표로 찍는다."""
    parser = argparse.ArgumentParser(description="청크 크기를 스윕한다.")
    parser.add_argument("--docs", default="cleaned_documents_v3")
    parser.add_argument("--sizes", default="600,900,1200,1800",
                        type=lambda s: [int(x) for x in s.split(",")])
    parser.add_argument("--overlap-ratio", type=float, default=1 / 6,
                        help="겹침 = 크기 × 이 값 (1200/200 이 1/6 이다)")
    parser.add_argument("--evalset", default="eval_qa_80")
    parser.add_argument(
        "--embed", default="tei", choices=["tei", "local", "openai", "fake"]
    )
    parser.add_argument(
        "--rerank", default="tei", choices=["tei", "local", "cohere", "fake"]
    )
    parser.add_argument("--pool", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--budget", type=int, default=settings.MAX_CONTEXT_CHARS)
    parser.add_argument("--force", action="store_true", help="있어도 다시 만든다")
    parser.add_argument("--out", default=str(settings.EVAL_RESULTS / "chunk_sizes.csv"))
    args = parser.parse_args()

    pairs = [p for p in ev.load_evalset(args.evalset)
             if p.get("answerable", True) and p.get("keywords")]
    print(f"{len(pairs)}문항 · 크기 {args.sizes}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for size in args.sizes:
        overlap = int(size * args.overlap_ratio)
        print(f"\n{'=' * 60}\n[{size}/{overlap}]")
        started = time.time()
        name = ensure(args.docs, size, overlap, args.embed, args.force)
        for row in score(name, pairs, args.embed, args.rerank,
                         args.pool, args.top_k, args.budget):
            row["크기"] = f"{size}/{overlap}"
            rows.append(row)
        # 도중에 끊겨도 거기까지는 남긴다
        pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
        print(f"  {time.time() - started:.0f}초 · 저장 → {out}")

    table = pd.DataFrame(rows)
    hit = [c for c in table.columns if c.startswith("적중률")][0]
    print("\n" + "=" * 60)
    for metric in (hit, "MRR"):
        print(f"\n크기 × 유형 — {metric}")
        print(table.pivot(index="크기", columns="유형", values=metric))
    print(f"\n저장 → {out}")


if __name__ == "__main__":
    main()
