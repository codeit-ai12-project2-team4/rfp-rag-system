"""청크 크기를 바꿔가며 확정 설정을 잰다.

1200/200 은 **옛 전처리본(v1)** 에서 고른 값이다. 코퍼스가 바뀌었으니 다시 잰다.

크기마다 **하위 프로세스로** 청킹 → 인덱스 2개 → `compare_retrieval.py` 를 돌린다.
한 크기가 죽어도 나머지는 계속 간다 — 실제로 600/100 에서 세그폴트가 났고,
그때 스윕 전체가 같이 죽었다. 프로세스를 나누면 메모리도 매번 반납된다.

    python scripts/retrieval/sweep_chunks.py --docs cleaned_documents_v3
    python scripts/retrieval/sweep_chunks.py --docs ... --sizes 800,1200,1800

한 크기에 20~30분. 결과는 크기마다 따로 저장하고 마지막에 합친다.
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

from config import settings
from vectorstore import list_stores

HERE = Path(__file__).parent


def run(command, label):
    """하위 명령을 돌린다. 실패해도 멈추지 않고 알려만 준다."""
    print(f"  $ {' '.join(str(c) for c in command)}")
    result = subprocess.run([str(c) for c in command], cwd=ROOT)
    if result.returncode != 0:
        print(f"  ⚠ {label} 실패 (코드 {result.returncode}) — 이 크기는 건너뜁니다")
        return False
    return True


def main():
    """크기별로 돌리고 결과를 합쳐 표로 찍는다."""
    parser = argparse.ArgumentParser(description="청크 크기를 스윕한다.")
    parser.add_argument("--docs", default="cleaned_documents_v3")
    parser.add_argument("--sizes", default="900,1200,1800,2400",
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
    parser.add_argument("--force", action="store_true", help="있어도 다시 만든다")
    parser.add_argument("--out", default=str(settings.EVAL_RESULTS / "chunk_sizes.csv"))
    args = parser.parse_args()

    frames = []
    for size in args.sizes:
        overlap = int(size * args.overlap_ratio)
        name = f"{args.docs}__recursive_{size}_{overlap}"
        print(f"\n{'=' * 60}\n[{size}/{overlap}]")
        started = time.time()

        if args.force or not (settings.CHUNKS / f"chunks_{name}.jsonl").exists():
            if not run([sys.executable, "src/chunking.py", "--docs", args.docs,
                        "--how", "recursive", "--size", size, "--overlap", overlap],
                       "청킹"):
                continue

        ok = True
        for suffix in ("", "__header"):
            if args.force or f"{name}{suffix}__{args.embed}" not in list_stores():
                ok = run([sys.executable, "src/vectorstore.py",
                          "--chunks", f"{name}{suffix}", "--embed", args.embed]
                         + (["--force"] if args.force else []), "인덱싱")
                if not ok:
                    break
        if not ok:
            continue

        out = settings.EVAL_RESULTS / f"chunk_{size}_{overlap}.csv"
        if not run([sys.executable, HERE / "compare_retrieval.py",
                    "--chunks", name, "--evalset", args.evalset, "--scoped",
                    "--embed", args.embed, "--rerank", args.rerank,
                    "--bm25-weights", "0.5", "--out", out], "측정"):
            continue

        frame = pd.read_csv(out, encoding="utf-8-sig")
        frame["크기"] = f"{size}/{overlap}"
        frames.append(frame)
        pd.concat(frames, ignore_index=True).to_csv(
            args.out, index=False, encoding="utf-8-sig"
        )
        print(f"  {time.time() - started:.0f}초 · 저장 → {args.out}")

    if not frames:
        sys.exit("성공한 크기가 없습니다.")

    table = pd.concat(frames, ignore_index=True)
    best = table[table["설정"] == "Hybrid+Rerank"]
    hit = [c for c in table.columns if c.startswith("적중률")][0]
    print("\n" + "=" * 60)
    for metric in ("MRR", hit):
        print(f"\n크기 × 유형 — {metric} (Hybrid+Rerank)")
        print(best.pivot(index="크기", columns="유형", values=metric))
    print(f"\n저장 → {args.out}")


if __name__ == "__main__":
    main()
