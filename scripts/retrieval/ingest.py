"""전처리팀 파이프라인을 돌려 청크를 우리 이름 규칙으로 놓는다.

    python scripts/retrieval/ingest.py --docs cleaned_documents_v8

전처리팀 산출은 `<OUTPUT_DIR>/<이름>_chunks.jsonl` 이고 우리는
`outputs/chunks/chunks_<이름>__<크기>_<겹침>.jsonl` 로 읽는다. **지금까지 이 접착제가
사람 손이었다.** 신규 공고를 자동으로 받기 시작하면 첫날 어긋난다 — 이름이 틀리면
오류가 안 나고 옛 인덱스를 조용히 그대로 쓴다.

`--run` 없이 부르면 이미 만들어 둔 파일을 옮기기만 한다. 원본 파일은 안 건드린다.
"""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from config import retrieval as cfg
from config import settings


def target_name(docs, size, overlap):
    """우리 청크 이름. `chunk_name()` 과 같은 규칙이되 방식은 `pipeline` 이다.

    우리가 자른 게 아니라는 걸 이름에 남긴다. `recursive` 라고 쓰면 나중에
    `chunking.py` 로 다시 만들 수 있다고 착각하게 된다 — 표 원자성이 사라진다.
    """
    return f"{docs}__pipeline_{size}_{overlap}"


def main():
    parser = argparse.ArgumentParser(description="전처리팀 청크 → 우리 이름")
    parser.add_argument("--docs", default=cfg.DOCS, help="전처리본 이름")
    parser.add_argument("--size", type=int, default=cfg.SIZE)
    parser.add_argument("--overlap", type=int, default=cfg.OVERLAP)
    parser.add_argument("--src", help="전처리팀 산출 jsonl. 생략하면 찾아본다")
    parser.add_argument(
        "--run",
        action="store_true",
        help="전처리 파이프라인부터 돌린다 (원본 문서 필요)",
    )
    args = parser.parse_args()

    settings.make_dirs()
    name = target_name(args.docs, args.size, args.overlap)
    dst = settings.CHUNKS / f"{name}.jsonl"

    if args.run:
        from preprocessing.rfp import run_pipeline

        run_pipeline(
            settings.DATA / "raw",
            settings.PROCESSED,
            enable_chunk_output=True,
            chunk_size=args.size,
            chunk_overlap=args.overlap,
        )

    if args.src:
        src = Path(args.src)
    else:
        found = [p for p in settings.PROCESSED.glob(f"{args.docs}*chunk*.jsonl")]
        if not found:
            raise SystemExit(
                f"청크 파일을 못 찾았습니다: {settings.PROCESSED}/{args.docs}*chunk*.jsonl\n"
                f"  --src 로 경로를 주거나 --run 으로 파이프라인부터 돌리세요"
            )
        src = max(found, key=lambda p: p.stat().st_mtime)

    shutil.copyfile(src, dst)
    lines = sum(1 for _ in open(dst, encoding="utf-8"))
    print(f"{src}\n  → {dst}  ({lines:,}줄)")
    print(f"\n다음:  CHUNKS={name} python scripts/retrieval/prepare.py --build")


if __name__ == "__main__":
    main()
