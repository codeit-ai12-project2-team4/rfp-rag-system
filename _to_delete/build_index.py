#!/usr/bin/env python
"""청크를 만들고 FAISS 인덱스를 만든다.

    python scripts/build_index.py                      기본 설정
    python scripts/build_index.py --how recursive --size 1200
    python scripts/build_index.py --embed fake         서버 없이 배관만 확인
    python scripts/build_index.py --dry-run            자르기만 하고 임베딩은 안 함

노트북에서 설정을 정한 뒤, 그 설정을 여기에 적어 돌리면 팀원 누구나
같은 인덱스를 만들 수 있다.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))   # 평평한 import: chunking, preprocessing …
sys.path.insert(0, str(ROOT))           # config.settings

from config import settings

import chunking

from preprocessing import load_documents
from models import load_embedder
from vectorstore import build_store, estimate_cost


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--how",
        default="section",
        choices=["section", "recursive", "semantic"],
        help="자르는 방법",
    )
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=150)
    parser.add_argument("--embed", default="tei", choices=["tei", "local", "fake"])
    parser.add_argument("--name", default=None, help="인덱스 이름 (생략하면 자동)")
    parser.add_argument("--dry-run", action="store_true", help="자르기만")
    parser.add_argument("--force", action="store_true", help="이미 있어도 다시 만든다")
    args = parser.parse_args()

    documents = load_documents()
    print(f"문서 {len(documents)}건")

    embedder = (
        load_embedder(args.embed)
        if args.how == "semantic" or not args.dry_run
        else None
    )

    if args.how == "recursive":
        chunks = chunking.split_recursive(
            documents, size=args.size, overlap=args.overlap
        )
    elif args.how == "section":
        chunks = chunking.split_by_section(
            documents, size=args.size, overlap=args.overlap
        )
    else:
        chunks = chunking.split_semantic(documents, embedder)

    stats = chunking.chunk_stats(chunks)
    print(f"자르기: {args.how} size={args.size} overlap={args.overlap}")
    for key, value in stats.items():
        print(f"  {key:<8} {value:,}")

    name = args.name or f"{args.how}_{args.size}_{args.overlap}__{args.embed}"
    chunking.save_chunks(chunks, name)
    print(f"청크 저장 → {settings.CHUNKS / f'chunks_{name}.jsonl'}")

    if args.embed == "local" or args.embed == "tei":
        cost = estimate_cost(chunks)
        print(f"참고: OpenAI 였다면 약 ${cost['예상비용(달러)']} (TEI/로컬은 0원)")

    if args.dry_run:
        print("\n(dry-run — 인덱스는 만들지 않았습니다)")
        return

    build_store(chunks, embedder, name=name, force=args.force)
    print(f"\n인덱스 이름: {name}")
    print(f"노트북에서:  store = load_store('{name}', embedder)")


if __name__ == "__main__":
    main()
