#!/usr/bin/env python
"""옮긴 뒤 다 돌아가는지 한 번에 본다.

    python scripts/check_setup.py

보는 것: import 가 다 되는지 / 데이터가 제자리에 있는지 / 서버가 떠 있는지.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))   # 평평한 import: chunking, preprocessing …
sys.path.insert(0, str(ROOT))           # config.settings


def main():
    print("[1] import")
    import chunking
    import evaluate as ev
    from config import settings
    from preprocessing import load_documents
    from models import check_servers
    from pieces import BM25, Dense, Hybrid, Pipeline, Rerank
    from vectorstore import build_store

    print("    OK — 평평한 import 가 된다 (src 가 경로에 있음)")
    print(f"    프로젝트 폴더: {settings.ROOT}")

    print("\n[2] 데이터")
    for label, path in [
        ("원본 폴더", settings.RAW),
        ("메타 CSV", settings.META_CSV),
        ("추출 본문", settings.DOCUMENTS_JSONL),
    ]:
        mark = "O" if path.exists() else "X"
        extra = ""
        if path.exists() and path.is_dir():
            extra = f"  ({len(list(path.iterdir()))}개)"
        elif path.exists():
            extra = f"  ({path.stat().st_size / 1024**2:.1f}MB)"
        print(f"  {mark}  {label:<10} {path}{extra}")

    if settings.META_CSV.exists() and settings.RAW.exists():
        # CSV 에 적힌 파일이 실제로 있는지. 이름이 잘린 hwp 가 있어 자주 어긋난다.
        from preprocessing import load_metadata

        meta = load_metadata()
        missing = [n for n in meta["file_name"] if not (settings.RAW / str(n)).exists()]
        mark = "O" if not missing else "X"
        print(f"  {mark}  CSV↔파일     {len(meta) - len(missing)}/{len(meta)}건 일치")
        for name in missing[:5]:
            print(f"        없음: {name}")
        if len(missing) > 5:
            print(f"        … 외 {len(missing) - 5}건")

    if settings.DOCUMENTS_JSONL.exists():
        documents = load_documents()
        lengths = sorted(len(d["text"]) for d in documents)
        mid = lengths[len(lengths) // 2]
        print(f"     문서 {len(documents)}건 · 본문 중앙값 {mid:,}자 · 최소 {lengths[0]:,}자")
        if lengths[0] < 5000:
            print("     ⚠ 본문이 너무 짧은 문서가 있다. 추출이 덜 된 것일 수 있다.")
    else:
        print("     → python scripts/extract.py 를 먼저 돌린다")

    print("\n[3] 서버")
    check_servers()


if __name__ == "__main__":
    main()
