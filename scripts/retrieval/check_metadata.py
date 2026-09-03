"""원본 파일과 `data_list.csv` 가 맞는지만 본다.  python scripts/retrieval/check_metadata.py

전처리를 돌리기 **전에** 30초로 확인한다. 여기서 안 맞으면 파이프라인을 돌려도
제목·발주기관·금액이 빈 채로 색인까지 들어가고, 화면에서야 발견된다.

    data/raw 에는 있는데 CSV 에 없다   → 제목이 안 나온다. CSV 에 행을 채운다
    CSV 에는 있는데 data/raw 에 없다   → 파일을 못 받았다. 크롤러 로그를 본다
    공고번호가 빈 행                    → doc_id 가 nofile-<해시> 가 된다 (제목은 나온다)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import pandas as pd

from config import settings
from preprocessing.rfp.common import SUPPORTED_EXTENSIONS
from preprocessing.rfp.meta import _normalize_filename


# 병합이 실제로 쓰는 그 함수를 그대로 쓴다. 규칙을 여기 다시 적으면
# 언젠가 갈리고, 그러면 이 점검이 거짓말을 한다.
key = _normalize_filename


def main():
    raw = sorted(
        p for p in settings.RAW.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    df = pd.read_csv(settings.META_CSV)
    names = {c.replace(" ", ""): c for c in df.columns}
    for need in ("파일명", "사업명", "공고번호"):
        if need not in names:
            print(f"X CSV 에 '{need}' 컬럼이 없습니다. 컬럼: {list(df.columns)}")
            return 1

    df = df.rename(columns={v: k for k, v in names.items()})
    csv_keys = {key(n): n for n in df["파일명"].dropna()}
    raw_keys = {key(p.name): p.name for p in raw}

    print(f"data/raw {len(raw)}건 · CSV {len(df)}행\n")

    only_raw = sorted(set(raw_keys) - set(csv_keys))
    only_csv = sorted(set(csv_keys) - set(raw_keys))
    no_notice = df[df["공고번호"].isna() | (df["공고번호"].astype(str).str.strip() == "")]
    no_title = df[df["사업명"].isna() | (df["사업명"].astype(str).str.strip() == "")]

    print(f"CSV 에 없는 원본 {len(only_raw)}건   ← 이게 화면에 제목이 안 나오는 문서다")
    for k in only_raw[:30]:
        print(f"    {raw_keys[k]}")
    if len(only_raw) > 30:
        print(f"    … {len(only_raw) - 30}건 더")

    print(f"\n원본이 없는 CSV 행 {len(only_csv)}건")
    for k in only_csv[:10]:
        print(f"    {csv_keys[k]}")

    print(f"\n공고번호가 빈 행 {len(no_notice)}건  (doc_id 가 nofile-<해시> 가 된다)")
    print(f"사업명이 빈 행 {len(no_title)}건")

    ok = not only_raw
    print("\n" + ("전부 맞습니다" if ok else "맞춘 뒤에 prepare.py 를 돌리세요"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
