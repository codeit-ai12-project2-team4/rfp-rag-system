"""지금 설정에 맞는 파생물이 다 있는지 보고, 없으면 만든다.

    python scripts/retrieval/prepare.py            # 확인만
    python scripts/retrieval/prepare.py --build    # 없거나 어긋나면 만든다

이번 주에 같은 사고를 다섯 번 냈다. 전부 **오류 없이 다른 것을 보고 있었다.**

    v3 인덱스에 v4 임베더        차원이 같으면 안 죽는다
    옛 청크로 만든 Splade npz    청크 개수까지 같아서 안 죽는다
    옛 청크로 만든 평가 세트      정답이 사라진 걸 '검색 실패' 로 잡는다
    Dense 는 v3 · Hybrid 는 v4   한 실행 안에서 코퍼스가 섞인다
    retriever.CHUNKS 가 옛것      API 가 옛 설정으로 답한다

설정은 `.env` 와 `config/retrieval.py` 한 곳에만 있다. 여기서는 그것과 디스크를
대조한다. **사람이 기억할 일이 아니다.**
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import chunking
from config import retrieval as cfg
from config import settings


def run(*args):
    """파이썬 스크립트를 하위 프로세스로 돌린다.

    같은 프로세스에서 부르면 모듈 수준 상수가 이미 굳어 있어서, 방금 만든
    것을 못 본다. 프로세스를 새로 띄우는 게 안전하다.

    Args:
        *args: `python` 뒤에 붙일 인자들.

    Returns:
        bool: 성공 여부.
    """
    print(f"    $ python {' '.join(str(a) for a in args)}")
    return subprocess.run([sys.executable, *map(str, args)], cwd=ROOT).returncode == 0


def check(build=False, service=False):
    """파생물 넷을 차례로 본다.

    Args:
        build (bool): 없거나 어긋나면 만들지.
        service (bool): 서비스에 필요한 것까지만 본다(평가 세트를 건너뛴다).

    Returns:
        bool: 전부 준비됐으면 True.
    """
    chunks = cfg.chunk_name()
    index = cfg.index_name()
    ok = True

    print("== 지금 설정 ==")
    for key in (
        "DOCS",
        "HOW",
        "SIZE",
        "OVERLAP",
        "EMBED",
        "RERANK",
        "STORE",
        "POOL",
        "TOP_K",
    ):
        print(f"  {key:<9} {getattr(cfg, key)}")
    print(f"  청크      {chunks}")
    print(f"  인덱스     {index}")

    ran = False

    def preprocess():
        """원본 hwp/pdf → 전처리본 + 청크. **한 실행에서 한 번만 돈다.**

        `run_pipeline` 이 둘을 같이 만든다. [1] 과 [2] 가 각각 부르면 원본을
        두 번 읽는다 — 100건에 몇 분이라 두 번은 못 봐준다.

        Returns:
            bool: 돌렸으면 True. 원본이 없어서 못 돌렸으면 False.
        """
        nonlocal ran
        if ran:
            return True
        raw = [f for f in settings.RAW.glob("*") if f.is_file()]
        if not raw:
            print(f"    X 원본이 없습니다: {settings.RAW}")
            print("      python src/crawl.py --days 1     부터 돌리세요")
            return False
        print(f"    → 전처리 파이프라인을 돌립니다 (원본 {len(raw):,}건, 몇 분 걸립니다)")
        from preprocessing.rfp import run_pipeline

        run_pipeline(
            raw_path=settings.RAW,
            eda_output_path=settings.PROCESSED,
            prep_output_path=settings.OUTPUTS,
            metadata_path=settings.META_CSV,
            enable_chunk_output=True,
            # **명시해서 넘긴다.** 안 주면 run_pipeline 의 기본 인자가 쓰이는데,
            # 그건 build.py 를 import 할 때 한 번 굳는다. 위에서 cfg.SIZE 를
            # "지금 설정" 이라고 찍어 놓고 다른 값으로 자르면 그게 이 파일이
            # 막으려는 사고 그 자체다.
            chunk_size=cfg.SIZE,
            chunk_overlap=cfg.OVERLAP,
        )
        ran = True
        return True

    # 1. 전처리본. 없으면 여기서 만든다 — 청크까지 같이 나온다.
    docs = settings.PROCESSED / f"{cfg.DOCS}.jsonl"
    print(f"\n[1] 전처리본  {docs.name}")
    if not docs.exists() and build:
        print("    X 없습니다")
        preprocess()
    if not docs.exists():
        print("    X 없습니다 (--build 로 전처리부터 돌립니다)")
        return False
    print("    O")

    # 2. 청크
    path = settings.CHUNKS / f"{chunks}.jsonl"
    print(f"\n[2] 청크  {path.name}")
    if path.exists():
        print(f"    O  {sum(1 for _ in open(path)):,}개")
    elif cfg.CHUNKS and build:
        # 전처리팀 파이프라인이 만든다. `chunking.py` 로 만들면 이름만 비슷하고
        # 표 원자성도 검색용/생성용 두 벌도 없는 다른 물건이 나온다.
        print("    X 없습니다")
        preprocess()  # [1] 에서 이미 돌았으면 그냥 지나간다
        if not path.exists():
            print(f"    X 파이프라인이 돌았는데도 {path.name} 이 없습니다")
            print(f"      config 의 CHUNKS 와 파이프라인이 쓰는 이름이 다를 수 있습니다")
            return False
        print(f"    O  {sum(1 for _ in open(path)):,}개")
    elif cfg.CHUNKS:
        print("    X 없습니다 (--build 로 전처리부터 돌립니다)")
        return False
    elif build:
        ok = (
            run(
                "src/chunking.py",
                "--docs",
                cfg.DOCS,
                "--how",
                cfg.HOW,
                "--size",
                cfg.SIZE,
                "--overlap",
                cfg.OVERLAP,
            )
            and ok
        )
    else:
        print("    X 없습니다 (--build 로 만듭니다)")
        ok = False

    if not path.exists():
        return False

    # 3. 인덱스. 모델까지 대조한다 — 차원이 같으면 faiss 가 안 죽는다.
    #
    # **STORE 가 가리키는 쪽만 본다.** 절을 하나 더 만들지 않는 이유는,
    # 둘 다 검사하면 안 쓰는 쪽이 없다고 매번 빨간 줄이 뜨기 때문이다.
    # 이름(index)은 양쪽이 같고 폴더만 다르다.
    lance = cfg.STORE == "lance"
    print(f"\n[3] 인덱스  {index}  ({'lancedb' if lance else 'faiss'})")
    meta = (
        (settings.LANCEDB / f"{index}.json")
        if lance
        else (settings.VECTORSTORE / index / "meta.json")
    )
    stale = False
    if meta.exists():
        was = json.loads(meta.read_text())
        print(f"    O  {was.get('chunks')}청크 · {was.get('dim')}차원")
        print(f"       만든 모델  {was.get('model')}")
        # 이름도 개수도 같은데 내용만 바뀐 경우는 이것만 잡는다
        from pieces.search import chunk_signature

        now = chunk_signature(chunking.load_chunks(chunks))
        if was.get("signature") is None:
            print("    ? 청크 지문이 없는 옛 인덱스입니다. 다시 만드는 게 안전합니다")
            stale = True
        elif was["signature"] != now:
            print(
                f"    X 이 청크로 만든 게 아닙니다 (만들 때 {was['signature']} / 지금 {now})"
            )
            stale = True
    elif (
        settings.LANCEDB / f"{index}.lance" if lance else settings.VECTORSTORE / index
    ).exists():
        print("    ? 도장이 없는 옛 인덱스입니다. 다시 만드는 게 안전합니다")
        stale = True
    else:
        print("    X 없습니다")
        stale = True

    if stale and build:
        maker = "src/lance_store.py" if lance else "src/vectorstore.py"
        ok = run(maker, "--chunks", chunks, "--force") and ok
    elif stale:
        ok = False

    # 4. 평가 세트. **서비스에는 안 쓴다.**
    #
    # API 가 읽는 건 [2] 청크와 [3] 인덱스까지다. 평가 세트는 측정 도구
    # (compare_retrieval·eval_notices)만 쓴다. 그래서 --service 면 건너뛴다.
    #
    # 안 건너뛰면 크론이 조용히 멈춘다 — 평가 세트가 없으면 ok=False 가 되고,
    # `prepare.py --build && systemctl restart` 의 && 가 안 넘어간다. 색인은
    # 새로 만들어졌는데 API 는 옛것을 물고 있는 상태가 된다.
    if service:
        print("\n[4] 평가 세트  건너뜀 (--service)")
        print("\n" + ("전부 준비됐습니다" if ok else "빠진 게 있습니다"))
        return ok

    stem = cfg.EVALSET
    print(f"\n[4] 평가 세트  {stem}.json")
    stamp = settings.DATA / f"{stem}.meta.json"
    if not (settings.DATA / f"{stem}.json").exists():
        print("    X 없습니다")
        if build:
            ok = (
                run(
                    "scripts/retrieval/build_evalset.py",
                    "--chunks",
                    chunks,
                    "--out",
                    f"{stem}.json",
                )
                and ok
            )
        else:
            ok = False
    elif stamp.exists():
        made = json.loads(stamp.read_text()).get("chunks") or {}
        now = None
        try:
            from pieces.search import chunk_signature

            now = chunk_signature(chunking.load_chunks(chunks))
        except Exception as error:  # 청크를 못 읽으면 대조를 건너뛴다
            print(f"    ? 지문 확인 불가: {error}")
        if now and made.get(chunks) == now:
            print(f"    O  {json.loads(stamp.read_text()).get('문항수')}문항")
        elif now:
            print(
                f"    ! 이 청크로 만든 게 아닙니다 (만들 때: {', '.join(made) or '없음'})"
            )
            print("      compare_retrieval 이 시작할 때 알아서 다시 만듭니다")
    else:
        print("    ? 도장이 없습니다. 손으로 만든 세트면 정상입니다")

    print(
        "\n" + ("전부 준비됐습니다" if ok else "빠진 게 있습니다 — --build 로 만드세요")
    )
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="없거나 어긋나면 만든다")
    parser.add_argument(
        "--service",
        action="store_true",
        help="서비스에 필요한 것까지만 (평가 세트 건너뜀). 크론이 쓴다",
    )
    args = parser.parse_args()
    sys.exit(0 if check(args.build, args.service) else 1)


if __name__ == "__main__":
    main()
