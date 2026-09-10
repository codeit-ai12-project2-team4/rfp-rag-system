"""전처리본(문서 단위) → 우리 청크. **전처리팀 청커를 그대로 쓴다.**

    python scripts/retrieval/chunk_docs.py --docs cleaned_documents_v7
    python scripts/retrieval/chunk_docs.py --docs cleaned_documents_v2 --docs cleaned_documents_v3

`ingest.py` 는 전처리팀이 **이미 잘라 놓은** `*_chunks.jsonl` 을 옮기는 도구다.
문서 단위 jsonl 만 있을 때 쓸 창구가 없었다 — 판본 비교(v2~v8)를 하려니 그게 막혔다.

`--run` 은 답이 아니다. 그건 **지금 코드로 파이프라인을 다시 도는 것**이라 v2 를
넣어도 v2 가 안 나온다. 옛 판본은 그때 코드의 산출물이고 우리에겐 그 결과물만 있다.

그래서 이 스크립트는 **자르기만 한다.** `preprocessing.rfp.chunk.chunk_pairs()` —
`build.py` 가 쓰는 바로 그 함수다. 판마다 같은 칼을 대야 비교가 성립한다.

    자르는 대상  page_content_for_generation (없으면 page_content)
    길이 기준    검색용 문자 수 (chunk_pairs 가 그렇게 잰다)
    이름         chunks_<전처리본>__pipeline_<크기>_<겹침>

**정답 보존율을 같이 찍는다.** 전처리본이 바뀌면 정답 문자열이 청크에서 사라질 수
있다 — 9/8 에 `<br>` 하나로 5문항이 매칭 안 된 걸 겪었다. 그 판의 성적이 낮으면
검색이 나쁜 게 아니라 **정답이 애초에 없는 것**이고, 둘은 전혀 다른 이야기다.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from config import retrieval as cfg  # noqa: E402
from config import settings  # noqa: E402
from evaluation import load_evalset  # noqa: E402
from evaluation.evalset import normalize  # noqa: E402
from preprocessing.rfp.chunk import chunk_pairs  # noqa: E402


def cut(docs_path, out_path, size, overlap):
    """문서 jsonl 을 청크 jsonl 로. `build.py` 와 같은 레코드 모양으로 쓴다."""
    total = 0
    with docs_path.open(encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            meta = dict(row.get("metadata") or {})
            source = row.get("page_content_for_generation") or row.get("page_content") or ""
            pairs = chunk_pairs(source, chunk_size=size, chunk_overlap=overlap)
            for i, (search_chunk, gen_chunk) in enumerate(pairs):
                dst.write(json.dumps({
                    "page_content": search_chunk,
                    "page_content_for_generation": gen_chunk,
                    "metadata": {**meta, "chunk_index": i, "chunk_total": len(pairs)},
                }, ensure_ascii=False) + "\n")
                total += 1
    return total


def gold_survival(out_path, pairs):
    """정답 문자열이 청크 어딘가에 남아 있나.

    **이걸 안 보면 전처리 손실을 검색 성적으로 오해한다.** 공백을 지우고 본다 —
    전처리가 줄바꿈을 손보면 글자 그대로는 안 맞는다.

    Returns:
        (살아있는 문항 수, 잰 문항 수)
    """
    body = "".join(
        normalize(json.loads(line)["page_content"])
        for line in out_path.open(encoding="utf-8")
        if line.strip()
    )
    alive = asked = 0
    for pair in pairs:
        keywords = pair.get("keywords") or []
        if not keywords:
            continue  # 답이 없는 문항(`없음` 유형)은 검색 지표 대상이 아니다
        asked += 1
        alive += any(normalize(k) in body for k in keywords if k)
    return alive, asked


def main():
    parser = argparse.ArgumentParser(description="전처리본 → 청크 (판본 비교용)")
    parser.add_argument("--docs", action="append", required=True,
                        help="data/processed 의 전처리본 이름. 여러 번 줄 수 있다")
    parser.add_argument("--size", type=int, default=cfg.SIZE)
    parser.add_argument("--overlap", type=int, default=cfg.OVERLAP)
    parser.add_argument("--evalset", default=cfg.EVALSET, help="정답 보존율을 잴 세트")
    args = parser.parse_args()

    settings.make_dirs()
    try:
        pairs = load_evalset(args.evalset)
    except FileNotFoundError as e:
        print(f"정답 보존율은 건너뜁니다 — {e}")
        pairs = []

    print(f"{'전처리본':<28}{'문서':>6}{'청크':>8}{'정답 보존':>12}")
    print("-" * 56)
    for docs in args.docs:
        docs_path = settings.PROCESSED / f"{docs}.jsonl"
        if not docs_path.exists():
            print(f"{docs:<28}  없음 ({docs_path})")
            continue
        name = f"chunks_{docs}__pipeline_{args.size}_{args.overlap}"
        out_path = settings.CHUNKS / f"{name}.jsonl"
        docs_n = sum(1 for line in docs_path.open(encoding="utf-8") if line.strip())
        chunks_n = cut(docs_path, out_path, args.size, args.overlap)
        if pairs:
            alive, asked = gold_survival(out_path, pairs)
            rate = f"{alive}/{asked} ({alive / asked * 100:.0f}%)" if asked else "—"
        else:
            rate = "—"
        print(f"{docs:<28}{docs_n:>6}{chunks_n:>8}{rate:>12}")

    print(f"\n다음:  CHUNKS=<이름> python scripts/retrieval/prepare.py --build --service")
    print("정답 보존율이 낮은 판은 성적이 아니라 **전처리 손실**을 재게 된다.")


def demo():
    """레코드 모양과 보존율 계산만 본다. 파일도 인덱스도 안 쓴다."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        docs = Path(tmp) / "d.jsonl"
        docs.write_text(json.dumps({
            "page_content_for_generation": "가나다 " * 400 + "\n배정예산 : 1억 5,000만원\n" + "라마바 " * 400,
            "metadata": {"source": "x.hwp", "공고번호": "1"},
        }, ensure_ascii=False) + "\n", encoding="utf-8")
        out = Path(tmp) / "c.jsonl"
        n = cut(docs, out, 1500, 250)
        rows = [json.loads(line) for line in out.open(encoding="utf-8")]
        assert n == len(rows) > 1, n
        assert set(rows[0]) == {"page_content", "page_content_for_generation", "metadata"}
        assert rows[0]["metadata"]["source"] == "x.hwp"
        assert rows[0]["metadata"]["chunk_total"] == n

        alive, asked = gold_survival(out, [
            {"keywords": ["배정예산 : 1억 5,000만원"]},   # 공백이 달라도 맞아야 한다
            {"keywords": ["있을 리 없는 문자열"]},
            {"keywords": []},                            # 답 없는 문항은 안 센다
        ])
        assert (alive, asked) == (1, 2), (alive, asked)
        print(f"demo ok: 청크 {n}개 · 보존 {alive}/{asked}")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        main()
