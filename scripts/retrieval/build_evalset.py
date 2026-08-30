"""흩어진 평가 세트를 하나로 합치고, 채점이 되는 문항만 남긴다.

지금 평가 세트가 넷이다. 만든 사람도 방식도 다르고 결함도 제각각이다.

    data/eval_qa_80.json        팀원이 만든 80문항
    data/eval_qa_gen.json       ollama 로 만든 101문항
    data/eval_qa_notitle.json   내가 만든 133문항 (사업명 제거판)
    data/eval_qa.json           같은 133문항 (사업명 포함 — 안 쓴다)

문제는 문항 수가 아니라 **한 유형에 9문항짜리 칸이 생기는 것**이다. 한 문제가
0.11 을 흔들면 무엇을 재도 판정이 안 된다. 합쳐서 유형별 표본을 키운다.

거르는 기준은 전부 "채점이 되느냐" 하나다. 검색이 어려운 문항은 남긴다.

    빈키워드      채점할 정답 문자열이 없다
    중복질문      같은 질문이 두 세트에 있다
    코퍼스에없음   정답 문자열이 현재 청크 어디에도 없다 (전처리본이 바뀌면 생긴다)
    라벨불일치    정답은 있는데 라벨이 가리키는 공고에 없다
    공고특정불가   정답 문자열이 여러 공고에 있다 — 질문만으로 공고를 못 고른다

마지막 둘이 핵심이다. `eval_qa_gen` 의 doc_hits 가 0.562 인 이유가 여기 있다.

    python scripts/retrieval/build_evalset.py --chunks cleaned_documents_v3__recursive_1200_200
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import chunking  # noqa: E402
from config import settings  # noqa: E402

SOURCES = ["eval_qa_80.json", "eval_qa_gen.json", "eval_qa_notitle.json"]


def squeeze(text):
    """공백을 지운다. 띄어쓰기 차이로 못 찾는 일을 막는다.

    Args:
        text: 아무 값.

    Returns:
        str: 공백 없는 문자열.
    """
    return re.sub(r"\s+", "", str(text))


def load(name):
    """평가 세트 하나를 읽는다. json 도 jsonl 도 받는다.

    Args:
        name (str): `data/` 아래 파일 이름.

    Returns:
        list[dict]: 문항 리스트. 파일이 없으면 빈 리스트.
    """
    path = settings.DATA / name
    if not path.exists():
        print(f"  ({name} 없음 — 건너뜁니다)")
        return []
    raw = path.read_text()
    try:
        rows = json.loads(raw)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    for row in rows:
        row["source"] = name
    return rows


def owners(chunks):
    """공고별로 본문을 한 덩어리로 붙인다.

    청크 9,200개를 문항마다 통째로 훑는 대신 공고로 묶어둔다. 정답을 찾으면
    그 공고에서 바로 멈출 수 있어 실제 비교 횟수가 크게 준다.

    Args:
        chunks: 청크 리스트.

    청크를 이어 붙이지는 않는다. 채점이 청크 단위라, 두 청크에 걸쳐 있는
    정답은 어차피 못 맞힌다. 붙여버리면 그런 문항을 살려두게 된다.

    Returns:
        dict[str, list[str]]: doc_id → 공백 없는 청크 본문들.
    """
    joined = {}
    for chunk in chunks:
        doc = chunk.metadata.get("doc_id")
        joined.setdefault(doc, []).append(squeeze(chunk.page_content))
    return joined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", required=True, help="대조할 청크 이름")
    parser.add_argument("--sources", nargs="+", default=SOURCES)
    parser.add_argument("--out", default="eval_qa_merged.json")
    parser.add_argument(
        "--keep-ambiguous",
        action="store_true",
        help="여러 공고에 걸리는 문항도 남긴다",
    )
    args = parser.parse_args()

    chunks = chunking.load_chunks(args.chunks)
    print(f"청크 {len(chunks):,}개로 대조합니다")
    bodies = owners(chunks)
    print(f"공고 {len(bodies)}건")

    rows = []
    for name in args.sources:
        got = load(name)
        rows.extend(got)
        print(f"  {name}: {len(got)}문항")

    kept, seen, dropped = [], set(), Counter()
    for row in rows:
        question = squeeze(row.get("question", ""))
        keywords = [k for k in (row.get("keywords") or []) if squeeze(k)]

        if not keywords:
            dropped["빈키워드"] += 1
            continue
        if question in seen:
            dropped["중복질문"] += 1
            continue

        # 정답이 실제로 들어 있는 공고들을 모은다
        found = set()
        for keyword in keywords:
            needle = squeeze(keyword)
            found |= {
                doc
                for doc, parts in bodies.items()
                if any(needle in part for part in parts)
            }

        if not found:
            dropped["코퍼스에없음"] += 1
            continue
        if row.get("doc_id") not in found:
            dropped["라벨불일치"] += 1
            continue
        if len(found) > 1 and not args.keep_ambiguous:
            dropped["공고특정불가"] += 1
            continue

        seen.add(question)
        kept.append(row)

    out = settings.DATA / args.out
    out.write_text(json.dumps(kept, ensure_ascii=False, indent=2))

    print(f"\n{len(rows)}문항 → {len(kept)}문항 ({out.name})")
    for reason, count in dropped.most_common():
        print(f"  버림 {reason:12s} {count}")
    print("\n유형별")
    for kind, count in Counter(r.get("type") for r in kept).most_common():
        print(f"  {kind or '없음':10s} {count}")
    print("\n출처별")
    for name, count in Counter(r["source"] for r in kept).most_common():
        print(f"  {name:24s} {count}")


if __name__ == "__main__":
    main()
