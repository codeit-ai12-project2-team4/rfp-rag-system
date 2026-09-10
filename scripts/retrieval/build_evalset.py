"""흩어진 평가 세트를 하나로 합치고, 채점이 되는 문항만 남긴다.

지금 평가 세트가 넷이다. 만든 사람도 방식도 다르고 결함도 제각각이다.

    data/eval_qa_160.json       팀원이 만든 160문항 (80문항판을 대체한다)
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
from pieces.search import chunk_signature  # noqa: E402

SOURCES = ["eval_qa_160.json", "eval_qa_gen.json", "eval_qa_notitle.json"]


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


def build(chunks, sources=None, keep_ambiguous=False, verbose=True):
    """세트들을 합치고 채점 안 되는 문항을 거른다.

    Args:
        chunks: 대조할 청크 리스트.
        sources (list[str], optional): 합칠 파일 이름들. 없으면 `SOURCES`.
        keep_ambiguous (bool): 여러 공고에 걸리는 문항도 남길지.
        verbose (bool): 진행 상황을 찍을지.

    Returns:
        tuple[list[dict], Counter, list[dict]]: 남은 문항, 사유별 개수,
        버린 문항들(`_사유` 가 붙는다). 마지막 것은 다른 팀에 넘길 목록이다.
    """
    bodies = owners(chunks)
    rows = []
    for name in sources or SOURCES:
        got = load(name)
        rows.extend(got)
        if verbose:
            print(f"  {name}: {len(got)}문항")

    kept, seen, dropped, out = [], set(), Counter(), []

    def toss(row, why):
        dropped[why] += 1
        out.append({**row, "_사유": why})

    for row in rows:
        question = squeeze(row.get("question", ""))
        keywords = [k for k in (row.get("keywords") or []) if squeeze(k)]

        if not keywords:
            toss(row, "빈키워드")
            continue
        if question in seen:
            toss(row, "중복질문")
            continue

        found = set()
        for keyword in keywords:
            needle = squeeze(keyword)
            found |= {
                doc
                for doc, parts in bodies.items()
                if any(needle in part for part in parts)
            }

        if not found:
            toss(row, "코퍼스에없음")
            continue
        if row.get("doc_id") not in found:
            toss(row, "라벨불일치")
            continue
        if len(found) > 1 and not keep_ambiguous:
            toss(row, "공고특정불가")
            continue

        seen.add(question)
        kept.append(row)
    return kept, dropped, out


def save(kept, dropped, out, made_from, verbose=True):
    """결과와 함께 **어느 청크로 만든 세트인지** 옆에 적어 둔다.

    Args:
        kept (list[dict]): 남은 문항들.
        dropped (Counter): 버린 사유별 개수.
        out (str): `data/` 아래 저장할 파일 이름.
        made_from (dict): {청크이름: 지문}. **여럿일 수 있다** — 교집합 세트는
            여러 코퍼스로 만들어지고, 그중 아무 코퍼스로 재도 유효하다.
        verbose (bool): 요약을 찍을지.

    Returns:
        Path: 저장한 경로.
    """
    path = settings.DATA / out
    path.write_text(json.dumps(kept, ensure_ascii=False, indent=2))
    meta(out).write_text(
        json.dumps(
            {
                "chunks": made_from,
                "문항수": len(kept),
                "버림": dict(dropped),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if verbose:
        print(f"\n{len(kept) + sum(dropped.values())}문항 → {len(kept)}문항 ({path.name})")
        for reason, count in dropped.most_common():
            print(f"  버림 {reason:12s} {count}")
        print("\n유형별")
        for kind, count in Counter(r.get("type") for r in kept).most_common():
            print(f"  {kind or '없음':10s} {count}")
    return path


def meta(out):
    """평가 세트 옆에 두는 도장 파일 경로.

    Args:
        out (str): 평가 세트 파일 이름.

    Returns:
        Path: 도장 파일 경로.
    """
    return settings.DATA / f"{Path(out).stem}.meta.json"


def ensure(out, chunks_name, chunks, verbose=True):
    """청크가 바뀌었으면 평가 세트를 다시 만든다.

    청크를 다시 자르면 정답 문자열이 사라지는 문항이 생긴다. 세트를 그대로 두면
    그게 "검색 실패" 로 잡혀서 **성능 저하로 오독하게 된다.** 실제로 v2 때
    그렇게 하루를 썼다. 사람이 기억할 일이 아니라 코드가 볼 일이다.

    손으로 만든 세트(도장이 없는 것)는 건드리지 않는다.

    Args:
        out (str): 평가 세트 파일 이름. 예: `eval_qa_merged.json`.
        chunks_name (str): 지금 쓰는 청크 이름.
        chunks: 지금 쓰는 청크 리스트.
        verbose (bool): 진행 상황을 찍을지.

    Returns:
        bool: 다시 만들었으면 True.
    """
    stamp = meta(out)
    if not stamp.exists():
        return False  # 자동 관리 대상이 아니다

    was = json.loads(stamp.read_text()).get("chunks") or {}
    now = chunk_signature(chunks)

    # 교집합 세트는 여러 코퍼스로 만들어진다. **그중 하나로 재는 건 정상이다.**
    # 이걸 안 보면 v3 로 잴 때 v4·v3 교집합 세트를 v3 단독으로 덮어쓰고,
    # 두 코퍼스를 서로 다른 문항으로 비교하게 된다. 실제로 그렇게 났다.
    if was.get(chunks_name) == now:
        return False

    print(f"\n청크가 바뀌었습니다 — {out} 를 다시 만듭니다")
    print(f"  만들 때  {', '.join(was) or '(없음)'}")
    print(f"  지금     {chunks_name} / {now}")
    kept, dropped, _ = build(chunks, verbose=verbose)
    save(kept, dropped, out, {chunks_name: now}, verbose=verbose)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunks",
        required=True,
        nargs="+",
        help="대조할 청크 이름. 여럿 주면 **전부에서** 채점되는 문항만 남긴다 — "
        "코퍼스끼리 비교할 때는 반드시 그렇게 해야 한다",
    )
    parser.add_argument("--sources", nargs="+", default=SOURCES)
    parser.add_argument("--out", default="eval_qa_merged.json")
    parser.add_argument(
        "--keep-ambiguous",
        action="store_true",
        help="여러 공고에 걸리는 문항도 남긴다",
    )
    args = parser.parse_args()

    kept, dropped, thrown = None, None, []
    for name in args.chunks:
        chunks = chunking.load_chunks(name)
        print(f"{name}: 청크 {len(chunks):,}개")
        got, why, tossed = build(chunks, args.sources, args.keep_ambiguous)
        if kept is None:
            kept, dropped, thrown = got, why, tossed
            continue
        # 교집합. 코퍼스가 다르면 살아남는 문항도 달라서, 안 맞추면 서로 다른
        # 시험지로 채점하게 된다. 8/28 에 그걸로 하루를 버렸다.
        alive = {squeeze(row["question"]) for row in got}
        gone = [r for r in kept if squeeze(r["question"]) not in alive]
        kept = [r for r in kept if squeeze(r["question"]) in alive]
        dropped["다른코퍼스에없음"] += len(gone)
        thrown += [{**r, "_사유": f"다른코퍼스에없음({name})"} for r in gone]

    made_from = {
        name: chunk_signature(chunking.load_chunks(name)) for name in args.chunks
    }
    save(kept, dropped, args.out, made_from)

    report = settings.DATA / f"{Path(args.out).stem}_dropped.json"
    report.write_text(json.dumps(thrown, ensure_ascii=False, indent=2))
    print(f"\n버린 문항 {len(thrown)}개 → {report.name}")

    # 다른 팀에 그대로 넘길 수 있게 사유별로 몇 개씩 보여준다
    for why in ["코퍼스에없음", "빈키워드", "라벨불일치"]:
        group = [r for r in thrown if r["_사유"] == why]
        if not group:
            continue
        print(f"\n[{why}] {len(group)}문항")
        for row in group[:6]:
            answer = (row.get("keywords") or [row.get("note", "")])[0]
            print(f"  {row.get('doc_id', '?'):<18} {row['question'][:44]}")
            print(f"  {'':<18} 정답  {str(answer)[:60]}")
        if len(group) > 6:
            print(f"  … 나머지 {len(group) - 6}개는 {report.name} 에")

    print("\n출처별")
    for name, count in Counter(r["source"] for r in kept).most_common():
        print(f"  {name:24s} {count}")


if __name__ == "__main__":
    main()
