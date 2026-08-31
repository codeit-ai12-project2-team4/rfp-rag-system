"""팀원이 만든 평가 세트를 우리 형식으로 바꾼다.

새 스크립트를 짤 필요는 없다. 세트만 우리 형식으로 바꿔 놓으면
`compare_retrieval.py`, `sweep_pool.py`, `misses.py` 가 `--evalset` 으로
그대로 받는다. 문제는 형식이 아니라 **정답 필드**다.

    doc_id          파일명            →  공고번호-차수 (metadata 의 source 로 찾는다)
    eval_category   유형              →  type
    question_type   unanswerable 표시  →  answerable
    evidence_text   원문 근거          →  keywords

`evidence_text` 를 그대로 정답으로 쓰면 안 된다. 문서 여러 곳을 이어 붙였고
쪽번호와 표 찌꺼기가 섞여 있어서, **글자 그대로는 코퍼스에 38~53% 밖에 없다.**
그 상태로 재면 검색이 아무리 좋아도 적중률이 0.5 에서 막힌다.

그래서 근거와 문서 본문의 **가장 긴 연속 일치 조각**을 찾아 그걸 정답으로 쓴다.
문서에 실제로 있는 문자열이라 `matches()` 를 손대지 않아도 되고, 기존 133문항
세트와 같은 잣대로 잰다.

    python scripts/adapt_evalset.py --src eval_qa_80 --docs cleaned_documents_v3
"""

import argparse
import ast
import difflib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # scripts/retrieval/ 아래다
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from config import settings

MIN_SPAN = 20  # 이보다 짧은 조각은 아무 문서에나 걸려서 정답 구실을 못 한다


def squeeze(text):
    """공백을 뺀 문자열과, 각 글자가 원문 몇 번째였는지.

    `matches()` 가 공백을 무시하므로 여기서도 무시하고 비교한다. 원문 위치를
    같이 들고 다녀야 찾은 조각을 원래 표기로 돌려줄 수 있다.
    """
    chars, index = [], []
    for i, char in enumerate(text):
        if not char.isspace():
            chars.append(char)
            index.append(i)
    return "".join(chars), index


def load_corpus(docs_name):
    """`파일명 → doc_id`, `doc_id → 본문` 을 만든다. 두 형식을 다 받는다."""
    path = settings.PROCESSED / f"{docs_name}.jsonl"
    if not path.exists():
        sys.exit(f"없음: {path}")

    by_name, body = {}, {}
    for line in open(path, encoding="utf-8"):
        row = json.loads(line)
        if "meta" in row:                       # 우리 형식
            meta, text = row["meta"], row["text"]
            doc_id, name = str(meta.get("doc_id")), meta.get("file_name") or ""
        else:                                   # 팀원 형식
            meta = row["metadata"]
            meta = ast.literal_eval(meta) if isinstance(meta, str) else meta
            text = row["page_content"]
            doc_id = f"{meta.get('공고번호')}-{str(meta.get('공고차수')).replace('.0', '')}"
            name = meta.get("source") or ""
        by_name[name] = doc_id
        by_name[Path(name).stem] = doc_id
        body[doc_id] = body.get(doc_id, "") + "\n" + text
    return by_name, body


def _overlap(span, answer):
    """조각이 답을 담고 있는 정도. 답을 8자씩 잘라 몇 조각이나 들어 있는지 본다."""
    if not answer:
        return 0.0
    grams = {answer[i : i + 8] for i in range(0, max(1, len(answer) - 8), 4)}
    return sum(1 for gram in grams if gram in span) / max(1, len(grams))


def best_span(evidence, text, answer=""):
    """근거와 본문이 겹치는 조각 중 **답을 담은** 것을 원문 표기로 돌려준다.

    제일 긴 조각을 그냥 쓰면 엉뚱한 데를 집는다. 실제로 "개발 비용은
    얼마인가요?" 질문에서 근거에 사업기간과 사업비가 같이 있었는데, 사업기간
    줄이 더 길게 겹쳐서 그쪽이 정답이 돼 버렸다. 그래서 겹치는 조각을 여러 개
    모아 두고 `answer` 와 제일 많이 겹치는 것을 고른다.

    Args:
        evidence: 팀원이 적어 준 `evidence_text`.
        text: 그 공고의 본문 전체.
        answer: 팀원이 적어 준 참고 답안. 조각을 고르는 기준이다.

    Returns:
        `(조각, 길이)`. 본문이 없으면 `("", 0)`.
    """
    flat_evidence = re.sub(r"\s+", "", evidence)
    flat_answer = re.sub(r"\s+", "", answer)
    flat_text, index = squeeze(text)
    if not flat_evidence or not flat_text:
        return "", 0

    matcher = difflib.SequenceMatcher(None, flat_evidence, flat_text, autojunk=False)
    blocks = [b for b in matcher.get_matching_blocks() if b.size >= MIN_SPAN]
    if not blocks:
        longest = matcher.find_longest_match(0, len(flat_evidence), 0, len(flat_text))
        blocks = [longest] if longest.size else []
    if not blocks:
        return "", 0

    def pick(block):
        piece = flat_text[block.b : block.b + block.size]
        return (_overlap(piece, flat_answer), block.size)

    best = max(blocks, key=pick)
    start, end = index[best.b], index[best.b + best.size - 1]
    return text[start : end + 1].strip(), best.size


def main():
    """팀원 세트를 우리 형식으로 저장하고, 정답을 얼마나 건졌는지 찍는다."""
    parser = argparse.ArgumentParser(description="평가 세트를 우리 형식으로.")
    parser.add_argument("--src", default="eval_qa_80", help="data/ 의 jsonl 이름")
    parser.add_argument("--docs", default="cleaned_documents_v3", help="맞춰 볼 코퍼스")
    parser.add_argument("--out", help="저장 이름 (생략하면 --src 와 같게)")
    parser.add_argument("--min-span", type=int, default=MIN_SPAN)
    args = parser.parse_args()

    src = settings.DATA / f"{args.src}.jsonl"
    if not src.exists():
        sys.exit(f"없음: {src}")

    by_name, body = load_corpus(args.docs)
    rows = [json.loads(line) for line in open(src, encoding="utf-8") if line.strip()]

    pairs, weak, unmapped = [], [], 0
    kept = Counter()
    for row in rows:
        name = row["doc_id"]
        doc_id = by_name.get(name) or by_name.get(Path(name).stem)
        kind = row.get("eval_category") or "기타"
        answerable = row.get("question_type") != "unanswerable" and kind != "없음"

        span, size = "", 0
        if answerable:
            if doc_id is None:
                unmapped += 1
            else:
                span, size = best_span(
                    row.get("evidence_text") or "",
                    body[doc_id],
                    row.get("answer") or "",
                )
                kept[(kind, size >= args.min_span)] += 1
                if size < args.min_span:
                    weak.append((kind, size, row["question"][:55]))

        pairs.append({
            "question": row["question"],
            "keywords": [span] if size >= args.min_span else [],
            "doc_id": doc_id,
            "type": kind,
            "answerable": answerable,
            "note": (row.get("answer") or "")[:200],   # 생성 채점용 참고 답안
            "checked_by": "team",
            "span_chars": size,
        })

    out = settings.DATA / f"{args.out or args.src}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    print(f"{len(pairs)}문항 → {out}")
    print(f"유형: {dict(Counter(p['type'] for p in pairs))}")
    if unmapped:
        print(f"⚠ doc_id 를 못 찾은 문항 {unmapped}개")

    print(f"\n정답 조각을 {args.min_span}자 이상 건진 비율 ({args.docs} 기준)")
    for kind in sorted({k for k, _ in kept}):
        ok, no = kept[(kind, True)], kept[(kind, False)]
        print(f"  {kind:<6} {ok:>2}/{ok + no:<2} ({ok / (ok + no):.0%})")

    if weak:
        print(f"\n조각이 짧아 정답을 비워 둔 {len(weak)}문항 — 채점에서 빠진다")
        for kind, size, question in weak[:8]:
            print(f"  [{kind}] {size}자  {question}…")

    print(f"\n다음:  python scripts/compare_retrieval.py --chunks <청크> "
          f"--evalset {args.out or args.src}")


if __name__ == "__main__":
    main()
