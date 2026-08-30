"""청크에서 질답을 만든다. **정답을 청크에 못박아서.**

지금 쓰는 평가 세트 두 개가 각각 다른 이유로 부정확하다.

    내가 만든 133문항   질문의 100%가 「사업명」으로 시작한다. 청크에 사업명을
                       붙이는 기법이 좋아 보였던 게 그 탓이었다.
    팀원의 80문항       `evidence_text` 가 문서의 딴 곳을 가리키는 게 3~4개 있다.
                       거기서 정답 스팬을 뽑느라 오차가 한 겹 더 얹혔다.

**문제는 누가 질문을 만드느냐가 아니라 정답을 어디에 붙이느냐다.** 여기서는
청크 하나를 주고 "그 안에서 그대로 복사한 한 줄"을 정답으로 받는다. 받은 뒤
**진짜로 그 청크에 있는지 대조해서, 아니면 버린다.** 그래서

- 정답이 코퍼스에 없는 일이 구조적으로 안 생긴다
- `chunk_id` 가 남아서 코퍼스가 바뀌면 바로 안다
- 사업명이 들어간 질문을 걸러낼 수 있다

    ollama pull gpt-oss:20b
    python scripts/retrieval/make_evalset.py --chunks <청크이름> --n 120 --llm ollama
    python scripts/retrieval/make_evalset.py --chunks <청크이름> --n 120 --llm openai
"""

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import chunking
from config import settings
from models import load_llm

SYSTEM = "\n".join([
    "당신은 제안요청서(RFP)로 검색 평가 문제를 만드는 사람입니다.",
    "규칙:",
    "- 질문 하나와 정답 한 줄을 만듭니다.",
    "- **정답은 주어진 발췌에서 글자 그대로 복사**합니다. 요약하거나 바꾸지 않습니다.",
    "- 정답은 20자 이상 120자 이하로, 답이 실제로 담긴 부분을 고릅니다.",
    "- 질문에 사업명이나 기관명을 쓰지 않습니다. 사용자는 그걸 모르고 묻습니다.",
    "- 발췌만 보고 답할 수 있는 질문만 만듭니다.",
    '- JSON 하나만 출력합니다: {"question": "...", "answer": "..."}',
])

# 유형을 섞어야 한 축만 재는 걸 피한다. 지금 세트가 유형별로 성격이 아주 다르다.
STYLES = {
    "배점": "평가표의 항목과 점수를 묻습니다. 예: '기술평가는 몇 점인가?'",
    "요구사항": "요구사항 목록의 한 항목을 묻습니다. 예: 'SFR-001 은 무엇인가?'",
    "의역": (
        "**발췌에 없는 표현으로** 묻습니다. 문서가 '사업예산'이라 쓰면 질문은 "
        "'돈이 얼마나 드나'로 씁니다. 라벨을 그대로 인용하지 않습니다."
    ),
}


def squeeze(text):
    """공백을 지운다. 대조할 때 띄어쓰기 차이로 떨어지지 않게."""
    return re.sub(r"\s+", "", str(text))


def parse(raw):
    """LLM 이 뱉은 것에서 JSON 을 건져낸다. 코드블록이 자주 붙는다."""
    text = raw.strip()
    if "```" in text:
        text = text.split("```")[1]
        text = text.removeprefix("json")
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def pick_chunks(chunks, n, min_chars, seed):
    """길고 내용 있는 청크만 고른다. 목차나 표지에서는 문제가 안 나온다."""
    usable = [
        c for c in chunks
        if len(c.page_content) >= min_chars and c.metadata.get("title")
    ]
    random.Random(seed).shuffle(usable)
    return usable[:n]


def main():
    """청크에서 질답을 만들고, 정답이 그 청크에 있는 것만 남긴다."""
    parser = argparse.ArgumentParser(description="평가 세트를 만든다.")
    parser.add_argument("--chunks", required=True, help="청크 이름 (__header 없이)")
    parser.add_argument("--n", type=int, default=120, help="시도할 문항 수")
    parser.add_argument("--llm", default="ollama",
                        choices=["ollama", "openai", "vllm", "hf", "echo"])
    parser.add_argument("--model", help="모델 이름. 생략하면 종류별 기본값")
    parser.add_argument("--min-chars", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="eval_qa_gen", help="data/ 에 저장할 이름")
    args = parser.parse_args()

    chunks = chunking.load_chunks(args.chunks)
    picked = pick_chunks(chunks, args.n, args.min_chars, args.seed)
    print(f"청크 {len(chunks):,}개 중 {len(picked)}개로 문제를 만듭니다 ({args.llm})")

    llm = load_llm(args.llm, model=args.model)
    kinds = list(STYLES)
    pairs, dropped = [], Counter()
    started = time.time()

    for i, chunk in enumerate(picked, 1):
        kind = kinds[i % len(kinds)]
        body = chunk.page_content[:2500]
        title = chunk.metadata.get("title") or ""
        user = "\n".join([
            f"[유형] {kind} — {STYLES[kind]}",
            "",
            "[발췌]",
            body,
        ])
        try:
            raw = llm.ask(SYSTEM, user, max_tokens=400)
        except Exception as error:
            dropped["호출실패"] += 1
            print(f"  {i}/{len(picked)} 호출 실패: {str(error)[:60]}", end="\r")
            continue

        item = parse(raw)
        if not item or not item.get("question") or not item.get("answer"):
            dropped["JSON실패"] += 1
            continue

        answer = item["answer"].strip()
        question = item["question"].strip()

        # 여기가 핵심 — 정답이 **그 청크에 글자 그대로** 있어야 한다
        if squeeze(answer) not in squeeze(body):
            dropped["정답이발췌에없음"] += 1
            continue
        if not 20 <= len(answer) <= 200:
            dropped["정답길이"] += 1
            continue
        # 사업명이 든 질문은 검색을 쉽게 만든다. 실사용과도 멀다.
        if title and squeeze(title[:12]) in squeeze(question):
            dropped["질문에사업명"] += 1
            continue

        pairs.append({
            "question": question,
            "keywords": [answer],
            "doc_id": chunk.metadata.get("doc_id"),
            "chunk_id": chunk.metadata.get("chunk_id"),   # 정답이 붙은 자리
            "type": kind,
            "answerable": True,
            "checked_by": f"{args.llm}:{llm.name}",
        })
        print(f"  {i}/{len(picked)} · 통과 {len(pairs)}", end="\r")

    print(" " * 40, end="\r")
    path = settings.DATA / f"{args.out}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    print(f"\n{len(pairs)}문항 → {path}  ({time.time() - started:.0f}초)")
    print(f"유형: {dict(Counter(p['type'] for p in pairs))}")
    if dropped:
        print("\n버린 것")
        for reason, count in dropped.most_common():
            print(f"  {reason:<16} {count}")
    print("\n버린 비율이 높으면 모델이 발췌를 그대로 못 옮기는 것이다.")
    print("더 큰 모델로 바꾸거나 --min-chars 를 올려 본다.")
    print(f"\n다음:  python scripts/retrieval/compare_retrieval.py "
          f"--chunks {args.chunks} --evalset {args.out} --scoped")


if __name__ == "__main__":
    main()
