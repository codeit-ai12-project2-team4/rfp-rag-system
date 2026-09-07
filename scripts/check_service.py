"""돌고 있는 VM 에 붙어 전 구간을 돌려 본다. **팀 공용 도구다.**

    python scripts/check_service.py --health              배선 확인 (제일 먼저)
    python scripts/check_service.py                       대화형 — 찾기 → 고르기 → 묻기
    python scripts/check_service.py --ask "질문"           공고를 안 고르고 한 번만
    python scripts/check_service.py --eval eval_qa_both   평가 한 바퀴 걸고 지켜보기

**로컬에 아무것도 안 깔아도 된다.** `test_main.py` 는 `retriever` 를 직접 불러서
TEI·인덱스·BM25 를 전부 로컬에 올려야 했다. 여기서는 API 를 부른다.
**표준 라이브러리만 쓴다** — 이 파일 하나만 있으면 어디서든 돈다.

준비 — 둘 중 하나

    export RFP_API=http://<VM 주소>:8010
    export RFP_TOKEN=<API_TOKEN>

저장소 안에서 돌리면 옆의 `.env` 에서 `API_TOKEN` 을 알아서 읽는다.
**토큰을 명령줄 인자로 안 받는다.** 적으면 셸 기록에 남는다.

VM 밖이라면 터널을 먼저 연다.

    ssh -L 8010:localhost:8010 <VM>
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_API = "http://localhost:8010"


def settings():
    """API 주소와 토큰. 환경변수가 먼저고, 없으면 저장소의 `.env` 를 본다.

    Returns:
        tuple[str, str]: (주소, 토큰). 없으면 기본값과 빈 문자열.
    """
    api = os.environ.get("RFP_API", "").rstrip("/")
    token = os.environ.get("RFP_TOKEN", "")
    env = Path(__file__).resolve().parents[1] / ".env"
    if env.exists() and not (api and token):
        for line in env.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            value = value.strip().strip('"').strip("'")
            if not token and key.strip() == "API_TOKEN":
                token = value
            if not api and key.strip() == "RFP_API":
                api = value.rstrip("/")
    return api or DEFAULT_API, token


API, TOKEN = settings()


def call(path, body=None, stream=False):
    """API 한 번. `body` 가 있으면 POST, 없으면 GET.

    Args:
        path: `/search` 처럼 슬래시로 시작하는 경로.
        body: 보낼 dict. None 이면 GET.
        stream: True 면 응답을 줄 단위로 흘린다 (`/ask/stream` 용).

    Returns:
        dict, 또는 stream 이면 줄 제너레이터.

    Raises:
        SystemExit: 못 부를 때. **왜 안 되는지 적고 끝낸다** — 팀원이 스택
            트레이스를 읽게 하지 않는다.
    """
    request = urllib.request.Request(API + path, method="POST" if body else "GET")
    request.add_header("Content-Type", "application/json")
    if TOKEN:
        request.add_header("x-api-token", TOKEN)
    data = json.dumps(body, ensure_ascii=False).encode() if body else None

    try:
        response = urllib.request.urlopen(request, data, timeout=600)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:300]
        if error.code == 401:
            sys.exit(f"토큰이 틀립니다. RFP_TOKEN 또는 .env 의 API_TOKEN 을 보세요.\n  {detail}")
        if error.code == 404:
            sys.exit(f"{path} 가 없습니다. VM 이 옛 코드일 수 있습니다.\n  {detail}")
        sys.exit(f"{error.code} {path}\n  {detail}")
    except urllib.error.URLError as error:
        sys.exit(
            f"{API} 에 못 붙습니다 ({error.reason}).\n"
            "  RFP_API 를 확인하세요. VM 밖이라면 터널을 먼저 여세요:\n"
            "    ssh -L 8010:localhost:8010 <VM>"
        )

    if stream:
        return (line.decode("utf-8") for line in response)
    return json.loads(response.read().decode())


def money(value):
    """1억 5,000만원. 없으면 빈칸."""
    if not value:
        return ""
    eok, man = int(value) // 10**8, int(value) % 10**8 // 10**4
    return (f"{eok}억 " if eok else "") + (f"{man:,}만" if man else "") + "원"


def health():
    """무엇이 떠 있고 무엇을 보고 있는지. **막히면 여기부터 본다.**"""
    got = call("/health")
    print(f"{'API':10s}{API}")
    print(f"{'토큰':10s}{'있음' if TOKEN else '없음'}")
    for key in ("ok", "embedder", "reranker", "generator", "store", "index", "chunks"):
        print(f"{key:10s}{got.get(key)}")
    refresh = got.get("refresh")
    if refresh:
        state = "성공" if refresh.get("ok") else f"실패 ({refresh.get('step')})"
        print(f"{'갱신':10s}{state} · {str(refresh.get('at'))[:19]}")
    if not got.get("ok"):
        print("\n! embedder 나 reranker 가 비어 있으면 검색이 안 됩니다.")


def show(answer):
    """답변·근거와 **잰 값**을 찍는다. 정량 확인은 마지막 줄로 한다."""
    if not answer.get("ok"):
        print(f"\n  [오류] {answer.get('error')}")
        return

    found = answer.get("sources") or []
    if found:
        print("\n  근거")
        for source in found:
            title = (source.get("title") or "")[:42]
            print(f"    [{source['n']}] {title} · {source['chunk_id']}")

    # 답변이 실제로 인용을 달았는지, 없는 번호를 쓰지 않았는지.
    # 인용정확도를 손으로 확인하는 판이다.
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer.get("answer") or "")}
    ghost = cited - {s["n"] for s in found}
    bits = [f"발췌 {len(found)}", f"인용 {len(cited)}"]
    if ghost:
        bits.append(f"** 없는 번호 {sorted(ghost)}")
    for key, label in (("search_sec", "검색"), ("latency_sec", "생성"), ("total_sec", "합")):
        if answer.get(key) is not None:
            bits.append(f"{label} {answer[key]:.1f}초")
    if answer.get("model"):
        bits.append(str(answer["model"]))
    print("\n  " + " · ".join(bits))


def ask(question, doc_ids=None, model="mini"):
    """`/ask/stream` 으로 묻고 글자를 흘리며 받는다.

    스트리밍이 없는 서버(옛 코드)면 `/ask` 로 조용히 되돌아간다.

    Returns:
        dict: `/ask` 와 같은 모양.
    """
    body = {"question": question, "model": model}
    if doc_ids:
        body["doc_ids"] = doc_ids

    try:
        lines = call("/ask/stream", body, stream=True)
    except SystemExit:
        return call("/ask", body)

    answer = {"ok": True, "answer": "", "sources": []}
    out = []
    print()
    for line in lines:
        if not line.strip():
            continue
        event = json.loads(line)
        kind = event.get("type")
        if kind == "meta":
            answer["search_sec"] = event["search_sec"]
            answer["sources"] = event["sources"]
        elif kind == "delta":
            out.append(event["text"])
            print(event["text"], end="", flush=True)
        elif kind == "done":
            for key in ("model", "latency_sec", "total_sec"):
                answer[key] = event.get(key)
        elif kind == "error":
            answer["ok"] = False
            answer["error"] = event["error"]
    answer["answer"] = "".join(out)
    print()
    return answer


def pick():
    """1단계 — 목적을 받아 공고를 찾고 하나를 고르게 한다.

    Returns:
        dict or None: 고른 공고. 엔터만 치면 None.
    """
    query = input("\n어떤 사업을 찾으세요? (엔터=끝) > ").strip()
    if not query:
        return None

    started = time.time()
    rows = call("/search", {"query": query, "top_n": 10})
    print(f"  {len(rows)}건 · {time.time() - started:.1f}초\n")
    if not rows:
        print("  못 찾았습니다. 다른 말로 해보세요.")
        return pick()

    for i, row in enumerate(rows, 1):
        print(f"  [{i}] {(row.get('title') or '')[:56]}")
        print(f"      {row.get('agency', '')}  {money(row.get('budget'))}")

    choice = input(f"\n번호 (1-{len(rows)}, 엔터=다시, a=고르지 않고 묻기) > ").strip()
    if choice == "a":
        return {"doc_id": None, "title": f"「{query}」 — 공고를 안 고름"}
    if not choice.isdigit() or not 1 <= int(choice) <= len(rows):
        return pick()
    return rows[int(choice) - 1]


def conversation(notice, model):
    """2단계 — 고른 공고 안에서 계속 묻는다. 엔터만 치면 나간다."""
    print(f"\n── {(notice.get('title') or '')[:56]}")
    print("   질문하세요. 엔터만 치면 공고를 다시 고릅니다.")
    while True:
        question = input("\n질문 > ").strip()
        if not question:
            return
        doc_ids = [notice["doc_id"]] if notice.get("doc_id") else None
        show(ask(question, doc_ids, model))


def watch(job_id):
    """평가 작업을 끝날 때까지 지켜본다. 로그는 새 줄만 이어 찍는다."""
    seen = 0
    while True:
        job = call(f"/eval/{job_id}")
        for line in job.get("log", [])[seen:]:
            print(f"  {line}")
        seen = len(job.get("log", []))

        if job["status"] != "running":
            print(f"\n{job['status']} · {job['step']}")
            if job.get("metrics"):
                print(json.dumps(job["metrics"], ensure_ascii=False, indent=2))
            if job.get("error"):
                print(job["error"])
            return
        if job.get("total"):
            print(f"  … {job['step']} {job['done']}/{job['total']}", end="\r")
        time.sleep(3)


def main():
    parser = argparse.ArgumentParser(
        description="돌고 있는 VM 에 붙어 전 구간을 돌려 본다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", default="mini", help="답변 모델 키 (기본 mini)")
    parser.add_argument("--health", action="store_true", help="배선만 확인하고 끝")
    parser.add_argument("--ask", help="공고를 안 고르고 한 번만 묻는다")
    parser.add_argument("--eval", help="평가 한 바퀴. 세트 이름을 준다")
    parser.add_argument("--sets", action="store_true", help="쓸 수 있는 평가 세트 목록")
    parser.add_argument("--limit", type=int, help="평가를 앞에서 몇 문항만")
    parser.add_argument("--no-judge", action="store_true", help="충실성 채점을 뺀다 (공짜)")
    args = parser.parse_args()

    if args.health:
        return health()

    if args.sets:
        for row in call("/evalsets"):
            print(f"  {row['name']:28s} {row['count']:4d}문항")
        return

    if args.eval:
        body = {"evalset": args.eval, "model": args.model, "judge": not args.no_judge}
        if args.limit:
            body["limit"] = args.limit
        job_id = call("/eval", body)["id"]
        print(f"작업 {job_id}\n")
        return watch(job_id)

    if args.ask:
        return show(ask(args.ask, model=args.model))

    got = call("/health")
    print(f"{API} · 코퍼스 {got.get('chunks')} · 모델 {args.model}")
    while True:
        notice = pick()
        if notice is None:
            print("\n끝.")
            return
        conversation(notice, args.model)


if __name__ == "__main__":
    main()
