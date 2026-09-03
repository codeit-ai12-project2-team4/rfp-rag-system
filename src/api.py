"""웹앱이 부르는 HTTP 경계. 엔드포인트 두 개가 전부다.

프론트(Next.js)는 별도 repo 다. 파이썬 소스를 볼 이유가 없고 배포 주기도 다르다.
둘을 잇는 건 이 파일뿐이라, 여기만 안 바뀌면 양쪽이 따로 움직인다.

    uvicorn src.api:app --reload --port 8088

    POST /search   자연어로 공고 찾기 (1단계 — 사람이 목록에서 고르는 화면)
    POST /ask      고른 공고 안에서 질문 (2단계 — 발췌 → 답변 → 출처)
    GET  /file/{doc_id}  그 공고의 원본 RFP 내려받기
    GET  /health   무엇이 떠 있고 무엇을 보고 있는지 (Vercel 에서 열면 배선 전체가 보인다)
    GET  /models   드롭다운에 채울 모델 목록

**시나리오 A/B 는 여기 없다.** 그건 우리 인프라 선택이지 고객의 선택이 아니고,
B 는 원본 RFP 를 외부 API 로 내보낸다. 환경변수로 배포 시점에 정한다.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import hmac
import json
import re
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from config import MODEL_CONFIGS, settings
import evalrun
from evaluation import load_evalset
from generation import generate_answer
from retriever import (
    build_context,
    file_for,
    retrieve,
    search_notices,
    sources,
)

app = FastAPI(title="입찰메이트 RFP")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("UI_ORIGINS", "http://localhost:3000").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


# 포트 8010 은 외부에 열려 있다. 원본 RFP 는 NDA 이고, `POST /eval` 은 누르면
# OpenAI 요금이 나간다. 토큰 하나로 막는다.
#
# **브라우저가 이 토큰을 들고 다니면 안 된다.** 그러면 번들에 박혀 공개된다.
# UI 쪽은 Next 의 라우트 핸들러(`app/api/[...path]/route.ts`)가 **서버에서**
# 붙인다. 그래서 `next.config.ts` 의 rewrite 를 걷어냈다 — rewrite 는 요청을
# 그대로 넘기기만 해서 헤더를 붙일 자리가 없다.
API_TOKEN = os.environ.get("API_TOKEN", "")
OPEN_PATHS = {"/health"}  # 감시용. 여기엔 데이터가 없다


@app.middleware("http")
async def guard(request: Request, call_next):
    """토큰이 맞아야 통과. `API_TOKEN` 이 비어 있으면 검사하지 않는다.

    비어 있을 때 막지 않는 이유는, 막으면 토큰을 넣기 전에 배포한 순간
    화면이 통째로 죽기 때문이다. 대신 뜰 때 경고하고 `/health` 에
    `auth: false` 로 드러낸다 — 조용히 열려 있는 것보다 낫다.
    """
    if API_TOKEN and request.url.path not in OPEN_PATHS:
        sent = request.headers.get("x-api-token", "")
        # 글자 수로 새는 것까지 막는다. 짧은 코드에 비용이 없다.
        if not hmac.compare_digest(sent, API_TOKEN):
            return JSONResponse({"detail": "토큰이 없거나 틀립니다"}, status_code=401)
    return await call_next(request)


class Search(BaseModel):
    query: str
    top_n: int = 10
    min_budget: int | None = None
    max_budget: int | None = None
    agency: str | None = None


class Upload(BaseModel):
    name: str
    # 파일 내용을 그대로. **multipart 를 안 쓴다** — 그러면 `python-multipart` 를
    # 새로 깔아야 하는데, 평가 세트는 커야 수백 KB 라 그럴 값이 없다.
    # 브라우저가 `file.text()` 로 읽어 보낸다.
    content: str


class Eval(BaseModel):
    evalset: str
    model: str = "mini"
    judge: bool = True
    judge_model: str = "nano"
    limit: int | None = None
    generation: bool = True
    # False 면 공고를 안 알려주고 검색부터 — 전 구간 E2E
    scoped: bool = True


class Ask(BaseModel):
    question: str
    doc_ids: list[str] | None = None
    model: str = "mini"
    history: list[dict] | None = None


@app.on_event("startup")
def warm():
    """뜰 때 BM25 색인을 지어 둔다.

    청크 9,200개를 형태소 분석하는 데 6초 걸린다. 안 해두면 **첫 사용자가**
    그 6초를 낸다. 워커는 하나로 둔다 — FAISS·청크·BM25 가 워커마다 통째로
    복제되어 2GB 씩 먹는다.
    """
    retrieve("준비")
    evalrun.sweep()  # 재시작 전에 돌던 평가는 끝난 걸로 표시한다
    if not API_TOKEN:
        print("경고: API_TOKEN 이 비어 있습니다. 8010 이 통째로 열려 있습니다.")


@app.get("/models")
def models():
    """드롭다운에 채울 모델 목록.

    `ready` 가 False 면 **고르는 순간 첫 답변이 1~2분 걸린다.** GPU 한 장에
    생성 모델을 하나만 올릴 수 있어서, 다른 모델을 고르면 컨테이너를
    갈아끼우기 때문이다. 화면에서 이 사실을 알려주라고 붙인 필드다.
    """
    from models.sglang import current

    loaded = current()
    return [
        {
            "key": key,
            "name": cfg.model,
            "provider": cfg.provider,
            "ready": cfg.provider != "sglang" or cfg.model == loaded,
            # 화면의 예상 비용이 이 값으로 계산된다. VM 모델은 0 이다.
            "usd_per_call": cfg.usd_per_call,
        }
        for key, cfg in MODEL_CONFIGS.items()
    ]


@app.post("/search")
def search(body: Search):
    """공고를 찾는다. 리랭커는 안 쓴다 — 목록 화면이라 1위보다 목록 안이 중요하다."""
    return search_notices(**body.model_dump())


@app.post("/ask")
def ask(body: Ask):
    """질문에 답한다. 발췌와 출처를 같이 준다 — 근거 없이 답만 주면 못 쓴다."""
    chunks = retrieve(body.question, doc_ids=body.doc_ids)
    result = generate_answer(
        model_key=body.model,
        query=body.question,
        context=build_context(chunks),
        history=body.history,
    )
    return {**result, "sources": sources(chunks)}


@app.get("/file/{doc_id}")
def file(doc_id: str):
    """공고의 원본 RFP 를 내려준다.

    **나라장터 링크가 아니라 우리가 받아둔 파일을 준다.** 답변의 근거가 된 그
    문서여야 하기 때문이다. 공고가 변경·재공고되면 나라장터 쪽 파일은 바뀐다.

    경로는 `data_list.csv` 에 적힌 이름으로만 만든다 — `doc_id` 를 그대로
    경로에 붙이지 않는다.
    """
    found = file_for(doc_id)
    if not found:
        raise HTTPException(404, f"원본 파일이 없습니다: {doc_id}")
    path, name = found
    return FileResponse(path, filename=name, media_type="application/octet-stream")


@app.get("/evalsets")
def evalsets():
    """`data/` 에 있는 평가 세트와 문항 수, 그리고 예상 비용.

    **시작 전에 얼마 드는지 보여 주려는 것이다.** 팀 예산이 $20 인데 191문항
    한 바퀴가 $1.34 다. 눌러 놓고 나중에 아는 것과 누르기 전에 아는 것은 다르다.
    """
    rows = []
    for path in sorted(settings.DATA.glob("*.json")) + sorted(settings.DATA.glob("*.jsonl")):
        if path.stem.endswith(".meta"):
            continue
        try:
            count = len(load_evalset(path.stem))
        except Exception:  # noqa: BLE001  평가 세트가 아닌 json 이 섞여 있다
            continue
        # 비용은 모델을 골라야 정해지므로 여기서 안 낸다. 화면이
        # `GET /models` 의 `usd_per_call` 과 곱한다 — 출처가 하나여야 한다.
        rows.append({"name": path.stem, "count": count})
    return rows


# 업로드한 세트는 이 접두어를 단다. **채점이 끝나면 지운다** — 안 지우면
# `data/` 에 남의 파일이 무한정 쌓인다. 지우는 판정도 이 접두어로 한다.
UPLOAD_PREFIX = "upload_"
UPLOAD_MAX = 5 << 20  # 5MB. 191문항짜리가 1MB 남짓이다


@app.post("/eval/upload")
def eval_upload(body: Upload):
    """평가 세트를 올린다. `.json`(배열)과 `.jsonl`(한 줄에 하나) 둘 다 받는다.

    저장은 항상 `.json` 배열로 한다 — `load_evalset` 이 그것만 읽는다.
    **여기서 파싱해 보고 안 되면 받지 않는다.** 안 그러면 몇 분 뒤 평가가
    엉뚱한 데서 죽고, 원인이 업로드였다는 걸 알기 어렵다.

    Returns:
        `{"evalset": 이름, "count": 문항 수}`.
    """
    if len(body.content) > UPLOAD_MAX:
        raise HTTPException(400, f"파일이 너무 큽니다 ({UPLOAD_MAX >> 20}MB 까지)")

    text = body.content.strip()
    try:
        if text.startswith("["):
            rows = json.loads(text)
        else:  # jsonl
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    except json.JSONDecodeError as error:
        raise HTTPException(400, f"JSON 을 못 읽었습니다: {error}")

    if not isinstance(rows, list) or not rows:
        raise HTTPException(400, "문항이 없습니다")
    missing = [i for i, row in enumerate(rows, 1)
               if not isinstance(row, dict) or not row.get("question")]
    if missing:
        raise HTTPException(400, f"question 이 없는 줄: {missing[:5]}")

    # **정답 문서가 코퍼스에 있는지 여기서 본다.** 안 보면 5분 뒤에 물러섬
    # 0.99 · 인용정확도 0.000 을 보고 "성능이 나쁘다" 고 읽게 된다. 그건
    # 성능이 아니라 발췌가 빈 것이다 — `retrieve(doc_ids=...)` 가 아무것도
    # 못 골라서 컨텍스트가 빈 문자열이 된다. 2026-09-03 에 두 세트가 그랬다.
    rows, report = evalrun.normalize(rows)
    if not report["matched"]:
        raise HTTPException(400, {
            "message": "정답 문서가 코퍼스에 하나도 없습니다. 이대로 돌리면 "
                       "발췌가 전부 비어 0점이 나옵니다. 세트에 evidence_text 가 "
                       "있다면 검색 없이 잴 수 있습니다 — "
                       "scripts/retrieval/contexts_from_evidence.py 를 보세요.",
            **report,
        })

    # 경로가 섞여 들어오면 `data/` 밖에 쓰게 된다. 이름은 우리가 만든다.
    safe = re.sub(r"[^0-9A-Za-z가-힣._-]", "_", Path(body.name).stem)[:40]
    stem = f"{UPLOAD_PREFIX}{datetime.now():%m%d-%H%M%S}_{safe}"
    (settings.DATA / f"{stem}.json").write_text(
        json.dumps(rows, ensure_ascii=False), encoding="utf-8"
    )
    return {"evalset": stem, "count": len(rows), **report}


@app.post("/eval")
def eval_start(body: Eval):
    """평가 한 바퀴를 백그라운드로 시작하고 작업번호를 돌려준다."""
    try:
        load_evalset(body.evalset)
    except Exception as error:  # noqa: BLE001
        raise HTTPException(400, f"평가 세트를 못 읽습니다: {error}")
    return {"job_id": evalrun.start(**body.model_dump())}


@app.get("/eval")
def eval_list():
    """최근 작업 목록. 로그는 뺀다."""
    return evalrun.listing()


@app.get("/eval/{job_id}")
def eval_status(job_id: str):
    """작업 하나의 지금 상태. UI 가 몇 초마다 부른다.

    화면을 떠났다 돌아와도 이것만 부르면 된다 — 상태가 메모리가 아니라
    파일에 있다.
    """
    job = evalrun.read(job_id)
    if job is None:
        raise HTTPException(404, "그런 작업이 없습니다")
    return job


@app.get("/health")
def health():
    """무엇이 떠 있고 무엇을 보고 있는지.

    **브라우저로 `https://<vercel주소>/api/health` 를 열면 배선 전체가 한 번에
    확인된다** — Vercel rewrite → VM:8010 → TEI·SGLang·인덱스. 어느 칸이
    비어 있는지로 어디가 끊겼는지 바로 안다. 크론이나 감시에도 그대로 쓴다.

    무거운 걸 새로 열지 않는다. 이미 뜬 것에 물어보기만 하므로 자주 불러도 된다.
    """
    import requests

    from config import retrieval as cfg
    from models.embed import TEI_EMBED_URL
    from models.rerank import TEI_RERANK_URL
    from models.sglang import current

    def tei(url):
        try:
            return requests.get(f"{url}/info", timeout=2).json().get("model_id")
        except Exception:  # noqa: BLE001 - 안 떠 있는 것도 상태다
            return None

    embed, rerank = tei(TEI_EMBED_URL), tei(TEI_RERANK_URL)
    generator = current()
    return {
        # 검색이 도는 데 반드시 필요한 것들. 하나라도 null 이면 /search 가 죽는다.
        "ok": bool(embed and rerank),
        "embedder": embed,
        "reranker": rerank,
        # 지금 GPU 에 올라와 있는 생성 모델. null 이면 첫 질문이 1~2분 걸린다.
        "generator": generator,
        # **어떤 코퍼스를 보고 있는지.** 배포 사고의 절반이 여기가 어긋난 것이다.
        "store": cfg.STORE,
        "index": cfg.index_name(),
        "chunks": cfg.chunk_name(),
    }
