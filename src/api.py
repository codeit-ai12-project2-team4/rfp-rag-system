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

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import MODEL_CONFIGS
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


class Search(BaseModel):
    query: str
    top_n: int = 10
    min_budget: int | None = None
    max_budget: int | None = None
    agency: str | None = None


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
