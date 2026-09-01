"""웹앱이 부르는 HTTP 경계. 엔드포인트 두 개가 전부다.

프론트(Next.js)는 별도 repo 다. 파이썬 소스를 볼 이유가 없고 배포 주기도 다르다.
둘을 잇는 건 이 파일뿐이라, 여기만 안 바뀌면 양쪽이 따로 움직인다.

    uvicorn src.api:app --reload --port 8088

    POST /search   자연어로 공고 찾기 (1단계 — 사람이 목록에서 고르는 화면)
    POST /ask      고른 공고 안에서 질문 (2단계 — 발췌 → 답변 → 출처)
    GET  /models   드롭다운에 채울 모델 목록

**시나리오 A/B 는 여기 없다.** 그건 우리 인프라 선택이지 고객의 선택이 아니고,
B 는 원본 RFP 를 외부 API 로 내보낸다. 환경변수로 배포 시점에 정한다.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import MODEL_CONFIGS
from generation import generate_answer
from retriever import build_context, retrieve, search_notices, sources

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
    """드롭다운에 채울 모델 목록."""
    return [{"key": key, "name": cfg.model} for key, cfg in MODEL_CONFIGS.items()]


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
