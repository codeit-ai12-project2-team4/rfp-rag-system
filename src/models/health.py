"""지금 무엇이 떠 있는지 한눈에. 노트북 맨 앞에서 부르면 편하다."""

import os

import requests

from models.embed import TEI_EMBED_URL
from models.llm import VLLM_URL
from models.rerank import TEI_RERANK_URL


def check_servers():
    """지금 무엇이 떠 있는지 한눈에. 노트북 맨 앞에서 부르면 편하다."""
    print("=" * 60)
    for label, url in [
        ("임베딩 (TEI)", TEI_EMBED_URL),
        ("리랭커 (TEI)", TEI_RERANK_URL),
    ]:
        try:
            info = requests.get(f"{url}/info", timeout=3).json()
            print(f"  O  {label:<14} {url}  →  {info.get('model_id')}")
        except Exception:
            print(f"  X  {label:<14} {url}  →  응답 없음")

    try:
        models = requests.get(f"{VLLM_URL}/models", timeout=3).json()
        served = [m["id"] for m in models.get("data", [])]
        print(f"  O  {'생성 (vLLM)':<14} {VLLM_URL}  →  {served}")
    except Exception:
        print(f"  X  {'생성 (vLLM)':<14} {VLLM_URL}  →  응답 없음")

    key = os.environ.get("OPENAI_API_KEY")
    print(f"  {'O' if key else 'X'}  {'OpenAI 키':<14} {'설정됨' if key else '없음'}")

    from pieces.search import has_kiwi

    print(
        f"  {'O' if has_kiwi() else 'X'}  {'형태소 분석기':<13} "
        f"{'kiwipiepy 동작' if has_kiwi() else 'kiwipiepy 없음 → BM25 성능 떨어짐'}"
    )

    # 메모리가 바닥나면 디스크와 달리 VM 이 통째로 멈춘다. 먼저 볼 것.
    from resources import DANGER_GB, free_disk_gb, memory, process_memory_gb

    stat = memory()
    if stat:
        mark = "X" if stat["available_gb"] < DANGER_GB else "O"
        mine = process_memory_gb()
        tail = f" · 이 커널 {mine}GB" if mine is not None else ""
        print(
            f"  {mark}  {'메모리':<15} 전체 {stat['total_gb']}GB · "
            f"남음 {stat['available_gb']}GB{tail}"
        )
        if mark == "X":
            print(
                "      ⚠ 이대로 큰 셀을 돌리면 VM 이 멈추고 SSH 가 끊깁니다. "
                "커널부터 재시작하세요."
            )
    print(f"  O  {'디스크':<15} 남음 {free_disk_gb():.1f}GB")
    print("=" * 60)
