"""SGLang 생성 서버 — **한 번에 한 모델만** 올린다.

L4 24GB 에 TEI 셋(임베딩·SPLADE·리랭커, 약 4GB)이 이미 상주해서 생성용으로
남는 건 20GB 다. 8B 를 fp16 으로 받으면 그것만으로 16GB 라 여러 모델을 같이
못 띄운다. 그래서 요청이 온 모델이 안 떠 있으면 컨테이너를 갈아끼운다.

    from models import sglang
    sglang.ensure("kakaocorp/kanana-nano-2.1b-instruct", mem="0.35")

갈아끼우는 데 30초~2분, 처음 받는 모델은 더 걸린다. 그 사이 요청은 기다린다.
UI 에 "모델 준비 중" 을 띄우려면 `current()` 를 보면 된다.

**vLLM 을 안 쓰고 SGLang 인 이유** — RadixAttention 이 프롬프트 앞부분을
KV 캐시로 돌려쓴다. 우리는 system 프롬프트가 고정이고 같은 공고에 여러 질문을
하므로 앞부분이 자주 겹친다.
"""

import os
import subprocess
import threading
import time
from pathlib import Path

import requests

_ROOT = Path(__file__).resolve().parents[2]

SGLANG_URL = os.environ.get("SGLANG_URL", "http://localhost:8087")
COMPOSE = os.environ.get("SGLANG_COMPOSE", str(_ROOT / "docker" / "docker-compose.yml"))
SERVICE = os.environ.get("SGLANG_SERVICE", "gen")
CONTAINER = os.environ.get("SGLANG_CONTAINER", "bidmate-gen")
# 처음 받는 8B 는 내려받기만 몇 분이다. 넉넉히 두고, 넘으면 에러 메시지로 알린다.
SWAP_TIMEOUT = int(os.environ.get("SGLANG_SWAP_TIMEOUT", "900"))

# ponytail: 전역 락. 워커가 하나라 이걸로 충분하다. 워커를 늘리면
# 프로세스 간 락(파일 락)이나 별도 관리 프로세스로 올려야 한다.
_swap_lock = threading.Lock()


def _restart_count():
    """컨테이너가 몇 번 재시작했는지. 못 읽으면 0.

    `restart: unless-stopped` 라 launch 가 실패하면 **조용히 무한 재시도한다.**
    그러면 `current()` 는 영원히 None 이고 호출자는 SWAP_TIMEOUT 을 꽉 채운다.
    실제로 kanana-nano-2.1b 가 config 검증에서 죽어 이걸 겪었다 — 15분을 기다린
    끝에 "안 떴습니다" 를 보게 된다. 크래시는 기다릴 이유가 없으니 바로 나간다.

    `--force-recreate` 는 컨테이너를 새로 만들어서 이 값이 0 부터 시작한다.
    """
    try:
        done = subprocess.run(
            ["docker", "inspect", "-f", "{{.RestartCount}}", CONTAINER],
            capture_output=True, text=True, timeout=5,
        )
        return int(done.stdout.strip()) if done.returncode == 0 else 0
    except Exception:  # noqa: BLE001 - 상태 확인이 교체를 막으면 안 된다
        return 0


def _tail_logs(lines=30):
    """실패했을 때 에러 메시지에 붙일 로그 꼬리. 보러 가지 않아도 되게."""
    try:
        done = subprocess.run(
            ["docker", "compose", "-f", COMPOSE, "logs", "--tail", str(lines), SERVICE],
            capture_output=True, text=True, timeout=10,
        )
        return done.stdout.strip() or "(로그 없음)"
    except Exception as e:  # noqa: BLE001
        return f"(로그를 못 읽었습니다: {e})"


def current(timeout=1.0):
    """지금 올라와 있는 모델의 repo id. 서버가 없거나 뜨는 중이면 None.

    상태를 파일에 안 적는 이유 — 적어 두면 컨테이너가 죽었을 때 거짓말을 한다.
    서버한테 직접 물어보는 게 항상 맞다.

    Args:
        timeout: 초. 화면 그리는 길목에서도 부르므로 짧게 잡는다.

    Returns:
        str | None: repo id.
    """
    try:
        response = requests.get(f"{SGLANG_URL}/v1/models", timeout=timeout)
        return response.json()["data"][0]["id"] if response.ok else None
    except Exception:  # noqa: BLE001 - 안 떠 있는 것도 정상 상태다
        return None


def ensure(repo, mem="0.45", args=""):
    """이 모델이 올라와 있게 만든다. 이미 그 모델이면 아무것도 안 한다.

    Args:
        repo: HuggingFace repo id. SGLang 이 이 값을 그대로 서빙 이름으로 쓴다.
        mem: `--mem-fraction-static`. **GPU 전체 대비 비율**이다(남은 양이 아니다).
            TEI 가 4GB 를 쓰고 있으니 0.83 을 넘기면 위험하다.
        args: 그 모델에만 필요한 추가 인자. 예) 채팅 템플릿 없는 베이스 모델.

    Raises:
        RuntimeError: SWAP_TIMEOUT 안에 안 뜰 때.
        subprocess.CalledProcessError: docker compose 자체가 실패할 때.
    """
    if current() == repo:
        return

    with _swap_lock:
        if current() == repo:  # 락 기다리는 동안 다른 요청이 이미 올렸을 수 있다
            return
        print(f"[sglang] 모델 교체: {current()} -> {repo} (mem={mem})")
        subprocess.run(
            ["docker", "compose", "-f", COMPOSE, "up", "-d", "--force-recreate", SERVICE],
            env={**os.environ, "GEN_MODEL": repo, "GEN_MEM": str(mem), "GEN_ARGS": args},
            check=True,
            capture_output=True,
        )
        started = time.time()
        while time.time() - started < SWAP_TIMEOUT:
            if current() == repo:
                print(f"[sglang] {repo} 준비 완료 ({time.time() - started:.0f}초)")
                return
            if _restart_count() > 0:
                raise RuntimeError(
                    f"{repo} 를 띄우다 죽었습니다 (컨테이너 재시작 루프).\n"
                    f"모델을 못 받은 게 아니라 서버가 실행에 실패한 것이라 기다려도 안 됩니다.\n\n"
                    f"{_tail_logs()}"
                )
            time.sleep(3)

    raise RuntimeError(
        f"{repo} 가 {SWAP_TIMEOUT}초 안에 안 떴습니다. 아직 내려받는 중일 수 있습니다.\n"
        f"  docker compose -f {COMPOSE} logs --tail=50 {SERVICE}\n\n"
        f"{_tail_logs(10)}"
    )
