"""메모리·디스크가 얼마나 남았는지 보고, 모자라면 미리 막는다.

VM 이 멈추고 SSH 가 끊기는 사고는 대부분 디스크가 아니라 **메모리** 때문이다.
리눅스는 메모리가 바닥나면 OOM killer 를 부르는데, 이때 하필 sshd 나
JupyterHub 가 죽으면 접속 자체가 안 된다. 팀 공용 VM 이면 남의 커널까지
같이 날아간다.

그래서 비싼 걸 만들기 전에 먼저 물어본다.

    from resources import show_memory, need_memory

    show_memory()               # 지금 얼마 남았나
    need_memory(2.0)            # 2GB 가 필요한데 없으면 여기서 멈춘다
"""

import os
import shutil

import psutil

# 이 값보다 남은 메모리가 적으면 위험하다고 본다 (GB)
DANGER_GB = 1.5


def memory():
    """전체 / 쓰는 중 / 남은 메모리를 GB 로.

    `available` 이 실제로 쓸 수 있는 양이다. `free` 는 캐시를 빼고 세기 때문에
    항상 작게 나오는데, 캐시는 필요하면 커널이 알아서 비운다. available 을 볼 것.

    `/proc/meminfo` 를 직접 읽다가 맥에서 `FileNotFoundError` 로 터졌다.
    psutil 은 이미 requirements 에 있고 맥·리눅스를 다 본다.

    Returns:
        총량·가용·사용·스왑을 GB 로 담은 dict. 못 읽으면 빈 dict.
    """
    try:
        virtual = psutil.virtual_memory()
        swap = psutil.swap_memory()
    except OSError:
        return {}
    return {
        "total_gb": round(virtual.total / 1024**3, 1),
        "available_gb": round(virtual.available / 1024**3, 1),
        "used_gb": round((virtual.total - virtual.available) / 1024**3, 1),
        "swap_gb": round(swap.total / 1024**3, 1),
        "swap_used_gb": round(swap.used / 1024**3, 1),
    }


def process_memory_gb():
    """이 파이썬 프로세스(=커널) 하나가 실제로 붙잡고 있는 물리 메모리.

    Returns:
        GB. 못 읽으면 None.
    """
    try:
        return round(psutil.Process(os.getpid()).memory_info().rss / 1024**3, 2)
    except (psutil.Error, OSError):
        return None


def free_disk_gb(path="/"):
    return shutil.disk_usage(path).free / 1024**3


def show_memory(label=""):
    """한 줄로 찍는다. 무거운 셀 앞뒤에 넣어 두면 어디서 늘었는지 보인다."""
    stat = memory()
    if not stat:
        print("메모리 정보를 읽을 수 없습니다")
        return stat

    mine = process_memory_gb()
    head = f"[{label}] " if label else ""
    line = (
        f"{head}메모리  전체 {stat['total_gb']}GB · "
        f"쓰는 중 {stat['used_gb']}GB · 남음 {stat['available_gb']}GB"
    )
    if mine is not None:
        line += f"  |  이 커널 {mine}GB"
    print(line)

    if stat["available_gb"] < DANGER_GB:
        print("  ⚠ 위험합니다. 여기서 더 쓰면 VM 이 멈추고 SSH 가 끊길 수 있습니다.")
        print("    커널을 재시작(Kernel → Restart)해서 메모리를 비우세요.")
    print(f"디스크  남음 {free_disk_gb():.1f}GB")
    return stat


def need_memory(gb, what=""):
    """이만큼 필요한데 없으면 **미리** 멈춘다.

    VM 이 죽고 나서 아는 것보다, 셀 하나가 에러로 끝나는 게 훨씬 낫다.
    """
    stat = memory()
    if not stat:
        return True
    available = stat["available_gb"]
    if available >= gb:
        return True

    what = f"'{what}' 은(는) " if what else ""
    raise MemoryError(
        f"{what}약 {gb}GB 가 필요한데 {available}GB 밖에 안 남았습니다.\n"
        f"  전체 {stat['total_gb']}GB 중 {stat['used_gb']}GB 사용 중.\n"
        "  할 수 있는 것:\n"
        "   1) 커널 재시작 (Kernel → Restart Kernel) — 제일 빠르다\n"
        "   2) 노트북 위쪽의 MAX_DOCS 를 줄여 문서 일부로만 돌린다\n"
        "   3) 안 쓰는 변수를 del 하고 import gc; gc.collect()\n"
        "  자세한 건 docs/디스크_관리.md 의 '메모리' 절."
    )


def limit_memory(gb):
    """이 커널이 쓸 수 있는 메모리에 천장을 씌운다.

    천장을 넘으면 **커널만** MemoryError 로 죽는다. VM 전체가 멈추고 SSH 가
    끊기는 것보다 백 배 낫다. 팀 공용 VM 이면 켜 두는 게 예의다.

        from resources import limit_memory
        limit_memory(8)          # 이 커널은 8GB 까지만

    주의 — torch/CUDA 를 쓸 거면 켜지 말 것. CUDA 는 실제로 안 쓰는
    가상 주소를 수십 GB 씩 잡아 두는데 이 천장은 가상 주소 기준이라
    엉뚱한 곳에서 죽는다. TEI + OpenAI 구성(기본)에서는 문제없다.
    """
    import resource

    limit = int(gb * 1024**3)
    _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    print(
        f"이 커널의 메모리 상한을 {gb}GB 로 걸었습니다. "
        "넘으면 VM 이 아니라 커널만 죽습니다."
    )
