"""E2E 평가를 백그라운드로 돌리고 진행 상황을 파일에 적는다.

UI 에서 "돌려" 를 누르면 몇 분에서 십수 분이 걸린다. 그 사이 사용자는 다른
화면으로 가고, 새로고침도 하고, 노트북도 닫는다. 그래서 **상태를 메모리에
두지 않는다.** 작업 하나가 파일 하나다.

    outputs/eval_runs/<작업번호>.json

메모리에 두면 uvicorn 이 재시작되는 순간(크론이 새벽마다 한다) 결과가 사라진다.
파일이면 재시작 뒤에도 지난 결과를 그대로 본다. 조회 쪽은 이 파일만 읽으므로
작업을 돌린 프로세스가 죽어 있어도 상관없다.

세 단계를 그대로 이어 붙인다. 이미 있는 것들이다.

    1. 발췌 뽑기   retriever.export_contexts   **이 프로세스에서** 돈다.
                                              색인·BM25 가 이미 올라와 있다.
    2. 답변 생성   scripts/retrieval/answer.py        따로 띄운다
    3. 채점       scripts/retrieval/score_answers.py 따로 띄운다

2·3 을 따로 띄우는 이유는 색인이 필요 없기 때문이다. 같은 프로세스에서 돌리면
LLM 을 기다리는 동안 API 워커 하나가 통째로 묶인다. 반대로 1 을 따로 띄우면
BM25 색인을 처음부터 다시 지어(2분 30초) 메모리에 한 벌 더 얹는다 — VM 이 16GB 다.
"""

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from config import settings

UPLOAD_PREFIX = "upload_"  # api.py 와 같은 값. 여기서 지울지 판정한다
RUNS = settings.OUTPUTS / "eval_runs"
LOG_LINES = 400  # 이보다 오래된 줄은 버린다. 파일이 무한정 자라면 읽는 쪽이 느려진다
PYTHON = sys.executable
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "retrieval"

# 단가는 `config/model_config.py` 의 `usd_per_call` 하나에서만 온다.
# VM 모델(sglang)은 0 이다 — 우리 GPU 를 쓴다.


def _path(job_id):
    return RUNS / f"{job_id}.json"


def _write(job):
    RUNS.mkdir(parents=True, exist_ok=True)
    tmp = _path(job["id"]).with_suffix(".tmp")
    tmp.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(_path(job["id"]))  # 반쯤 쓰인 파일을 UI 가 읽는 일이 없게


def read(job_id):
    """작업 하나의 지금 상태. 없으면 None."""
    path = _path(job_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:  # 쓰는 중일 수 있다. 다음 폴링에서 읽힌다
        return None


def listing(limit=20):
    """최근 작업 목록. 로그는 빼고 준다 — 목록 화면에는 필요 없다."""
    rows = []
    for path in sorted(RUNS.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        job = read(path.stem)
        if job:
            rows.append({k: v for k, v in job.items() if k != "log"})
    return rows[:limit]


def estimate(count, model="mini", judge=True, judge_model="nano"):
    """이 조합으로 돌리면 대략 얼마인가. 달러.

    UI 도 같은 값을 보여줘야 하므로 단가를 여기 적지 않는다 —
    `MODEL_CONFIGS[key].usd_per_call` 이 유일한 출처다.

    Args:
        count: 문항 수.
        model: 답변 모델 키.
        judge: 충실성 채점까지 할지.
        judge_model: 채점 모델 키.

    Returns:
        float: 달러. VM 모델만 쓰면 0.0 이다.
    """
    from config import MODEL_CONFIGS

    def per(key):
        cfg = MODEL_CONFIGS.get(key)
        return cfg.usd_per_call if cfg else 0.0

    return round(count * (per(model) + (per(judge_model) if judge else 0.0)), 4)


@lru_cache(maxsize=1)
def _corpus():
    """코퍼스의 doc_id 와, 파일명 → doc_id 표.

    평가 세트의 `doc_id` 가 우리 것과 다르면 `retrieve(doc_ids=...)` 가 청크를
    **하나도** 못 고른다. 그러면 발췌가 빈 문자열이 되고, 모델은 답할 게 없어
    전부 "확인되지 않습니다" 를 낸다. 지표는 물러섬 0.99 · 인용정확도 0.000 으로
    찍히는데, 그건 성능이 아니라 **입력이 비었다는 신호다.**
    """
    import chunking

    from config import retrieval as cfg

    ids, by_file = set(), {}
    for chunk in chunking.load_chunks(cfg.chunk_name()):
        doc_id = str(chunk.metadata.get("doc_id") or "")
        if not doc_id:
            continue
        ids.add(doc_id)
        name = str(chunk.metadata.get("file_name") or "")
        if name:
            by_file.setdefault(name, doc_id)
            by_file.setdefault(Path(name).stem, doc_id)
    return ids, by_file


def normalize(rows):
    """업로드한 평가 세트를 우리 형식으로 맞추고 정답 문서를 대조한다.

    받아 주는 차이는 셋이다. 팀마다 만든 세트의 필드 이름이 다르다.

        question_type → type      (`없음` 이면 answerable=False 로도 쓴다)
        evidence_text → keywords
        doc_id 가 파일명이면 → 코퍼스의 doc_id (`공고번호-차수`)

    Args:
        rows: 업로드한 줄들.

    Returns:
        (고친 줄들, 보고 dict). 보고는 `{total, matched, converted, unknown}`.
    """
    ids, by_file = _corpus()
    out, converted, unknown = [], 0, []
    for row in rows:
        row = dict(row)
        if "type" not in row and "question_type" in row:
            row["type"] = row["question_type"]
        if row.get("type") == "없음":
            row.setdefault("answerable", False)
        if "keywords" not in row and row.get("evidence_text"):
            row["keywords"] = [row["evidence_text"]]

        doc_id = str(row.get("doc_id") or "")
        if doc_id and doc_id not in ids:
            found = by_file.get(doc_id) or by_file.get(Path(doc_id).stem)
            if found:
                row["doc_id"] = found
                converted += 1
            else:
                unknown.append(doc_id)
        out.append(row)

    return out, {
        "total": len(rows),
        "matched": len(rows) - len(unknown),
        "converted": converted,
        "unknown": sorted(set(unknown))[:10],
        "unknown_count": len(set(unknown)),
    }


def start(evalset, model="mini", judge=True, judge_model="nano", limit=None,
          generation=True):
    """작업을 만들고 스레드에서 돌린다. 바로 작업번호를 돌려준다.

    Args:
        evalset: `data/` 의 평가 세트 이름.
        model: 답변 생성 모델 키 (config.MODEL_CONFIGS).
        judge: 충실성까지 잴지. LLM 호출이 문항 수만큼 더 든다.
        judge_model: 채점 모델 키.
        limit: 앞에서 몇 문항만. 연습용.
        generation: 생성용 본문을 쓸지 (9/8 결정은 True).

    Returns:
        str: 작업번호.
    """
    job_id = f"{datetime.now():%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"
    job = {
        "id": job_id,
        "status": "running",
        "step": "준비",
        "done": 0,
        "total": 0,
        "started_at": time.time(),
        "finished_at": None,
        "options": {
            "evalset": evalset, "model": model, "judge": judge,
            "judge_model": judge_model, "limit": limit, "generation": generation,
        },
        "files": {
            "contexts": f"contexts_eval_{job_id}.jsonl",
            "answers": f"answers_eval_{job_id}.jsonl",
            "metrics": f"metrics_eval_{job_id}.json",
        },
        "log": [],
        "metrics": None,
        "error": None,
    }
    _write(job)
    threading.Thread(target=_run, args=(job,), daemon=True).start()
    return job_id


def _log(job, line):
    job["log"] = (job["log"] + [line])[-LOG_LINES:]
    _write(job)


def _stream(job, args, step):
    """스크립트 하나를 띄우고 출력을 로그에 붙인다. 실패하면 예외."""
    _log(job, f"$ {' '.join(str(a) for a in args[1:])}")
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    process = subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", env=env,
        cwd=str(SCRIPTS.parents[1]),
    )
    last = 0.0
    for line in process.stdout:
        line = line.rstrip()
        if not line:
            continue
        job["log"] = (job["log"] + [line])[-LOG_LINES:]
        # 매 줄마다 파일을 쓰면 디스크만 때린다. 0.5초에 한 번이면 화면은 충분하다.
        if time.time() - last > 0.5:
            _write(job)
            last = time.time()
    process.wait()
    _write(job)
    if process.returncode != 0:
        raise RuntimeError(f"{step} 가 코드 {process.returncode} 로 끝났습니다")


def _run(job):
    options = job["options"]
    stem = f"eval_{job['id']}"
    contexts = settings.EVAL_RESULTS / f"contexts_{stem}.jsonl"
    answers = settings.EVAL_RESULTS / f"answers_{stem}.jsonl"
    metrics = settings.EVAL_RESULTS / f"metrics_{stem}.json"

    try:
        # 1. 발췌 — 이 프로세스에서. 색인이 이미 올라와 있다.
        from retriever import export_contexts

        job["step"] = "발췌 뽑기"
        _log(job, f"평가 세트 {options['evalset']} · 발췌를 뽑습니다")

        last = [0.0]

        def progress(done, total):
            job["done"], job["total"] = done, total
            if time.time() - last[0] > 0.5:
                _write(job)
                last[0] = time.time()

        export_contexts(
            options["evalset"], contexts,
            generation=options["generation"], on_progress=progress,
        )
        _log(job, f"발췌 {job['total']}문항 완료")

        # 2. 답변
        job["step"] = "답변 생성"
        job["done"] = 0
        _write(job)
        args = [PYTHON, str(SCRIPTS / "answer.py"), str(contexts),
                "--model", options["model"], "--out", str(answers)]
        if options["limit"]:
            args += ["--limit", str(options["limit"])]
        _stream(job, args, "답변 생성")

        # 3. 채점
        job["step"] = "채점"
        _write(job)
        args = [PYTHON, str(SCRIPTS / "score_answers.py"), str(answers),
                "--json", str(metrics)]
        if options["judge"]:
            args += ["--judge", "--model", options["judge_model"]]
        _stream(job, args, "채점")

        job["metrics"] = json.loads(metrics.read_text(encoding="utf-8"))
        job["status"] = "done"
        job["step"] = "끝"
    except Exception as error:  # noqa: BLE001
        job["status"] = "failed"
        job["error"] = f"{type(error).__name__}: {error}"
        _log(job, f"X {job['error']}")
    finally:
        dropped = _drop_upload(options["evalset"])
        if dropped:
            _log(job, f"업로드본 {dropped} 을 지웠습니다")
        job["finished_at"] = time.time()
        _write(job)


def _drop_upload(evalset):
    """업로드해서 쓴 평가 세트를 지운다. 채점이 끝나면(실패해도) 부른다.

    **접두어로만 판정한다.** 이름을 그대로 믿고 지우면 `eval_qa_both` 같은
    진짜 세트를 날릴 수 있다. 경로도 다시 확인한다 — `../` 가 섞여 들어오면
    `data/` 밖을 지우게 된다.
    """
    if not str(evalset).startswith(UPLOAD_PREFIX):
        return
    path = (settings.DATA / f"{evalset}.json").resolve()
    if path.parent == settings.DATA.resolve() and path.exists():
        path.unlink()
        return path.name
    return None


def sweep():
    """서버가 뜰 때 한 번. 돌던 중에 죽은 작업을 표시한다.

    안 하면 재시작 전에 돌던 작업이 영원히 `running` 으로 남아, UI 가 끝나지
    않는 진행률을 계속 보여 준다.
    """
    for path in RUNS.glob("*.json"):
        job = read(path.stem)
        if job and job.get("status") == "running":
            job["status"] = "interrupted"
            job["error"] = "서버가 재시작되어 중단됐습니다"
            job["finished_at"] = time.time()
            _write(job)
