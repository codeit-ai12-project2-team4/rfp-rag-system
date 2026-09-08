#!/usr/bin/env bash
# 임베더·리랭커를 바꿔가며 다시 잰다. **서비스는 안 건드린다.**
#
#     bash scripts/retrieval/sweep_models.sh
#
# 왜 다시 재나 — 8/31 의 임베더·리랭커 표에 **어느 평가 세트로 쟀는지가 안 적혀
# 있다.** 그때는 `--evalset` 기본값이 스크립트마다 달랐고(eval_qa / _80 / _both),
# 안 적었으면 무엇이 들어갔는지 지금은 모른다. 발표에 쓸 수 없다.
#
# 격리 — 네 겹이다. 하나라도 빠지면 배포 중인 서비스를 건드린다.
#
#   ① 크론 정지     실험 도중 청크가 늘면 판끼리 다른 코퍼스를 재게 된다
#   ② 청크 스냅샷   전용 이름으로 복사. 크론이 실수로 돌아도 무관하고,
#                   인덱스 이름이 청크 이름에서 나오므로 서비스 것과 안 겹친다
#   ③ STORE=faiss   LanceDB 테이블을 아예 안 연다. 서비스가 그걸 물고 있다
#   ④ 포트 8095/8096 실험용 TEI 를 따로 띄운다. 서비스의 8085/8086 은 그대로
#
# VRAM — 지금 15GB/24GB 를 쓰고 있다. 임베더든 리랭커든 568M fp16 은 약 1.5GB 라
# **하나씩만** 띄운다. 둘을 같이 올리지 않는다.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD

PY=.venv/bin/python
# **기본값을 안 둔다.** 안 주면 터진다 — 8/31 표가 세트를 안 적어 못 쓰게 됐고,
# 그걸 다시 재려고 만든 스크립트가 또 기본값으로 옛 세트를 물면 같은 일이 난다.
#
# 자동 관리 세트(`*.meta.json` 이 있는 것)를 쓰면 **청크가 자란 만큼 세트가 다시
# 만들어진다.** 설계된 동작이지만, 스윕 도중에 그러면 판끼리 다른 문항을 재게
# 된다. 도장이 없는 고정 세트를 쓰는 게 맞다 (예: 어댑터가 만든 eval_160_ours).
: "${EVALSET:?EVALSET 를 지정하세요 — 예: EVALSET=eval_160_ours bash $0}"
LIVE=${LIVE:-chunks_cleaned_documents__pipeline}
SNAP=sweep_$(date +%m%d)                  # 실험 전용 청크 이름
OUT=outputs/eval_results/sweep_$(date +%m%d)
IMAGE=ghcr.io/huggingface/text-embeddings-inference:89-1.9

# **모델마다 접두어가 다르다. 이걸 안 맞추면 모델이 아니라 배선을 재게 된다.**
# 9/2 에 `.env` 의 변수 이름이 코드와 어긋나 **문서에만 접두어가 붙고 질의엔 안
# 붙은** 적이 있다. 질의와 문서가 다른 공간에 놓여서 gemma 를 실제보다 나쁘게
# 기록할 뻔했다. 그래서 판마다 여기서 명시하고, 실행할 때 화면에 찍는다.
#
#   문서 접두어는 **인덱싱 시점**에 들어간다 → 바꾸면 인덱스를 다시 지어야 한다
#   질의 접두어는 검색할 때 붙는다          → 재인덱싱 불필요
#
# 형식: 모델|질의접두어|문서접두어
EMBEDDERS=(
  "dragonkue/snowflake-arctic-embed-l-v2.0-ko|query: |"          # 현재 채택
  "dragonkue/BGE-m3-ko||"                                        # 접두어 없음
  "nlpai-lab/KURE-v2||"                                          # BGE-M3 계열 → 없음
  "Qwen/Qwen3-Embedding-0.6B|@QWEN@|"                            # 질의에만 지시문
)
# Qwen3 는 지시문을 질의에만 붙인다. 안 붙이면 재현율이 크게 떨어진다
# (8/31 실측: 의역 0.469 → 0.502). 줄바꿈이 들어가야 해서 따로 둔다.
QWEN_INSTRUCT=$'Instruct: Given a search query, retrieve relevant passages that answer the query\nQuery: '
RERANKERS=(
  "dragonkue/bge-reranker-v2-m3-ko"              # 현재 채택
  "BAAI/bge-reranker-v2-m3"
)

mkdir -p "$OUT"
echo "결과 → $OUT"

# ── ① 크론 정지 ──
#
# **살아있는 크론탭을 그대로 백업하면 안 된다.** 앞판이 트랩을 못 타고 죽었으면
# (Ctrl-C 두 번, kill -9, 세션 끊김) 크론은 **이미 꺼져 있다.** 그 상태에서
# `crontab -l` 을 받으면 빈 파일이 백업으로 덮이고, 트랩이 그 빈 걸 "복원"해서
# 크론이 영영 사라진다. 백업 파일 하나에 서비스 일정을 맡기는 구조 자체가 틀렸다.
#
# 정본은 저장소 안에 있다 — `docker/crontab.txt`. 설치도 원래 이 파일로 한다
# (`crontab docker/crontab.txt`). 백업이 비면 여기서 되살린다.
BAK=/tmp/crontab.sweep.bak
crontab -l > "$BAK" 2>/dev/null || true
if [ ! -s "$BAK" ]; then
    echo "⚠ 크론이 이미 꺼져 있다 — 앞판이 비정상 종료했을 수 있다. 정본에서 복원한다."
    cp "$ROOT/docker/crontab.txt" "$BAK"
fi
crontab -r 2>/dev/null || true
echo "크론 정지. 복원본 $BAK ($(grep -c "^[^#]*refresh\.sh" "$BAK") 줄이 refresh.sh)"

cleanup() {
    echo
    echo "정리 중…"
    docker rm -f tei-sweep >/dev/null 2>&1 || true
    # 복원은 **확인까지 해야 복원이다.** 명령이 0을 뱉고도 빈 크론탭이 앉는 경우가 있다.
    crontab "$BAK" 2>/dev/null || true
    if [ "$(crontab -l 2>/dev/null | grep -c "^[^#]*refresh\.sh")" -ge 1 ]; then
        echo "크론 복원됨 — refresh.sh $(crontab -l | grep -c "^[^#]*refresh\.sh") 줄"
    else
        echo "⚠ 크론 복원 실패. 손으로: crontab $ROOT/docker/crontab.txt"
    fi
}
trap cleanup EXIT

# ── ② 청크 스냅샷 ──
cp "outputs/chunks/$LIVE.jsonl" "outputs/chunks/$SNAP.jsonl"
echo "청크 스냅샷 $SNAP ($(wc -l < "outputs/chunks/$SNAP.jsonl") 줄)"

# 실험용 TEI 하나를 띄운다. 뜰 때까지 기다린다.
start_tei() {   # $1=모델 $2=포트 $3=추가인자
    docker rm -f tei-sweep >/dev/null 2>&1 || true
    docker run -d --name tei-sweep --gpus all -p "$2:80" \
        -v "$HOME/.cache/huggingface:/data" "$IMAGE" \
        --model-id "$1" ${3:-} >/dev/null
    for _ in $(seq 1 90); do
        curl -fsS "http://localhost:$2/info" >/dev/null 2>&1 && { echo "  TEI $1 준비됨"; return 0; }
        sleep 2
    done
    echo "  ⚠ TEI 가 안 뜬다: $1"; docker logs --tail 20 tei-sweep; return 1
}

run_compare() {  # $1=꼬리표
    STORE=faiss $PY scripts/retrieval/compare_retrieval.py \
        --chunks "$SNAP" --evalset "$EVALSET" --scoped \
        --embed tei --rerank tei \
        --out "$OUT/$1.csv"
}

echo
echo "════════ 임베더 ════════"
for SPEC in "${EMBEDDERS[@]}"; do
    IFS='|' read -r MODEL QPFX DPFX <<< "$SPEC"
    [ "$QPFX" = "@QWEN@" ] && QPFX="$QWEN_INSTRUCT"
    TAG=$(basename "$MODEL")
    echo; echo "── $MODEL"
    printf '   질의접두어 %q\n   문서접두어 %q\n' "$QPFX" "$DPFX"
    start_tei "$MODEL" 8095 || continue
    # 판마다 인덱스를 새로 짓는다. 이름이 임베더까지 물고 있어 서로 안 덮는다.
    # **문서 접두어가 여기서 들어간다.** 색인과 측정에 같은 값을 써야 한다.
    export TEI_EMBED_URL=http://localhost:8095 EMBED_MODEL="$MODEL" \
           EMBED_QUERY_PREFIX="$QPFX" EMBED_DOC_PREFIX="$DPFX"
    STORE=faiss $PY src/vectorstore.py --chunks "$SNAP" --force \
        || { echo "  색인 실패"; continue; }
    run_compare "embed_$TAG" || echo "  측정 실패"
done
unset TEI_EMBED_URL EMBED_MODEL EMBED_QUERY_PREFIX EMBED_DOC_PREFIX

echo
echo "════════ 리랭커 (임베더는 채택본으로 고정) ════════"
IFS='|' read -r BASE_MODEL BASE_Q BASE_D <<< "${EMBEDDERS[0]}"
export TEI_EMBED_URL=http://localhost:8095 EMBED_MODEL="$BASE_MODEL" \
       EMBED_QUERY_PREFIX="$BASE_Q" EMBED_DOC_PREFIX="$BASE_D"
start_tei "$BASE_MODEL" 8095
STORE=faiss $PY src/vectorstore.py --chunks "$SNAP" --force
docker rm -f tei-sweep >/dev/null 2>&1 || true   # 임베딩 끝. VRAM 을 비우고 리랭커를 올린다

# 인덱스는 위에서 만든 것 하나로 고정한다. **리랭커만 바뀐다** — 임베딩을 다시
# 하면 변수가 둘이 되어 무엇이 성적을 움직였는지 못 가른다.
for MODEL in "${RERANKERS[@]}"; do
    TAG=$(basename "$MODEL")
    echo; echo "── $MODEL"
    start_tei "$MODEL" 8096 "--auto-truncate" || continue
    STORE=faiss TEI_RERANK_URL=http://localhost:8096 RERANK_MODEL="$MODEL" \
        run_compare "rerank_$TAG" || echo "  측정 실패"
done

echo
echo "════════ 표 ════════"
ls "$OUT"/*.csv
for A in "$OUT"/embed_*.csv; do
    [ "$A" = "$OUT/embed_$(basename "${EMBEDDERS[0]}").csv" ] && continue
    echo; echo "── $(basename "$A") vs 채택 임베더"
    $PY scripts/retrieval/compare_runs.py "$A" "$OUT/embed_$(basename "${EMBEDDERS[0]}").csv" || true
done
echo; echo "── 리랭커"
$PY scripts/retrieval/compare_runs.py "$OUT/rerank_$(basename "${RERANKERS[1]}").csv" \
    "$OUT/rerank_$(basename "${RERANKERS[0]}").csv" || true

echo
echo "세트 $EVALSET · 청크 $SNAP · STORE=faiss"
echo "임베더는 판마다 접두어를 따로 걸었다 (위 로그에 찍혀 있다)"
echo "**이 두 줄을 표와 같이 기록한다.** 8/31 에 안 적어서 다시 재고 있다."
