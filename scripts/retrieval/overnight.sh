#!/usr/bin/env bash
# pool 과 BM25 가중치를 한 번에 쓸어 담는다. 자는 동안 돌리는 용도다.
#
# 왜 지금 다시 재나 — 지난주에 "pool 도 가중치도 차이 없음" 으로 닫았는데,
# 그건 평가 세트가 쉬워서 리랭커가 상류를 전부 덮어쓰고 있었기 때문이다.
# 세트를 어렵게 만드니 설정 간 차이가 다시 보인다. 그러면 이 둘도 다시 재야 한다.
#
#     bash scripts/retrieval/overnight.sh
#
# pool 이 커지면 리랭커가 그만큼 더 많은 후보를 채점하므로 시간이 거의 비례한다.
# 30/50/80/120 이면 대략 10 + 17 + 27 + 40 = 95분쯤 걸린다.

set -u

CHUNKS=cleaned_documents_v3__recursive_1200_200
EVALSET=eval_qa_merged
POOLS="30 50 80 120"
WEIGHTS="0.3,0.5,0.7,0.9"
OUT=outputs/eval_results
LOG=$OUT/pool_sweep.log

cd "$(dirname "$0")/../.." || exit 1
mkdir -p "$OUT"

# --- 먼저 확인한다. 90분 돌린 뒤에 틀린 걸 아는 것보다 30초가 낫다 -----------
echo "== 사전 점검 =="

MODEL=$(curl -s -m 5 localhost:8085/info | python -c \
  "import json,sys; print(json.load(sys.stdin).get('model_id','(응답없음)'))" 2>/dev/null)
echo "  임베더  $MODEL"
case "$MODEL" in
  *arctic*) ;;
  *) echo "  !! arctic 이 아닙니다. docker 를 되돌리고 인덱스를 --force 로 다시 만드세요"
     exit 1 ;;
esac

python - <<'PY' || exit 1
import sys
sys.path[:0] = ["src", "."]
from models import load_embedder

health = load_embedder("tei").health()
print(f"  질의 접두어  {health['질의 접두어']}")
print(f"  문서 접두어  {health['문서 접두어']}")
if health["질의 접두어"] == "(없음)":
    print("  !! arctic 은 'query: ' 가 필요합니다. .env 의 EMBED_QUERY_PREFIX 를 보세요")
    sys.exit(1)
PY

ls outputs/vectorstore/*__splade__*.npz >/dev/null 2>&1 \
  && SPLADE=--splade \
  || { SPLADE=""; echo "  Splade npz 가 없어 빼고 잽니다"; }

# 평가 세트를 여기서 한 번 맞춰 둔다. 안 그러면 첫 실행이 이걸 하느라 늦는다
python scripts/retrieval/build_evalset.py --chunks "$CHUNKS" || exit 1

# --- 쓸어 담는다 -------------------------------------------------------------
: > "$LOG"
for pool in $POOLS; do
  echo "" | tee -a "$LOG"
  echo "===== pool $pool  ($(date '+%H:%M')) =====" | tee -a "$LOG"
  # 하나가 죽어도 나머지는 살린다. 지난번 스윕이 세그폴트로 통째로 날아갔다
  python scripts/retrieval/compare_retrieval.py \
    --chunks "$CHUNKS" --evalset "$EVALSET" $SPLADE \
    --pool "$pool" --bm25-weights "$WEIGHTS" \
    --out "$OUT/pool_${pool}.csv" >>"$LOG" 2>&1 \
    || echo "  !! pool $pool 실패 (코드 $?) — 건너뜁니다" | tee -a "$LOG"
done

# --- 아침에 볼 표 ------------------------------------------------------------
echo "" | tee -a "$LOG"
python - "$OUT" $POOLS <<'PY' | tee -a "$LOG"
import sys
from pathlib import Path

import pandas as pd

out, pools = Path(sys.argv[1]), sys.argv[2:]
frames = []
for pool in pools:
    path = out / f"pool_{pool}.csv"
    if not path.exists():
        continue
    frame = pd.read_csv(path)
    frame["pool"] = int(pool)
    frames.append(frame)

if not frames:
    sys.exit("결과 파일이 하나도 없습니다. pool_sweep.log 를 보세요")

all_rows = pd.concat(frames)
for kind in ["배점", "요구사항", "의역"]:
    part = all_rows[all_rows["유형"] == kind]
    print(f"\n[{kind}] MRR")
    print(part.pivot_table(index="설정", columns="pool", values="MRR").round(3))

print("\n유형별 1위")
for kind in ["배점", "요구사항", "의역"]:
    part = all_rows[all_rows["유형"] == kind]
    best = part.loc[part["MRR"].idxmax()]
    print(f"  {kind:6s} {best['설정']}  pool {best['pool']}  MRR {best['MRR']:.3f}")

print("\n주의 — 141문항에서 짝지어 비교하면 실질 노이즈가 0.03~0.04 다.")
print("차이가 그 안이면 pool 은 작은 쪽(=빠른 쪽)을 고른다.")
PY

echo ""
echo "끝났습니다. $LOG"
