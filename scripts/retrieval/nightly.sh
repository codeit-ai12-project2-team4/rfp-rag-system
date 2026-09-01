#!/usr/bin/env bash
# 지금 설정으로 1단계·2단계를 전부 잰다. 자는 동안 돌리는 용도다.
#
#     bash scripts/retrieval/nightly.sh 2>&1 | tee outputs/eval_results/nightly.log
#
# 설정은 config/settings.py 아래쪽 한 곳에만 있다. 파생물이 없거나 어긋나면
# prepare.py 가 만든다. **다른 코퍼스가 섞이는 걸 여기서 막는다.**
#
# 1시간 반쯤 걸린다.

set -u
cd "$(dirname "$0")/../.." || exit 1
OUT=outputs/eval_results

# --- 준비. 여기서 멈추면 뒤가 전부 거짓이 된다 --------------------------------
python scripts/retrieval/prepare.py --build || exit 1

CHUNKS=$(python -c "import sys; sys.path[:0]=['src','.']; from config import retrieval as cfg; print(cfg.chunk_name())")
SET=$(python -c "import sys; sys.path[:0]=['src','.']; from config import retrieval as cfg; print(cfg.EVALSET)")
echo ""
echo "청크 $CHUNKS · 평가 세트 $SET"

step() {  # 이름 결과파일 나머지인자…
  local name=$1 out=$2; shift 2
  echo ""
  echo "===== $name  ($(date '+%H:%M')) ====="
  python scripts/retrieval/compare_retrieval.py --chunks "$CHUNKS" --evalset "$SET" \
    --out "$OUT/$out" "$@" && echo "  → $out" || echo "  !! $name 실패"
}

# --- 2단계. scoped 가 제품의 측정이고 unscoped 는 하한이다 --------------------
step "2단계 scoped (제품 흐름)"  night_scoped.csv --scoped
step "2단계 unscoped (공고를 모를 때)" night_unscoped.csv

# --- 1단계. 병목이 여기로 옮겨갔다 -------------------------------------------
echo ""
echo "===== 1단계 공고 검색  ($(date '+%H:%M')) ====="
python scripts/retrieval/eval_notices.py --chunks "$CHUNKS" --evalset "$SET" \
  --out "$OUT/night_notices.csv" || echo "  !! 1단계 실패"

# --- 아침에 볼 것 -------------------------------------------------------------
echo ""
echo "======================================================================"
python - "$OUT" <<'PY'
import sys
from pathlib import Path

import pandas as pd

out = Path(sys.argv[1])
for tag, title in [("night_scoped", "2단계 scoped — 제품 숫자"),
                   ("night_unscoped", "2단계 unscoped — 공고를 모를 때")]:
    path = out / f"{tag}.csv"
    if not path.exists():
        continue
    frame = pd.read_csv(path)
    print(f"\n[{title}]")
    print(frame.pivot_table(index="설정", columns="유형", values="MRR").round(3).to_string())

path = out / "night_notices.csv"
if path.exists():
    print("\n[1단계 공고 검색]")
    print(pd.read_csv(path).to_string(index=False))
    print("\nTop10 이 곧 사용자가 목록에서 고를 수 있는 비율이다.")
    print("2단계가 아무리 좋아도 여기서 놓친 만큼은 그대로 잃는다.")
PY
