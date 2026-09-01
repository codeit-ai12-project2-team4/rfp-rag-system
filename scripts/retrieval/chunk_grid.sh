#!/usr/bin/env bash
# 청킹 기법 x 크기를 한 평가 세트로 쓸어 담는다. **VM 에서 돌린다.**
#
# v4 는 표 마커(`|` `---`)를 다 걷어냈고 실제 글자 손실은 문서 전체에서 2자다.
# 그래서 **같은 --size 라도 v4 청크가 v3 보다 내용을 14% 더 담는다.** v3 에서
# 고른 1200 이 v4 에서도 최적일 이유가 없고, 절 경계로 자르는 section 도
# 표 모양이 바뀌었으니 다시 재야 한다.
#
#     bash scripts/retrieval/chunk_grid.sh
#
# 셀마다 청킹 + 인덱싱 + 측정으로 7분쯤, 전부 50분쯤 걸린다.

set -u
cd "$(dirname "$0")/../.." || exit 1

OUT=outputs/eval_results
SET=eval_qa_grid
# 이름:전처리본:방식:크기:겹침
CELLS="
v3_rec1200:cleaned_documents_v3:recursive:1200:200
v4_rec900:cleaned_documents_v4:recursive:900:150
v4_rec1200:cleaned_documents_v4:recursive:1200:200
v4_rec1500:cleaned_documents_v4:recursive:1500:250
v4_rec1800:cleaned_documents_v4:recursive:1800:300
v4_rec2100:cleaned_documents_v4:recursive:2100:350
"

MODEL=$(curl -s -m 5 localhost:8085/info | python -c \
  "import json,sys; print(json.load(sys.stdin).get('model_id','(응답없음)'))" 2>/dev/null)
echo "임베더  $MODEL"
case "$MODEL" in *arctic*) ;; *) echo "  !! arctic 이 아닙니다"; exit 1 ;; esac

# --- 자르고 색인한다 ----------------------------------------------------------
NAMES=""
for cell in $CELLS; do
  IFS=: read -r tag docs how size overlap <<<"$cell"
  name="${docs}__${how}_${size}_${overlap}"
  NAMES="$NAMES $name"

  if [ ! -f "outputs/chunks/chunks_${name}.jsonl" ]; then
    echo ""
    echo "== 청킹 $tag =="
    python src/chunking.py --docs "$docs" --how "$how" --size "$size" \
      --overlap "$overlap" || exit 1
  fi
  if [ ! -f "outputs/vectorstore/${name}__tei/index.faiss" ]; then
    echo "== 색인 $tag =="
    python src/vectorstore.py --chunks "$name" || exit 1
  fi
done

# --- 모든 셀에서 채점되는 문항만. 이걸 안 하면 셀마다 시험지가 달라진다 -------
echo ""
echo "== 공통 평가 세트 =="
# shellcheck disable=SC2086
python scripts/retrieval/build_evalset.py --chunks $NAMES --out "${SET}.json" || exit 1

# --- 측정 --------------------------------------------------------------------
for cell in $CELLS; do
  IFS=: read -r tag docs how size overlap <<<"$cell"
  name="${docs}__${how}_${size}_${overlap}"
  echo ""
  echo "== 측정 $tag  ($(date '+%H:%M')) =="
  python scripts/retrieval/compare_retrieval.py --chunks "$name" --evalset "$SET" \
    --out "$OUT/grid_${tag}.csv" >/dev/null 2>&1 \
    && echo "  → grid_${tag}.csv" \
    || echo "  !! $tag 실패"
done

echo ""
python scripts/retrieval/compare_runs.py "$OUT"/grid_*.csv
