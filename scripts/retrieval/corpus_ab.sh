#!/usr/bin/env bash
# v3 대 v4 를 Splade 까지 붙여 같은 문항으로 잰다. **VM 에서 돌린다.**
#
# 두 코퍼스를 비교할 때 지켜야 하는 것은 하나뿐이다 — 같은 문항으로 잴 것.
# 세 번 연속 이걸 어겨서 결과가 무효였다. 그래서 교집합 세트를 먼저 만들고,
# 그 세트로만 두 번 잰다.
#
#     bash scripts/retrieval/corpus_ab.sh
#
# Splade 인덱스는 코퍼스마다 따로 필요하다. GPU 로 코퍼스를 통째로 인코딩하므로
# 코퍼스당 몇 분 걸린다. 이미 있고 지문이 맞으면 건너뛴다.

set -u
cd "$(dirname "$0")/../.." || exit 1

A=cleaned_documents_v4__recursive_1200_200
B=cleaned_documents_v3__recursive_1200_200
SET=eval_qa_both
OUT=outputs/eval_results

echo "== 사전 점검 =="
MODEL=$(curl -s -m 5 localhost:8085/info | python -c \
  "import json,sys; print(json.load(sys.stdin).get('model_id','(응답없음)'))" 2>/dev/null)
echo "  임베더  $MODEL"
case "$MODEL" in
  *arctic*) ;;
  *) echo "  !! arctic 이 아닙니다"; exit 1 ;;
esac

for name in "$A" "$B"; do
  [ -f "outputs/chunks/chunks_${name}.jsonl" ] || { echo "  !! 청크 없음: $name"; exit 1; }
  ls "outputs/vectorstore/${name}__tei/index.faiss" >/dev/null 2>&1 \
    || { echo "  !! 인덱스 없음: $name — python src/vectorstore.py --chunks $name"; exit 1; }
done

# --- 못 쓰게 된 것들을 지운다 -------------------------------------------------
# 문항 세트가 달랐던 실행들이다. 남겨두면 나중에 이 숫자를 인용하게 된다.
rm -f "$OUT"/v4.csv "$OUT"/v4_qa_both.csv "$OUT"/v4_both.csv "$OUT"/v3_both.csv
rm -f data/eval_qa_merged.json data/eval_qa_merged.meta.json   # 98문항짜리 반쪽

# --- Splade 인덱스. 코퍼스마다 하나씩 -----------------------------------------
for name in "$A" "$B"; do
  echo ""
  echo "== Splade 인덱스: $name =="
  python scripts/retrieval/build_splade.py --chunks "$name" || exit 1
done

# --- 교집합 세트. 두 코퍼스 모두에서 채점되는 문항만 --------------------------
echo ""
echo "== 교집합 평가 세트 =="
python scripts/retrieval/build_evalset.py --chunks "$A" "$B" --out "${SET}.json" || exit 1

# --- 같은 세트로 두 번 --------------------------------------------------------
for name in "$A" "$B"; do
  tag=$(echo "$name" | sed 's/cleaned_documents_//; s/__recursive.*//')
  echo ""
  echo "== 측정: $name  ($(date '+%H:%M')) =="
  python scripts/retrieval/compare_retrieval.py \
    --chunks "$name" --evalset "$SET" --splade \
    --out "$OUT/${tag}_splade.csv" || echo "  !! $name 실패"
done

# --- 부호를 센다 --------------------------------------------------------------
echo ""
python scripts/retrieval/compare_runs.py "$OUT/v4_splade.csv" "$OUT/v3_splade.csv"
