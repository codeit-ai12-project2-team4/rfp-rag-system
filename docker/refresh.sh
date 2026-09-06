#!/usr/bin/env bash
# 수집 → 전처리 → 색인 → 무중단 반영. **크론이 부르는 유일한 진입점이다.**
#
#     docker/refresh.sh --hours 4      낮에 3시간마다
#     docker/refresh.sh --days 2       새벽에 놓친 것 메우기
#
# 예전에는 크론에 이 세 단계가 흩어져 있었고, 마지막이 `systemctl restart` 였다.
# BM25 형태소 분석 결과를 캐시하고 나서 기동이 185초 → 1.9초가 됐고, 재시작이
# `/reload` 로 바뀌면서 **낮에도 돌릴 수 있게 됐다.** 그래서 한 파일로 모은다.
#
# `set -e` 라 앞 단계가 실패하면 뒤가 안 돈다. 크롤링이 실패했는데 색인을
# 다시 만드는 일은 없다.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# 겹쳐 돌면 같은 청크를 두 번 임베딩하고, /reload 가 반쯤 만든 색인을 집어 간다.
# 앞 실행이 아직이면 조용히 건너뛴다 — 다음 차례가 어차피 3시간 뒤에 온다.
exec 9>/tmp/bidmate-refresh.lock
flock -n 9 || { echo "[$(date '+%F %T')] 앞 실행이 아직 돌고 있어 건너뜁니다"; exit 0; }

PY=.venv/bin/python
echo "[$(date '+%F %T')] refresh 시작  $*"

"$PY" src/crawl.py "$@"
"$PY" scripts/retrieval/prepare.py --build --service

# 토큰은 .env 에서 꺼낸다. 값에 따옴표를 감싸 두면 그대로 헤더에 들어가 401 이 난다.
TOKEN=$(grep '^API_TOKEN=' .env | cut -d= -f2-)
curl -fsS -X POST -H "x-api-token: $TOKEN" http://localhost:8010/reload
echo

echo "[$(date '+%F %T')] refresh 끝"
