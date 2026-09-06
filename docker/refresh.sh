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
STAMP=outputs/refresh.json

# 어디서 끝났든 결과를 남긴다. ssh 가 없는 사람이 /health 로 확인할 수 있게.
stamp() { printf '{"at":"%s","ok":%s,"step":"%s"}\n' "$(date -Is)" "$1" "$2" > "$STAMP"; }
trap 'stamp false "$STEP"' ERR

STEP=시작
echo "[$(date '+%F %T')] refresh 시작  $*"

STEP=수집
"$PY" src/crawl.py "$@"
STEP=전처리·색인
"$PY" scripts/retrieval/prepare.py --build --service

# 토큰은 .env 에서 꺼낸다. 값에 따옴표를 감싸 두면 그대로 헤더에 들어가 401 이 난다.
STEP=반영
TOKEN=$(grep '^API_TOKEN=' .env | cut -d= -f2-)
if ! curl -fsS -X POST -H "x-api-token: $TOKEN" http://localhost:8010/reload; then
    echo
    echo "  /reload 가 실패했습니다. 색인은 이미 만들어졌고 반영만 안 됐습니다."
    echo "    404  API 가 옛 코드입니다 — git pull 후 한 번만 재시작하세요"
    echo "    401  .env 의 API_TOKEN 값에 따옴표가 붙어 있는지 보세요"
    echo "    7    API 가 안 떠 있습니다 (systemctl status bidmate-api)"
    echo "  손으로 반영: sudo systemctl restart bidmate-api"
    exit 1
fi
echo

stamp true 끝
echo "[$(date '+%F %T')] refresh 끝"
