"""나라장터에서 용역 입찰공고를 받아 `data/metadata/data_list.csv` 와 `data/raw/` 를 채운다.

    python src/crawl.py --count --days 1        **먼저 이것.** 안 받고 몇 건인지만
    python src/crawl.py --days 1                오늘 것 (IT 사업만)
    python src/crawl.py --days 1 --all          안 거르고 전부
    python src/crawl.py --from 202410010000 --to 202410312359
    python src/crawl.py --days 1 --limit 5      먼저 5건만 (처음엔 이걸로)
    python src/crawl.py --selftest              망·키 없이 로직만 30초

**상세 조회를 안 부른다.** 목록 응답(`getBidPblancListInfoServc`)에 이미
`ntceSpecDocUrl1~10` 과 `ntceSpecFileNm1~10` 이 들어 있다. 공고당 요청이
한 번에서 두 번으로 늘 이유가 없다.

`.env` 의 `DATA_API_KEY` 는 **디코딩 키**다. `requests` 가 params 로 넘기면서
한 번 인코딩하므로, 인코딩된 키를 넣으면 두 번 인코딩돼 401 이 난다.

이어서 돌려도 안전하다 — 이미 `data/raw/` 에 있는 공고는 건너뛴다.
크론에 그대로 걸어도 된다.
"""

import argparse
import csv
import os
import sys
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _folder in (_ROOT / "src", _ROOT):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import requests

from config import settings

API = "https://apis.data.go.kr/1230000/ad/BidPublicInfoService/getBidPblancListInfoServc"

# data_list.csv 의 컬럼. 순서까지 원본과 같아야 한다 —
# preprocessing/run.py 의 COLUMNS 가 이 한글 이름으로 매핑한다.
HEADER = [
    "공고 번호", "공고 차수", "사업명", "사업 금액", "발주 기관",
    "공개 일자", "입찰 참여 시작일", "입찰 참여 마감일",
    "사업 요약", "파일형식", "파일명", "텍스트",
]

# 첨부 10개 중 어느 게 제안요청서인가. 앞에 있을수록 우선.
RFP_WORDS = ("제안요청서", "제안 요청서", "과업내용서", "과업 내용서",
             "과업지시서", "제안요청", "입찰설명서", "용역설명서")
# **전처리가 읽을 수 있는 것만 받는다** (rfp/common.py 의 SUPPORTED_EXTENSIONS).
# doc/docx 는 파이프라인이 못 읽는다 — 받아봐야 data/raw 에 쌓이기만 하고
# 문서가 안 된다. 첨부가 그것뿐인 공고는 차라리 "첨부 못 찾음" 으로 세는 게
# 낫다. 지원을 늘리려면 양쪽을 같이 고칠 것.
DOC_EXTS = (".hwp", ".hwpx", ".pdf")


def pick_attachment(item):
    """첨부 10개 중 제안요청서로 보이는 것 하나. 없으면 None.

    **휴리스틱이다.** 나라장터는 어느 첨부가 제안요청서인지 알려주지 않는다.
    파일명에 `제안요청서` 가 들어 있으면 거의 맞고, 없으면 문서 확장자 중
    첫 번째를 집는다(대개 입찰공고문이다).

    ponytail: 이름만 본다. 틀리면 첨부를 다 받아서 글자 수로 고르는 방법이
    있는데, 그건 공고당 다운로드가 10배가 된다. 놓친 게 눈에 띄면 그때.

    Args:
        item (dict): 목록 응답의 item 하나.

    Returns:
        (url, filename) 또는 None.
    """
    found = []
    for n in range(1, 11):
        name = (item.get(f"ntceSpecFileNm{n}") or "").strip()
        url = (item.get(f"ntceSpecDocUrl{n}") or "").strip()
        if not name or not url:
            continue
        # URL 에 공백이 섞여 오는 경우가 있다. 공백은 URL 에 그냥 못 들어간다.
        url = "".join(url.split())
        if not name.lower().endswith(DOC_EXTS):
            continue
        rank = next((i for i, word in enumerate(RFP_WORDS) if word in name), len(RFP_WORDS))
        found.append((rank, n, url, name))
    if not found:
        return None
    _, _, url, name = min(found)
    return url, name


def _text(value):
    """None 과 문자열 'null' 을 빈 칸으로. 나머지는 문자열로."""
    if value is None:
        return ""
    value = str(value).strip()
    return "" if value.lower() in ("null", "none", "nan") else value


def to_row(item, file_name):
    """목록 item 하나를 data_list.csv 한 줄로.

    `사업 요약` 과 `텍스트` 는 **비운다.** 요약은 API 에 없는 값이고(원본
    CSV 에서는 사람이 넣었다), 텍스트는 원본을 `data/raw/` 에 받으므로
    중복인 데다 잘려 있어서 못 쓴다 — 컬럼만 자리를 지킨다.

    Args:
        item (dict): 목록 응답의 item.
        file_name (str): `data/raw/` 에 저장한 파일 이름.

    Returns:
        HEADER 순서의 리스트.
    """
    # 배정예산이 비면 추정가격으로. 둘 다 없는 공고가 실제로 있다.
    amount = _text(item.get("asignBdgtAmt")) or _text(item.get("presmptPrce"))
    # 수요기관이 실제 발주처다. 공고기관은 조달청인 경우가 많다.
    agency = _text(item.get("dminsttNm")) or _text(item.get("ntceInsttNm"))
    return [
        _text(item.get("bidNtceNo")),
        _text(item.get("bidNtceOrd")),
        _text(item.get("bidNtceNm")),
        amount,
        agency,
        _text(item.get("bidNtceDt")) or _text(item.get("rgstDt")),
        _text(item.get("bidBeginDt")),
        _text(item.get("bidClseDt")),
        "",
        Path(file_name).suffix.lstrip(".").lower(),
        file_name,
        "",
    ]


def doc_id(item):
    """`20241001798-0`. `make_doc_id()` 가 CSV 에서 만드는 것과 같은 값이어야 한다.

    차수 `000` 을 그대로 두면 `-000` 이 되어 기존 코퍼스의 `-0` 과 안 맞는다.
    `preprocessing.run.make_doc_id` 가 `int(seq)` 를 쓰므로 여기서도 int 로 만든다.
    """
    order = _text(item.get("bidNtceOrd")) or "0"
    try:
        order = int(float(order))
    except ValueError:
        order = 0
    return f"{_text(item.get('bidNtceNo'))}-{order}"


def existing_ids(path=None):
    """이미 가진 공고의 doc_id 집합. **CSV 가 기준이다.**

    파일명으로 보면 안 된다. 처음 받은 100건은 `{기관}_{사업명}.hwp` 로 이름이
    붙어 있어서 doc_id 와 안 맞는다. 그걸로 판정하면 **같은 공고를 다시 받아
    doc_id 가 겹치는 행이 두 개 생긴다** — 전처리·평가가 조용히 깨진다.

    `preprocessing.run.make_doc_id` 와 같은 규칙으로 만든다.
    """
    path = path or settings.META_CSV
    if not path.exists():
        return set()
    found = set()
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            no = (row.get("공고 번호") or "").strip().removesuffix(".0")
            if not no:
                continue
            order = (row.get("공고 차수") or "0").strip()
            try:
                order = int(float(order))
            except ValueError:
                order = 0
            found.add(f"{no}-{order}")
    return found


def fetch_page(key, begin, end, page, rows=100, session=None):
    """목록 한 페이지. (items, totalCount).

    Raises:
        RuntimeError: 인증 실패 등 API 가 정상코드를 안 줄 때.
    """
    get = (session or requests).get
    response = get(API, timeout=30, params={
        "serviceKey": key, "type": "json", "inqryDiv": "1",
        "inqryBgnDt": begin, "inqryEndDt": end,
        "pageNo": page, "numOfRows": rows,
    })
    response.raise_for_status()
    try:
        body = response.json()
    except ValueError:
        # 키가 틀리면 JSON 이 아니라 XML 에러를 준다. 그 내용을 보여줘야 안다.
        raise RuntimeError(f"JSON 이 아닌 응답입니다:\n{response.text[:400]}") from None
    head = body.get("response", {}).get("header", {})
    if head.get("resultCode") not in ("00", "0"):
        raise RuntimeError(f"API 오류 {head.get('resultCode')}: {head.get('resultMsg')}")
    inner = body["response"].get("body") or {}
    items = (inner.get("items") or [])
    if isinstance(items, dict):  # 1건이면 리스트가 아니라 dict 로 온다
        items = [items]
    return items, int(inner.get("totalCount") or 0)


def download(url, path, session=None):
    """첨부 하나를 내려받는다. 실패하면 파일을 남기지 않는다."""
    get = (session or requests).get
    response = get(url, timeout=120, stream=True)
    response.raise_for_status()
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "wb") as f:
        for block in response.iter_content(1 << 16):
            f.write(block)
    tmp.rename(path)  # 다 받은 것만 제자리로 — 중간에 끊겨도 다음 실행이 다시 받는다
    return path.stat().st_size


def is_skippable(item):
    """받지 말아야 할 공고면 이유 문자열, 아니면 None.

    같은 공고를 두 번 받는 걸 막는 건 doc_id 로 되지만(`crawl()` 참고),
    **같은 사업이 다른 공고번호로 또 오는 건 그걸로 안 걸린다.**
    API 가 그 관계를 직접 알려주므로 내용 비교를 할 이유가 없다.

        ntceKindNm     취소공고 — 없어진 공고다. 색인에 들어가면 안 된다
        befBidBbancNo  이전입찰공고번호. 이게 있으면 재공고이고,
                       가리키는 옛 공고는 이 건으로 대체된 것이다
    """
    kind = _text(item.get("ntceKindNm"))
    if "취소" in kind:
        return "취소공고"
    return None


def superseded(item):
    """이 공고가 대체한 옛 공고번호. 없으면 None.

    재공고를 받을 때 옛 건을 같이 정리하려면 이 값이 필요하다.
    지금은 찍어만 준다 — 지우는 건 아카이브 정책이 정해진 뒤에.
    """
    return _text(item.get("befBidBbancNo")) or None


def count(begin, end, rows=100):
    """**받지 않고** 이 기간에 뭐가 오는지만 본다. 요청 한 번.

    보관 정책을 정하려면 하루에 몇 MB 가 쌓이는지부터 알아야 하는데, 그걸 아는
    제일 싼 방법이다. 분류·용역구분 분포도 같이 찍는다 — **무엇으로 거를지**를
    정하는 게 건수보다 어려운 문제이기 때문이다.

    `infoBizYn`(정보화사업여부)로 거르면 안 된다. 옵션 필드라 기관이 거의 안
    채운다 — 실측 1/100 이었다.
    """
    key = os.environ.get("DATA_API_KEY")
    if not key:
        raise SystemExit(".env 에 DATA_API_KEY 가 없습니다 (디코딩 키).")
    items, total = fetch_page(key, begin, end, 1, rows)
    if not items:
        print("0건")
        return 0

    n = len(items)
    live = [i for i in items if not is_skippable(i)]
    attach = [i for i in live if pick_attachment(i)]
    hit = [i for i in attach if looks_like_it(i)]

    print(f"전체 {total:,}건 · 표본 {n}건")
    print(f"  취소공고 제외          {len(live):3d}/{n}")
    print(f"  제안요청서 첨부 있음    {len(attach):3d}/{n}")
    print(f"  사업명이 IT 로 보임     {len(hit):3d}/{n}  ({len(hit)/n:.0%})")

    nego = [i for i in attach if is_negotiated(i)]
    both = [i for i in attach if is_negotiated(i) and looks_like_it(i)]
    print(f"  협상에 의한 계약        {len(nego):3d}/{n}  (제안서를 내는 사업)")
    print(f"  IT ∧ 협상              {len(both):3d}/{n}")

    # 공공조달 대/중분류와 용역구분은 안 찍는다 — 실측에서 93/93 빈칸,
    # 93/93 일반용역이라 아무것도 안 알려준다.
    for field, label in (("sucsfbidMthdNm", "낙찰방법"),
                         ("cntrctCnclsMthdNm", "계약체결방법")):
        tally = Counter(_text(i.get(field)) or "(빈칸)" for i in live)
        print(f"\n  {label} — 표본 {len(live)}건")
        for name, cnt in tally.most_common(6):
            print(f"    {cnt:3d}  {name[:56]}")

    print(f"\n  IT 로 보이는 사업명 예시")
    for i in hit[:8]:
        print(f"    {_text(i.get('bidNtceNm'))[:60]}")

    days = max(1, (_stamp(end) - _stamp(begin)).days + 1)
    print(f"\n조회 기간 {days}일 · 하루 평균 {total / days:,.0f}건")
    for label, ratio in (("전부 받으면", len(attach) / n),
                         ("IT 만", len(hit) / n),
                         ("협상만", len(nego) / n),
                         ("IT ∧ 협상", len(both) / n)):
        per_day = total / days * ratio
        mb = per_day * 3.3
        print(f"  {label:12} 하루 {per_day:6,.0f}건 · {mb:7,.0f}MB/일 · "
              f"{mb * 365 / 1024:6,.0f}GB/년")
    print("\n공고당 3.3MB = 원본 1.6 + 전처리본 0.5 + 청크 0.8 + 벡터 0.4 (100건 실측)")
    return total


def _stamp(text):
    """YYYYMMDDHHMM → datetime."""
    return datetime.strptime(text[:8], "%Y%m%d")


# 사업명에 이게 있으면 우리 고객이 볼 만한 건으로 센다.
#
# **명사만 넣는다.** "개발·구축·보안·정보보호" 같은 말을 넣었더니
# `개인정보보호 배상책임보험`, `긴급의료/보안 이·후송`, `교육콘텐츠 개발` 이
# 전부 걸렸다. 우리가 찾는 건 행위가 아니라 **대상**이다.
#
# 지금 코퍼스 100건으로 검증:
#   느슨한 목록(동사 포함)  재현율 100/100 · 오탐 4/4 (전부 헛발질)
#   이 목록                재현율 100/100 · 오탐 1/4
# 남은 오탐 하나는 "에너지소산율 기반 ISE 시스템 분석" 이다. `시스템` 이 들어
# 있어서 이름만으로는 못 거른다.
#
# ponytail: 이름만 본다. 공공조달분류번호로 거르려 했는데 **실측 93/93 이 빈칸**
# 이라 못 쓴다. 용역구분도 93/93 일반용역이라 변별력이 없다.
IT_WORDS = (
    "시스템", "정보화", "소프트웨어", "솔루션", "플랫폼", "포털",
    "홈페이지", "웹사이트", "전산", "데이터베이스", "빅데이터",
    "ISMP", "ISP", "BPR", "EA", "LMS", "ERP", "GIS", "DW", "CDW",
)

# 제안서를 내는 사업인가. 실측한 낙찰방법명 분포에서 뽑았다.
#     32  협상에의한계약-협상에 의한 낙찰자 결정
#      9  규격가격동시입찰-제안적격자 중 예가 내 최저가 투찰자   ← 이것도 제안서를 낸다
#      4  협상에의한계약-…(SW사업)
#     21  소액수의견적 / 14 수의시담 / 3 최저가낙찰제            ← 제안서가 없다
NEGOTIATED = ("협상", "경쟁적 대화", "경쟁적대화", "규격가격", "제안적격자")

# 낙찰방법명에 이게 있으면 사업명을 볼 것도 없이 우리 것이다.
SW_MARK = "SW사업"


def is_negotiated(item):
    """제안서를 평가하는 사업인가. 적격심사·수의계약은 제안서가 없다."""
    blob = " ".join(_text(item.get(f)) for f in
                    ("sucsfbidMthdNm", "cntrctCnclsMthdNm", "bidNtceNm"))
    return any(word in blob for word in NEGOTIATED)


def wanted(item):
    """받을 공고인가.

    **사업명이 IT 면 받는다. 협상 여부는 안 본다.**
    실측(하루 294건 기준):

        전부       253건/일  298GB/년
        IT 만       32건/일   38GB/년
        IT ∧ 협상   24건/일   28GB/년

    교집합으로 좁혀도 **10GB/년밖에 안 아끼면서** 유지보수·운영 건을 버린다
    (`산학협력단 정보시스템 운영 용역업체 선정` 같은 것이 코퍼스에 있다).
    싸지지 않는 쪽으로 재현율을 팔 이유가 없다.

    낙찰방법이 SW사업이면 이름을 안 봐도 받는다 — 실측 4/100 을 공짜로 얻는다.
    """
    return looks_like_it(item) or SW_MARK in _text(item.get("sucsfbidMthdNm"))


def looks_like_it(item):
    """사업명이 IT/시스템 사업으로 보이는가. 재기 위한 어림이다."""
    name = _text(item.get("bidNtceNm"))
    return any(word in name for word in IT_WORDS)


def crawl(begin, end, limit=None, rows=100, sleep=0.3, take_all=False):
    """받아서 저장한다. 이미 있는 공고는 건너뛴다.

    Returns:
        (새로 받은 수, 건너뛴 수, 첨부를 못 찾은 수).
    """
    key = os.environ.get("DATA_API_KEY")
    if not key:
        raise SystemExit(".env 에 DATA_API_KEY 가 없습니다 (디코딩 키).")

    settings.RAW.mkdir(parents=True, exist_ok=True)
    settings.METADATA.mkdir(parents=True, exist_ok=True)
    have = existing_ids()

    session = requests.Session()
    added = skipped = missing = 0
    page, total = 1, None

    while total is None or (page - 1) * rows < total:
        items, total = fetch_page(key, begin, end, page, rows, session)
        if not items:
            break
        print(f"  {page}쪽 {len(items)}건 (전체 {total:,}건)")
        for item in items:
            if limit and added >= limit:
                total = 0
                break
            key_id = doc_id(item)
            if key_id in have:
                skipped += 1
                continue
            if not take_all and not wanted(item):
                skipped += 1
                continue
            reason = is_skippable(item)
            if reason:
                print(f"    건너뜀 {key_id}: {reason}")
                skipped += 1
                continue
            old_no = superseded(item)
            if old_no:
                print(f"    재공고 {key_id} — {old_no} 를 대체함")
            picked = pick_attachment(item)
            if not picked:
                missing += 1
                continue
            url, name = picked
            path = settings.RAW / f"{key_id}{Path(name).suffix.lower()}"
            try:
                size = download(url, path, session)
            except Exception as error:  # noqa: BLE001 - 한 건 실패로 전체를 멈추지 않는다
                print(f"    받기 실패 {key_id}: {error}")
                missing += 1
                continue
            # 한 건씩 바로 붙인다. 끝에 몰아 쓰면 중간에 죽었을 때 파일만 남고
            # 행이 사라지는데, 다음 실행은 그 파일을 '이미 있음' 으로 보고
            # 건너뛰어서 **영영 행이 안 생긴다.**
            append_csv([to_row(item, path.name)])
            have.add(key_id)
            added += 1
            print(f"    {key_id}  {size / 1024:,.0f}KB  {name[:40]}")
            time.sleep(sleep)
        page += 1

    return added, skipped, missing


def append_csv(rows, path=None):
    """data_list.csv 에 붙인다. 없으면 헤더부터 쓴다.

    pandas 로 안 쓰는 이유 — 공고번호·차수를 숫자로 읽었다가 다시 쓰면
    `20241001798.0` / `0.0` 이 된다. 그게 doc_id 를 깨뜨려서 `tidy_doc_id()`
    가 생겼다. csv 모듈은 문자열을 문자열로 쓴다.
    """
    path = path or settings.META_CSV
    fresh = not path.exists()
    with open(path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        if fresh:
            writer.writerow(HEADER)
        writer.writerows(rows)


def selftest():
    """망도 키도 없이 고르기·매핑·doc_id 만 확인한다."""
    item = {
        "bidNtceNo": "20241001798", "bidNtceOrd": "000",
        "bidNtceNm": "트랙운영 학사정보시스템 고도화",
        "asignBdgtAmt": "130000000", "presmptPrce": "118181818",
        "dminsttNm": "한영대학", "ntceInsttNm": "조달청",
        "bidNtceDt": "2024-10-04 13:51:23", "bidBeginDt": "",
        "bidClseDt": "2024-10-15 17:00:00",
        "ntceSpecFileNm1": "1.입찰공고문.hwp", "ntceSpecDocUrl1": "https://x/1",
        "ntceSpecFileNm2": "2.제안요청서.hwpx", "ntceSpecDocUrl2": "https://x/2 ",
        "ntceSpecFileNm3": "3.물량내역서.xlsx", "ntceSpecDocUrl3": "https://x/3",
    }
    url, name = pick_attachment(item)
    assert name == "2.제안요청서.hwpx", name          # 공고문보다 제안요청서
    assert url == "https://x/2", repr(url)            # 공백 제거
    assert doc_id(item) == "20241001798-0", doc_id(item)   # -000 이 아니다

    row = to_row(item, "20241001798-0.hwpx")
    assert len(row) == len(HEADER)
    assert dict(zip(HEADER, row))["발주 기관"] == "한영대학"   # 조달청이 아니다
    assert dict(zip(HEADER, row))["사업 금액"] == "130000000"  # 배정예산 우선
    assert dict(zip(HEADER, row))["파일형식"] == "hwpx"

    assert pick_attachment({"ntceSpecFileNm1": "a.zip", "ntceSpecDocUrl1": "u"}) is None
    assert is_skippable({"ntceKindNm": "취소공고"}) == "취소공고"
    assert is_skippable({"ntceKindNm": "등록공고"}) is None
    assert superseded({"befBidBbancNo": "R26BK01482673"}) == "R26BK01482673"
    assert superseded(item) is None
    assert looks_like_it({"bidNtceNm": "학사정보시스템 고도화"})
    assert not looks_like_it({"bidNtceNm": "청사 조경수 전지 용역"})
    # 동사를 넣었다가 잡혔던 것들. 다시 들어오면 여기서 걸린다.
    assert not looks_like_it({"bidNtceNm": "개인정보보호 배상책임보험 용역"})
    assert not looks_like_it({"bidNtceNm": "해외 긴급의료/보안 이·후송 및 보험 지원"})
    assert not looks_like_it({"bidNtceNm": "핵심부품 시험평가를 위한 교육콘텐츠 개발 용역"})
    assert is_negotiated({"sucsfbidMthdNm": "협상에 의한 계약"})
    assert is_negotiated({"sucsfbidMthdNm": "규격가격동시입찰-제안적격자 중 예가 내 최저가"})
    assert not is_negotiated({"sucsfbidMthdNm": "소액수의견적", "bidNtceNm": "화단 정비"})
    assert wanted({"bidNtceNm": "학사정보시스템 고도화"})
    assert wanted({"bidNtceNm": "무엇이든", "sucsfbidMthdNm": "협상에의한계약-…(SW사업)"})
    assert not wanted({"bidNtceNm": "청사 조경수 전지 용역", "sucsfbidMthdNm": "적격심사"})
    assert not is_negotiated({"sucsfbidMthdNm": "적격심사제", "bidNtceNm": "청소 용역"})
    assert pick_attachment({}) is None
    # 배정예산이 비면 추정가격
    assert to_row({**item, "asignBdgtAmt": ""}, "x.hwp")[3] == "118181818"
    print("selftest 통과 — 첨부 고르기 / doc_id / 컬럼 매핑")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, help="오늘부터 N일. --days 1 은 오늘 하루")
    parser.add_argument("--hours", type=int,
                        help="지금부터 N시간 전까지. 하루에 여러 번 도는 크론용")
    parser.add_argument("--from", dest="begin", help="YYYYMMDDHHMM")
    parser.add_argument("--to", dest="end", help="YYYYMMDDHHMM")
    parser.add_argument("--limit", type=int, help="이만큼만 받고 멈춘다")
    parser.add_argument("--all", action="store_true",
                        help="IT 필터 없이 전부 받는다 (하루 253건 · 298GB/년)")
    parser.add_argument("--count", action="store_true",
                        help="받지 않고 건수·비율만 본다")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        return selftest()

    if args.hours:
        # 겹쳐도 된다 — 이미 가진 doc_id 는 건너뛴다. 그래서 상태 파일이 필요 없다.
        now = datetime.now()
        begin = (now - timedelta(hours=args.hours)).strftime("%Y%m%d%H%M")
        end = now.strftime("%Y%m%d%H%M")
    elif args.days:
        # --days 1 은 오늘 하루. days-1 을 빼야 달력 기준으로 N일이 된다.
        now = datetime.now()
        begin = (now - timedelta(days=args.days - 1)).strftime("%Y%m%d0000")
        end = now.strftime("%Y%m%d2359")
    elif args.begin and args.end:
        begin, end = args.begin, args.end
    else:
        raise SystemExit("--hours / --days / --from+--to 중 하나는 줘야 합니다.")

    print(f"조회 {begin} ~ {end}")
    if args.count:
        return count(begin, end)
    added, skipped, missing = crawl(begin, end, limit=args.limit,
                                    take_all=args.all)
    print(f"\n새로 받음 {added}건 · 이미 있음 {skipped}건 · 첨부 못 찾음 {missing}건")
    print(f"원본 {settings.RAW}")
    print(f"메타 {settings.META_CSV}")
    if added:
        print("\n다음: python -m preprocessing.run   그리고 인덱스 다시 만들기")


if __name__ == "__main__":
    main()
