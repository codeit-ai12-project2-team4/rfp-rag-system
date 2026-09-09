"""검색 파트가 지금 쓰는 설정. **여기가 유일한 진실이다.**

이번 주에 같은 사고를 다섯 번 냈는데 전부 설정이 여러 곳에 흩어져서 한쪽만
고쳐진 탓이었다. 오류가 안 나고 조용히 다른 것을 본다.

    v3 인덱스에 v4 임베더        차원이 같으면 faiss 가 안 죽는다
    옛 청크로 만든 Splade npz    청크 개수까지 같아서 안 죽는다
    옛 청크로 만든 평가 세트      정답이 사라진 걸 '검색 실패' 로 잡는다
    Dense 는 v3 · Hybrid 는 v4   한 실행 안에서 코퍼스가 섞인다
    retriever.CHUNKS 가 옛것      API 가 옛 설정으로 답한다

바꿀 때는 여기만 고친다. 파생물(청크·인덱스·평가세트)이 안 맞으면
`scripts/retrieval/prepare.py --build` 가 만들어 준다.

생성 파트의 모델 설정은 `config/model_config.py` 에 따로 있다.
"""

# 환경변수로 한 번만 덮어쓸 수 있다. 새 전처리본이 왔을 때 파일을 안 고치고
# 그대로 재보려는 것이다. 스크립트마다 인자를 뚫는 것보다 이게 짧고, 여기를
# 읽는 것(prepare · compare_retrieval · retriever · api)이 전부 따라온다.
#
#     DOCS=cleaned_documents_v5 bash scripts/retrieval/nightly.sh
#     DOCS=cleaned_documents_v5 SIZE=1200 python scripts/retrieval/prepare.py --build
#
# 좋다고 판단되면 그때 아래 기본값을 고친다.
import os

DOCS = os.environ.get("DOCS", "cleaned_documents")
HOW = os.environ.get("HOW", "recursive")
SIZE = int(os.environ.get("SIZE", "1500"))
OVERLAP = int(os.environ.get("OVERLAP", "250"))
EMBED = os.environ.get("EMBED", "tei")
# 벡터 저장소. faiss 가 기본이고 lance 는 나란히 두고 재 보는 중이다.
# 이름 규칙이 같아서 인덱스 이름은 그대로 쓴다 — 폴더만 다르다
# (outputs/vectorstore vs outputs/lancedb).
STORE = os.environ.get("STORE", "lance")
RERANK = os.environ.get("RERANK", "tei")
POOL = int(os.environ.get("POOL", "30"))  # 2단계. 리랭커에 넘길 후보 수 (9/4 스윕)
# 1단계 공고 검색. 청크를 공고로 묶으므로 목록 길이는 이것보다 짧다.
# 200 은 8/28 옛 세트에서 온 값이었고 9/4 스윕에서 전 구간 손해였다.
NOTICE_POOL = int(os.environ.get("NOTICE_POOL", "20"))
TOP_K = int(os.environ.get("TOP_K", "8"))  # 리랭커가 남길 수. 예산에서 다시 잘린다
EVALSET = os.environ.get("EVALSET", "eval_qa_both")


# 전처리팀이 청크까지 잘라서 주면 우리 이름 규칙(`__recursive_`)과 안 맞는다.
# 그럴 때만 이름을 통째로 덮어쓴다. 우리가 자른 게 아니라는 게 이름에 남는다.
#
#     CHUNKS=cleaned_documents_v8__pipeline_1500_250 bash scripts/retrieval/nightly.sh
#
# `pipeline` 은 전처리팀 파이프라인이 자른 것이라는 표시다. `recursive` 라고 쓰면
# 다음 사람이 chunking.py 로 다시 만들 수 있다고 착각하는데, 그러면 표 원자성과
# 검색용 길이 기준이 사라진다 — 오류는 안 나고 성적만 조용히 달라진다.
# 이름은 scripts/retrieval/ingest.py 의 target_name() 이 짓는다.
# 기본값은 DOCS 와 같은 버전으로 맞춰 둔다. 여기가 어긋나면 API 가 옛 코퍼스로
# 답하는데 아무 오류도 안 난다 — 이 파일 맨 위 목록의 마지막 줄이 그 사고다.
#
# **9/10 에 그 사고가 기본값 안에 들어 있었다.** 기본값이
# `chunks_cleaned_documents_v8__pipeline_1500_250`(8,920청크, 실험 잔재)을
# 가리키는 동안 서버·크론은 `.env` 를 따라 14,872청크를 보고 있었다. 두 파일이
# 다 있어서 `.env` 없는 환경(새 체크아웃, 컨테이너)에서는 조용히 옛 코퍼스로
# 답했을 것이다. **기본값을 산 파일로 맞춘다 — 없는 이름이면 차라리 터진다.**
CHUNKS = os.environ.get("CHUNKS", "chunks_cleaned_documents__pipeline")


# 발주기관이 **잘못 올린 첨부.** 차수가 오르며 문서가 통째로 교체된 경우다.
#
# 차수 필터(`chunking.drop_stale_revisions`)로는 못 거른다 — 본문이 다른 게
# 맞기 때문이다. 유사도로 가를 수도 없다: 9/9 실측에서 **완전히 다른 두 사업의
# 제안요청서가 0.918** 이었다. 나라장터 문서는 어휘가 겹쳐(계약·입찰·과업)
# 다른 사업끼리도 0.9 가 나온다. 진짜 개정(마감일 한 줄)은 0.999 였다.
# 그 사이에 임계값을 놓을 자리가 없다.
#
# 그래서 **사람이 확인하고 이유를 적는다.** 후보는 자동으로 찾는다:
#
#     python scripts/retrieval/check_revisions.py --diff
#
# 차수가 여럿인 공고만 나오므로(322건 중 2건) 눈으로 볼 수 있는 양이다.
# 값은 화면에 그대로 보여주는 문구다 — 컨설턴트가 옛 문서를 이미 받아 갔을 수
# 있으니 "왜 사라졌나" 를 알아야 한다.
REPLACED_DOCS = {
    "R26BK01719775-0": (
        "0차 첨부가 다른 사업(공공기술기반 창업 아이디어 특허 컨설팅 및 출원 지원)의 "
        "제안요청서였습니다. 1차에서 올바른 문서로 교체되었습니다."
    ),
}


def chunk_name(docs=None, how=None, size=None, overlap=None):
    """청크 세트 이름. 이름이 곧 실험 조건이다.

    Args:
        docs, how, size, overlap: 생략하면 위 기본값.

    Returns:
        str: 예) `cleaned_documents_v4__recursive_1500_250`.
        `CHUNKS` 가 있으면 인자와 무관하게 그 이름을 그대로 쓴다.
    """
    # 인자를 하나라도 주면 계산해서 쓴다. 무조건 CHUNKS 를 돌려주면 설정을
    # 바꿔 가며 도는 비교 스크립트(compare_chunking·sweep_chunks)가 전부
    # 같은 이름을 받아 같은 인덱스를 본다.
    if CHUNKS and docs is None and how is None and size is None and overlap is None:
        return CHUNKS
    return f"{docs or DOCS}__{how or HOW}_{size or SIZE}_{overlap or OVERLAP}"


def index_name(chunks=None, embed=None):
    """FAISS 인덱스 이름.

    Args:
        chunks: 청크 이름. 생략하면 `chunk_name()`.
        embed: 임베더 종류. 생략하면 `EMBED`.

    Returns:
        str: 예) `cleaned_documents_v4__recursive_1500_250__tei`
    """
    return f"{chunks or chunk_name()}__{embed or EMBED}"
