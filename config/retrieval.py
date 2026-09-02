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

DOCS = os.environ.get("DOCS", "cleaned_documents_v6")
HOW = os.environ.get("HOW", "recursive")
SIZE = int(os.environ.get("SIZE", "1500"))
OVERLAP = int(os.environ.get("OVERLAP", "250"))
EMBED = os.environ.get("EMBED", "tei")
RERANK = os.environ.get("RERANK", "tei")
POOL = int(os.environ.get("POOL", "80"))  # 리랭커에 넘길 후보 수. 30~120 을 재고 골랐다
TOP_K = int(os.environ.get("TOP_K", "8"))  # 리랭커가 남길 수. 예산에서 다시 잘린다
EVALSET = os.environ.get("EVALSET", "eval_qa_both")


def chunk_name(docs=None, how=None, size=None, overlap=None):
    """청크 세트 이름. 이름이 곧 실험 조건이다.

    Args:
        docs, how, size, overlap: 생략하면 위 기본값.

    Returns:
        str: 예) `cleaned_documents_v4__recursive_1500_250`
    """
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
