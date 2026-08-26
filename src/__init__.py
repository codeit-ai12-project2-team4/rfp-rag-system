"""입찰메이트 RFP RAG.

노트북에서는 보통 이렇게 시작한다.

    from src import setup
    setup()                      # src/ 를 경로에 넣고 서버 상태 확인

그 뒤로는 평평하게 쓴다.

    from config import settings
    from preprocessing import load_documents
"""

__version__ = "0.1.0"


def setup(check=True):
    """노트북 맨 앞에서 한 번 부른다. 경로를 잡고 서버 상태를 보여준다."""
    import sys
    from pathlib import Path

    # 이 파일이 src/__init__.py 다. src/ 를 경로에 넣어야 평평한 import 가 된다.
    here = Path(__file__).resolve().parent          # src/
    for folder in (here, here.parent):              # src/ 와 프로젝트 루트(config/)
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))

    from _config import settings

    settings.make_dirs()
    if check:
        from models import check_servers

        check_servers()
    return paths
