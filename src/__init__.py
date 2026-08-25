"""입찰메이트 RFP RAG.

노트북에서는 보통 이렇게 시작한다.

    import setup
    setup()                      # 경로 잡고 서버 상태 확인
"""

__version__ = "0.1.0"


def setup(check=True):
    """노트북 맨 앞에서 한 번 부른다. 경로를 잡고 서버 상태를 보여준다."""
    import paths

    paths.make_dirs()
    if check:
        from models import check_servers

        check_servers()
    return paths
