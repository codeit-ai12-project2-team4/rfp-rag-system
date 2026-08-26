## config 디렉토리 사용

- settings.py
  - 내용: 프로젝트 전반의 고정 세팅값을 정의
  - 메서드 사용법
    - load_env(): `.env`파일의 내용을 훑어서 `os.environ`에 입력함. ※ os.environ 은 python 런타임이 읽어오는 환경 변수를 담고 있음
    - make_dirs(): 프로젝트 초기 세팅시 필요한 디렉토리를 만들어줌