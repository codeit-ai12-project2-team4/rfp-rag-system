"""전처리팀 파이프라인. 원본 hwp/pdf → 전처리본 jsonl + 청크 jsonl.

    from preprocessing.rfp import run_pipeline

`preprocessing/pipeline.py` 한 파일(2,430줄)을 그 안의 섹션 표지대로 나눈 것이다.
경위는 `src/preprocessing/README.md`.

**바깥 폴더의 `preprocessing/{hwp,pdf,clean,toc,run}.py` 와는 다른 물건이다.**
그쪽은 검색 파트가 쓰는 우리 코드고 지금도 살아 있다. 이름이 겹쳐서 하위
패키지로 뺐다.
"""

from preprocessing.rfp.build import run_pipeline, write_jsonl

__all__ = ["run_pipeline", "write_jsonl"]
