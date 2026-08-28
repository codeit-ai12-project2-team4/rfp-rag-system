from .generator import EvalItem, QAEvalGenerator, load_documents, save_jsonl
from .sampler import inspect_quality, sample_balanced_items, sample_benchmark_dataset

__all__ = [
    "EvalItem",
    "QAEvalGenerator",
    "inspect_quality",
    "load_documents",
    "sample_balanced_items",
    "sample_benchmark_dataset",
    "save_jsonl",
]

"""

============================================
                실행 예시
============================================
import sys
from pathlib import Path

PROJECT_ROOT = Path("C:/Users/asd/Desktop/중급 프젝/v2_chosim")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generate_eval_set.generator import load_documents, save_jsonl
from generate_eval_set.sampler import sample_benchmark_dataset

# 경로 설정
RAW_PATH = PROJECT_ROOT / "Eval_Set" / "eval_set.jsonl"
CLEAN_80_PATH = PROJECT_ROOT / "Eval_Set" / "eval_set_80.jsonl"
DEFECT_PATH = PROJECT_ROOT / "Eval_Set" / "defect_items.jsonl"

# 1. 데이터 로드
raw_items = load_documents(RAW_PATH)

# 2. 품질 정제 및 80개 벤치마크 샘플링 (튜플 반환)
eval_80, defect_items = sample_benchmark_dataset(
    items=raw_items,
    quotas={"배점": 15, "요구사항": 15, "의역": 40, "없음": 10},
    filter_defects=True,
    seed=42
)

# 3. 저장
save_jsonl(eval_80, CLEAN_80_PATH)
save_jsonl(defect_items, DEFECT_PATH)

print(f"\n✅ 최종 80개 벤치마크셋 저장 완료: {CLEAN_80_PATH.name}")
print(f"❌ 걸러진 결함 데이터({len(defect_items)}개) 저장 완료: {DEFECT_PATH.name}")

"""
