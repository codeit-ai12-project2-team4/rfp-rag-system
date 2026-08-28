"""
[Q&A Sampler & Benchmark Builder Usage Guide]

1. 역할:
   - 이미 생성된 Q&A 원본(JSONL)을 불러와 결함 검사(루프, 빈 근거, 환각 등)를 수행하고
     4대 벤치마크 카테고리(배점 15, 요구사항 15, 의역 40, 없음 10 = 총 80개)로 정밀 샘플링.
   - LLM 로드가 불필요하여 수 초 내로 실행 가능.

2. 외부 실행 예시 (노트북/다른 스크립트):
   --------------------------------------------------------------------------------
   import sys
   from pathlib import Path

   PROJECT_ROOT = Path(__file__).resolve().parent.parent  # 또는 루트 경로 지정
   if str(PROJECT_ROOT) not in sys.path:
       sys.path.insert(0, str(PROJECT_ROOT))

   from generate_eval_set.generator import load_documents, save_jsonl
   from generate_eval_set.sampler import sample_benchmark_dataset, sample_balanced_items

   # 1. 파일 경로 설정
   RAW_PATH = PROJECT_ROOT / "Eval_Set" / "eval_set.jsonl"
   CLEAN_n_PATH = PROJECT_ROOT / "Eval_Set" / "eval_set_n.jsonl"
   DEFECT_PATH = PROJECT_ROOT / "Eval_Set" / "defect_items.jsonl"
   EVAL_n_PATH = PROJECT_ROOT / "Eval_Set" / "eval_set_n.jsonl"

   # 2. 원본 데이터 로드
   raw_items = load_documents(RAW_PATH)

   # [사용법 A] n개 벤치마크셋 추출 (품질 필터링 + 4대 유형별 쿼터 적용)
   eval_n, defect_items = sample_benchmark_dataset(
       items=raw_items,
       quotas={"배점": 15, "요구사항": 15, "의역": 40, "없음": 10},
       filter_defects=True,
       seed=42
   )
   save_jsonl(eval_n, CLEAN_n_PATH)
   save_jsonl(defect_items, DEFECT_PATH)

   # [사용법 B] 단순 문서별 균등 샘플링 (예: n개 생성기 테스트용)
   eval_n = sample_balanced_items(raw_items, target_count=30, seed=42)
   save_jsonl(eval_n, EVAL_n_PATH)
   --------------------------------------------------------------------------------
"""

import logging
import random
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

UNANSWERABLE_TEMPLATES = [
    "이 사업의 작년도 낙찰 업체 및 계약 금액은 얼마인가?",
    "제안서 작성 시 총괄 책임자의 주민등록번호 기재 란은 어디에 있는가?",
    "본 입찰에서 탈락한 업체에게 지급되는 제안서 보상비는 얼마인가?",
    "과업 수행 중 발생하는 모든 야간 근무 식대 지원 단가는 얼마인가?",
    "발주기관 담당자의 개인 휴대전화 번호는 무엇인가?",
    "이 사업의 전년도 유지보수 사업자 평가 점수는 몇 점이었는가?",
    "제안서 제출 후 2차 기술 면접이 진행되는 장소의 상세 주소는?",
    "본 프로젝트에 투입되는 모든 개발자의 MBTI 요구조건은 무엇인가?",
    "입찰 보증금을 비트코인 등 가상자산으로 납부할 수 있는가?",
    "사업 종료 후 발주처에서 제공하는 해외 연수 프로그램의 일정은?",
]


def inspect_quality(item: dict) -> tuple[bool, str]:
    """클로드 피드백 기반 4대 결함(루프, 빈 근거, 환각, 전처리 파싱 결함) 검사"""
    q = item.get("question", "").strip()
    a = item.get("answer", "").strip()
    ev = item.get("evidence_text", "").strip()

    # 의도된 미답변 질문은 통과
    if (
        "정답 없음" in a
        or "없음" in a
        or "확인할 수 없" in a
        or item.get("question_type") == "unanswerable"
    ):
        return True, "VALID_UNANSWERABLE"

    # [결함 1] 8자 이상 반복 구절 (LLM 생성 루프 폭주)
    loop_pattern = re.compile(r"(.{8,}?)\1{2,}")
    if loop_pattern.search(q) or loop_pattern.search(a):
        return False, "생성_루프_결함"

    # [결함 2] 실질 내용 없는 빈 근거 / 지시어만 있는 경우
    if (
        re.search(r"참고\s*[\'\"]?$|해당\s*없음|별첨|참조", ev)
        or len(re.sub(r"[\s\-_=|]", "", ev)) < 5
    ):
        return False, "근거_내용_부족"

    # [결함 4] 전처리 추출 결함 (대시/언더바/공백 반복 파싱 오류)
    if re.search(r"[-_=.]{6,}", ev):
        return False, "전처리_추출_결함"

    # [결함 3] 답변의 핵심 키워드가 근거/질문에 전무한 경우 (환각)
    tokens = [
        t
        for t in re.findall(r"[가-힣a-zA-Z0-9]{2,}", a)
        if t not in ["입니다", "있습니다", "하며", "위해", "경우", "따라", "통해"]
    ]
    if tokens:
        match_count = sum(1 for t in tokens if t in ev or t in q)
        if match_count == 0:
            return False, "근거_답변_키워드_불일치(환각)"

    return True, "CLEAN"


def classify_qa_type(item: dict) -> str:
    """질문, 정답, 근거 원문의 키워드를 분석하여 카테고리 분류"""
    q = item.get("question", "")
    a = item.get("answer", "")
    full_str = f"{q} {a} {item.get('evidence_text', '')}"

    if "정답 없음" in a or "없음" in a or "확인할 수 없" in a:
        return "없음"
    if re.search(
        r"[A-Z]{2,4}-\d{2,4}|요구사항|기능|규격|명칭|기능명|SFR|과업|내역|항목", q
    ):
        return "요구사항"
    if re.search(
        r"배점|점수|정성|정량|평가|몇\s*점|비율|가산점|가점|한도|기준|%", q
    ) or re.search(r"\d+점|\d+%", full_str):
        return "배점"
    if re.search(
        r"돈이|얼마|어떻게|어디서|언제|누가|기간이|방법은|골라|맡기|무엇|알려", q
    ):
        return "의역"

    return "일반"


def sample_benchmark_dataset(
    items: list[dict],
    quotas: dict[str, int] | None = None,
    filter_defects: bool = True,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    품질 검사 및 쿼터별 벤치마크 데이터셋 추출
    Returns: (final_dataset, defect_items)
    """
    if quotas is None:
        quotas = {"배점": 15, "요구사항": 15, "의역": 40, "없음": 10}

    random.seed(seed)

    # 1. 품질 필터링
    clean_pool = []
    defect_items = []

    for item in items:
        if filter_defects:
            is_valid, reason = inspect_quality(item)
            if is_valid:
                clean_pool.append(item)
            else:
                it_defect = dict(item)
                it_defect["defect_reason"] = reason
                defect_items.append(it_defect)
        else:
            clean_pool.append(item)

    # 2. 카테고리 분류
    categorized = defaultdict(list)
    for item in clean_pool:
        c_type = classify_qa_type(item)
        it_copy = dict(item)
        it_copy["eval_category"] = c_type
        categorized[c_type].append(it_copy)

    # 3. '없음' 데이터 보충
    needed_unanswerable = quotas.get("없음", 0) - len(categorized["없음"])
    if needed_unanswerable > 0:
        doc_ids = list({it.get("doc_id", "unknown") for it in clean_pool}) or [
            "unknown"
        ]
        for i in range(needed_unanswerable):
            doc_id = doc_ids[i % len(doc_ids)]
            template = UNANSWERABLE_TEMPLATES[i % len(UNANSWERABLE_TEMPLATES)]
            categorized["없음"].append(
                {
                    "doc_id": doc_id,
                    "question": f"「{doc_id}」 {template}",
                    "answer": "(정답 없음 - 문서에 명시되지 않은 정보)",
                    "evidence_text": "",
                    "question_type": "unanswerable",
                    "eval_category": "없음",
                }
            )

    # 4. 쿼터 샘플링 (부족 시 일반 풀에서 보충)
    final_dataset = []
    used_ids = set()

    for cat, count in quotas.items():
        pool = [it for it in categorized[cat] if id(it) not in used_ids]
        if len(pool) < count:
            fallback = [it for it in categorized["일반"] if id(it) not in used_ids]
            pool.extend(fallback[: (count - len(pool))])

        random.shuffle(pool)
        selected = pool[:count]
        for it in selected:
            it["eval_category"] = cat
            used_ids.add(id(it))
        final_dataset.extend(selected)

    return final_dataset, defect_items


def sample_balanced_items(
    items: list[dict], target_count: int, seed: int = 42
) -> list[dict]:
    """문서별 단순 균등 분배 샘플링"""
    if len(items) <= target_count:
        return items.copy()

    random.seed(seed)
    docs_map = defaultdict(list)
    for item in items:
        docs_map[item["doc_id"]].append(item)

    for doc_id in docs_map:
        random.shuffle(docs_map[doc_id])

    doc_keys = list(docs_map.keys())
    random.shuffle(doc_keys)

    sampled = []
    while len(sampled) < target_count:
        added = False
        for doc_id in doc_keys:
            if docs_map[doc_id] and len(sampled) < target_count:
                sampled.append(docs_map[doc_id].pop(0))
                added = True
        if not added:
            break

    return sampled
