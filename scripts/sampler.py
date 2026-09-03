"""
[Q&A Sampler & Benchmark Builder Usage Guide]
 
1. 역할:
   - 이미 생성된 Q&A 원본(JSONL)을 불러와 결함 검사(루프, 빈 근거, 환각 등)를 수행하고
     4대 벤치마크 카테고리(배점 15, 요구사항 15, 의역 40, 없음 10 = 총 80개)로 정밀 샘플링.
   - LLM 로드가 불필요하여 수 초 내로 실행 가능.
 
2. 정답(answer) 필드 안내 (검색 매칭 실패 방지):
   - generation 단계가 만든 답변은 사람이 읽기 좋게 재구성되어
     (예: '| 기술능력 | 40 |') 실제 청크(page_content)의 원문 표기(예: '기술능력 40')와
     달라, 검색 결과를 문자열로 비교하는 자동 평가에서 전부 매칭 실패로 버려지는 문제가 있었다.
   - 이를 막기 위해 sample_benchmark_dataset()이 만드는 최종 데이터셋의 "answer"는
     evidence_text(=page_content에서 그대로 추출한 근거 원문)를 그대로 사용한다.
     즉 "answer"는 이제 page_content 기준의 검색/채점용 정답이다.
   - generation이 만든 원래 답변은 "generation_answer" 필드로 남기며, 이는 사람이
     보기 편한 설명/근거로만 쓰고 자동 채점에는 쓰지 않는다.
   - 결함 검사(inspect_quality)는 여전히 generation 원본 답변을 대상으로 수행하여
     생성 단계의 결함(루프, 환각 등)을 그대로 잡아낸다. 필드 교체는 결함 검사를 통과한
     뒤, 최종 데이터셋을 만드는 시점에만 일어난다.
   - evidence_text 자체도 여러 위치를 이어 붙이며 쪽번호/표 찌꺼기가 섞여, 그대로 쓰면
     코퍼스 리터럴 매칭이 38~53%에 그친다(검색기 파트 build_evalset.py 기준). docs를
     넘기면 evidence_text와 실제 문서 원문 사이 가장 긴 연속 일치 조각을 찾아 그걸
     answer로 쓴다 — 문서에 실제로 있는 문자열이라 matches()를 건드리지 않아도 된다.
   - docs를 넘기면 추가로 검색기 파트와 같은 기준으로 판정해 defect_items에 담는다:
     중복질문(같은 질문 중복), 코퍼스에없음(정답이 문서 어디에도 없음),
     라벨불일치(정답은 있는데 doc_id가 가리키는 문서엔 없음),
     공고특정불가(정답이 여러 문서에 있어 특정 불가). 나머지는 전부 살려서 버려지는
     질문을 최소화하고, quotas의 "의역" 개수는 그대로 유지된다(카테고리별 쿼터 샘플링은
     안 바뀜).
 
3. 외부 실행 예시 (노트북/다른 스크립트):
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
 
import difflib
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
    """클로드 피드백 기반 4대 결함(루프, 빈 근거, 환각, 전처리 파싱 결함) 검사
 
    generation이 만든 원본 answer/evidence_text를 그대로 검사한다. 최종 데이터셋의
    answer를 page_content 기준으로 바꾸는 것은 이 함수를 통과한 뒤의 일이므로,
    여기서는 생성 단계의 결함만 순수하게 잡아낸다.
    """
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
 
 
# 쪽번호 / 표 찌꺼기(대시·언더바·등호·파이프 반복) — inspect_quality의 전처리_추출_결함
# 패턴과 같은 잡음 기준을 쓴다.
_NOISE_RE = re.compile(r"\(?\s*\d+\s*쪽\s*\)?|-\s*\d+\s*-|\bp\.?\s*\d+\b|[-_=|]{4,}", re.IGNORECASE)
 
 
def ground_evidence(evidence_text: str, doc_text: str, min_len: int = 4) -> str:
    """evidence_text와 문서 원문(doc_text) 사이 가장 긴 연속 일치 조각을 뽑는다.
 
    evidence_text는 여러 위치를 이어 붙이며 쪽번호/표 찌꺼기가 섞여 그대로 쓰면
    코퍼스 리터럴 매칭이 38~53%에 그친다. 매칭 전에 양쪽의 잡음을 서로 다른
    자리표시자(둘이 절대 같아질 수 없는 문자)로 치환해 "쪽번호만 우연히 일치"하는
    오탐을 막는다 — 치환은 길이를 바꾸지 않으므로 매칭 결과 인덱스를 원문(치환 전)
    문자열에 그대로 써도 안전하다. 반환값은 문서 원문에서 그대로 슬라이스하므로
    matches()를 건드리지 않고도 매칭률을 끌어올릴 수 있다.
    """
    if not evidence_text or not doc_text:
        return ""
    masked_ev = _NOISE_RE.sub(lambda m: "\x00" * len(m.group()), evidence_text)
    masked_doc = _NOISE_RE.sub(lambda m: "\x01" * len(m.group()), doc_text)
    matcher = difflib.SequenceMatcher(None, masked_ev, masked_doc, autojunk=False)
    match = matcher.find_longest_match(0, len(masked_ev), 0, len(masked_doc))
    if match.size < min_len:
        return ""
    return doc_text[match.b : match.b + match.size].strip()
 
 
def build_doc_lookup(docs: list[dict]) -> dict[str, str]:
    """doc_id(공고)별 청크를 이어 붙여 {doc_id: 문서 전문} 룩업을 만든다.
 
    LangChain Document 스타일({"page_content": ..., "metadata": {"source": ...}})과
    평평한 스타일({"doc_id"/"source": ..., "page_content": ...}) 둘 다 지원한다.
    """
    chunks_by_doc = defaultdict(list)
    for d in docs:
        meta = d.get("metadata", {})
        doc_id = meta.get("source") or d.get("source") or d.get("doc_id")
        content = d.get("page_content", "")
        if doc_id and content:
            chunks_by_doc[doc_id].append(content)
    return {doc_id: "\n".join(parts) for doc_id, parts in chunks_by_doc.items()}
 
 
def ground_in_corpus(
    evidence_text: str,
    doc_id: str,
    doc_lookup: dict[str, str],
    min_len: int = 4,
    ambiguous_min_len: int = 8,
) -> tuple[str, str | None]:
    """evidence_text를 코퍼스에 그라운딩하고 검색기 파트 기준 결함 사유를 판정한다.
 
    Returns: (grounded_answer, defect_reason) — 결함 없으면 defect_reason은 None.
    라벨(doc_id)이 가리키는 문서에서 먼저 찾고, 없으면 코퍼스 전체에서 찾는다:
 
    - 코퍼스에없음: 코퍼스 어디에도 없다
    - 라벨불일치: 다른 문서에는 있는데 doc_id가 가리키는 문서엔 없다
    - 공고특정불가: 라벨 문서에도 있지만 다른 문서에도 있어 질문만으로 특정 불가
 
    # ponytail: ambiguous_min_len=8은 "40"류 짧은 숫자가 여러 문서에 우연히 겹쳐
    # 공고특정불가로 오판되는 걸 막는 임시 문턱값. 질문 텍스트까지 함께 보는 판정이
    # 필요해지면 그때 올린다.
    """
    if not evidence_text or not doc_lookup:
        return evidence_text, None
 
    own_match = ground_evidence(evidence_text, doc_lookup.get(doc_id, ""), min_len)
    if own_match:
        if len(own_match) >= ambiguous_min_len:
            hit_count = sum(1 for text in doc_lookup.values() if own_match in text)
            if hit_count > 1:
                return own_match, "공고특정불가"
        return own_match, None
 
    _best_doc, best_match = None, ""
    for other_id, text in doc_lookup.items():
        m = ground_evidence(evidence_text, text, min_len)
        if len(m) > len(best_match):
            _best_doc, best_match = other_id, m
    if not best_match:
        return "", "코퍼스에없음"
    return best_match, "라벨불일치"
 
 
def build_page_content_answer(
    item: dict, doc_lookup: dict[str, str] | None = None
) -> tuple[dict, str | None]:
    """최종 정답을 generation 답변 대신 page_content 원문 기준으로 교체.
 
    - answer: doc_lookup이 있으면 evidence_text를 코퍼스에 그라운딩한 결과(가장 긴
      연속 일치 조각), 없으면 evidence_text를 그대로 사용 (검색 매칭/자동 채점용).
      셋 다 비어 있으면(예: 미답변 유형) generation 답변을 그대로 쓴다.
    - generation_answer: 원래 generation이 만든 답변. 사람이 읽는 설명/근거로만 사용.
    Returns: (item, corpus_defect_reason) — doc_lookup 없으면 defect_reason은 None.
 
    예) generation answer: '| 기술능력 | 40 |'
        evidence_text(page_content 원문): '기술능력 40'
        -> 검색 결과(page_content)와 정확히 일치해야 하는 자동 평가에는 후자를 써야
           '| 기술능력 | 40 |' != '기술능력 40' 로 인한 전량 매칭 실패를 막을 수 있다.
    """
    out = dict(item)
    generation_answer = out.get("answer", "").strip()
    ev = out.get("evidence_text", "").strip()
    out["generation_answer"] = generation_answer
 
    if not ev or not doc_lookup:
        out["answer"] = ev if ev else generation_answer
        return out, None
 
    grounded, defect = ground_in_corpus(ev, out.get("doc_id", ""), doc_lookup)
    out["answer"] = grounded or generation_answer
    return out, defect
 
 
def is_unanswerable(item: dict) -> bool:
    """미답변 유형인지 판정 (코퍼스 그라운딩/공고특정 검사에서 제외하기 위함)."""
    a = item.get("answer", "")
    return (
        item.get("question_type") == "unanswerable"
        or "정답 없음" in a
        or "확인할 수 없" in a
    )
 
 
def sample_benchmark_dataset(
    items: list[dict],
    quotas: dict[str, int] | None = None,
    filter_defects: bool = True,
    seed: int = 42,
    docs: list[dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    품질 검사 및 쿼터별 벤치마크 데이터셋 추출
    Returns: (final_dataset, defect_items)
 
    final_dataset의 각 항목은 answer가 page_content 기준(evidence_text를 문서 원문에
    그라운딩한 결과, 또는 evidence_text 원문)으로 치환되어 있고, generation이 만든
    원래 답변은 generation_answer에 별도로 담긴다.
 
    docs를 넘기면(문서 청크 리스트, "docs" 매개변수 참고) 검색기 파트
    (scripts/retrieval/build_evalset.py) 기준의 판정을 추가로 적용한다: 중복질문 /
    코퍼스에없음 / 라벨불일치 / 공고특정불가는 defect_items로 보내고, 나머지는 전부
    살린다 — evidence_text를 그대로 정답으로 쓰던 예전 방식보다 코퍼스 매칭 실패로
    버려지는 질문이 줄어든다. 카테고리별 쿼터(quotas)는 그대로 적용되므로 "의역"
    비율도 그대로 유지된다.
    """
    if quotas is None:
        quotas = {"배점": 15, "요구사항": 15, "의역": 40, "없음": 10}
 
    random.seed(seed)
    doc_lookup = build_doc_lookup(docs) if docs else {}
 
    # 0. 중복질문 제거 (검색기 파트 기준: 같은 질문이 여러 세트에 있으면 하나만 남긴다)
    seen_questions = set()
    defect_items = []
    deduped_items = []
    for item in items:
        key = re.sub(r"\s+", "", item.get("question", ""))
        if key and key in seen_questions:
            it_dup = dict(item)
            it_dup["defect_reason"] = "중복질문"
            defect_items.append(it_dup)
        else:
            seen_questions.add(key)
            deduped_items.append(item)
    items = deduped_items
 
    # 1. 품질 필터링 (generation 원본 answer/evidence_text 기준으로 검사)
    clean_pool = []
 
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
 
    # 2. 카테고리 분류 + 정답을 page_content 기준으로 교체 (+ 코퍼스 그라운딩 판정)
    categorized = defaultdict(list)
    for item in clean_pool:
        it_copy, corpus_defect = build_page_content_answer(item, doc_lookup=doc_lookup)
 
        if corpus_defect and not is_unanswerable(item):
            it_copy["defect_reason"] = corpus_defect
            defect_items.append(it_copy)
            continue
 
        c_type = classify_qa_type(item)
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
            no_answer_text = "(정답 없음 - 문서에 명시되지 않은 정보)"
            categorized["없음"].append(
                {
                    "doc_id": doc_id,
                    "question": f"「{doc_id}」 {template}",
                    "answer": no_answer_text,
                    "generation_answer": no_answer_text,
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
 
 
def _demo() -> None:
    """그라운딩/결함판정 자가 점검. 쪽번호 오탐 방지, 라벨불일치, 의역 쿼터 유지를 확인한다."""
    docs = [
        {
            "page_content": "제1장 총칙\n2. 평가기준\n기술능력 40 관리능력 20 가격 40\n(3쪽)",
            "metadata": {"source": "d1"},
        },
        {"page_content": "제출방법: 이메일 제출", "metadata": {"source": "d1"}},
        {"page_content": "완전히 다른 내용의 단독조항이 여기 있다.", "metadata": {"source": "d2"}},
    ]
    doc_lookup = build_doc_lookup(docs)
 
    # 쪽번호가 섞여도 진짜 내용 조각을 골라야 한다 (숫자/쪽번호 조각으로 새면 안 됨)
    grounded, defect = ground_in_corpus("기술능력 40\n(3쪽)", "d1", doc_lookup)
    assert grounded == "기술능력 40", grounded
    assert defect is None
 
    # 라벨(doc_id)이 가리키는 문서엔 없고 다른 문서에만 있으면 라벨불일치
    grounded, defect = ground_in_corpus("완전히 다른 내용의 단독조항이 여기 있다", "d1", doc_lookup)
    assert defect == "라벨불일치", defect
 
    # 코퍼스 어디에도 없으면 코퍼스에없음
    grounded, defect = ground_in_corpus("존재하지 않는 문장입니다", "d1", doc_lookup)
    assert defect == "코퍼스에없음", defect
 
    # 의역 쿼터는 요청한 개수 그대로 나와야 한다 (그라운딩/결함판정과 무관하게)
    items = [
        {
            "doc_id": "d1",
            "question": f"제출 비용은 얼마 {i}?",
            "answer": "이메일 제출",
            "evidence_text": "제출방법: 이메일 제출",
            "question_type": "extractive",
        }
        for i in range(5)
    ]
    final, _ = sample_benchmark_dataset(
        items, quotas={"배점": 0, "요구사항": 0, "의역": 5, "없음": 0}, seed=1, docs=docs
    )
    assert len(final) == 5 and all(it["eval_category"] == "의역" for it in final)
    assert all(it["answer"] in doc_lookup["d1"] for it in final)
 
    print("OK")
 
 
if __name__ == "__main__":
    _demo()