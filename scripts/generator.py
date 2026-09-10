"""RFP 코퍼스에서 LLM으로 Q&A를 생성해 바로 최종 160개 벤치마크셋을 만든다.

기존 generator.py/sampler.py/adapt_evalset.py 3단계 파이프라인을 걷어내고
한 파일로 합쳤다. 생성 단계에서부터 코퍼스 문서의 실제 doc_id(metadata.source)를
그대로 쓰기 때문에, "파일명 → 공고ID 해석" 같은 별도 단계가 필요 없다.

절차:
    1. 문서를 청크로 나눠 Qwen 모델로 후보 Q&A를 생성한다.
       - 대부분은 일반 질문(요구사항/배점/의역으로 나중에 분류됨)
       - 문서마다 일정 비율은 "이 문서엔 없지만 그럴듯한" 미답변 질문도 생성
         (고정 템플릿 반복 문제를 피하려고 매번 해당 문서 맥락에 맞게 새로 만든다)
    2. answer 대신, evidence_text와 문서 원문 사이 최장 연속 일치 조각을 keywords로 쓴다
       (표 렌더링 차이로 검색 매칭이 깨지는 문제 방지).
    3. 다음 결함은 걸러서 defect_items로 뺀다:
       빈키워드 / 중복질문 / 코퍼스에없음 / 라벨불일치 / 공고특정불가
    4. 요구사항/배점/의역/없음 = 15:15:120:10(총 160)으로 쿼터 샘플링한다.

사용법 (노트북):
    from generate_final_evalset import run
    final, defects = run(
        docs="Pipeline_Output/Pipeline_v2/cleaned_documents.jsonl",
        out="Eval_Set/eval_set_160.jsonl",
    )

사용법 (터미널):
    python generate_final_evalset.py --docs cleaned_documents.jsonl --out eval_set_160.jsonl
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)
logger.info("=== generator.py 로드됨 (rev: cuda-fallback-v2) ===")

# 요구사항 15 : 배점 15 : 의역 120 : 없음 10 = 총 160
DEFAULT_QUOTAS = {"배점": 15, "요구사항": 15, "의역": 120, "없음": 10}

_NOISE_RE = re.compile(
    r"\(?\s*\d+\s*쪽\s*\)?|-\s*\d+\s*-|\bp\.?\s*\d+\b|[-_=|]{4,}", re.IGNORECASE
)

QA_SYSTEM_PROMPT = """당신은 RFP(입찰제안요청서) 문서를 분석하여 RAG 시스템 평가용 질문-답변 쌍을 구축하는 전문가입니다.
주어진 본문에 명시된 사실만을 근거로 구체적이고 명확한 질문과 정답을 작성하세요.
질문은 요구사항 번호/명칭, 평가 배점, 계약방식·기간·예산처럼 표현을 바꿔 묻는 질문 등
다양한 성격이 섞이게 하세요.

반드시 다음 JSON 형식으로만 응답하세요:
```json
{
  "qa_pairs": [
    {"question": "구체적인 질문", "answer": "정답",
     "evidence_text": "답변 근거가 되는 원문 문장 (원문 그대로 복사)"}
  ]
}
```"""

QA_USER_PROMPT = """다음은 RFP 문서 "{doc_id}"의 본문 일부입니다.
이 본문 내용만을 바탕으로 {n_questions}개의 질문-답변 쌍을 JSON 형식으로 작성하세요.

--- 본문 내용 ---
{clean_text}
--- 본문 끝 ---"""

# "없음" 유형 전용 프롬프트: 고정 템플릿을 쓰지 않고 해당 문서 맥락에 맞게 매번 새로
# 만들게 해서, "낙찰 업체/휴대폰 번호"류 5~10개 템플릿이 계속 반복되는 문제를 피한다.
UNANSWERABLE_SYSTEM_PROMPT = """당신은 RAG 시스템의 "정답 없음" 판단력을 테스트할 질문을 만드는 전문가입니다.
주어진 RFP 문서 내용을 참고해서, 이 문서의 주제와는 그럴듯하게 관련 있어 보이지만
실제로는 이 문서 어디에도 답이 없는 질문을 만드세요. 문서마다 다른 종류의 질문을
만들어야 하며, 이미 흔히 쓰이는 "담당자 휴대폰 번호"나 "낙찰 업체" 같은 뻔한
질문은 피하고, 이 문서 특유의 사업 내용에 빗대어 만드세요.

반드시 다음 JSON 형식으로만 응답하세요:
```json
{"question": "이 문서엔 없는, 그럴듯한 질문 1개"}
```"""

UNANSWERABLE_USER_PROMPT = """다음은 RFP 문서 "{doc_id}"의 본문 일부입니다.

--- 본문 내용 ---
{clean_text}
--- 본문 끝 ---

이 문서에는 없지만, 이 사업 내용을 아는 사람이라면 궁금해할 법한 질문을 1개 만드세요."""


@dataclass
class RawCandidate:
    doc_id: str
    question: str
    answer: str
    evidence_text: str
    is_unanswerable: bool = False


# --------------------------------------------------------------------------
# LLM 생성
# --------------------------------------------------------------------------
class QAGenerator:
    def __init__(
        self, model_id: str = "Qwen/Qwen2.5-3B-Instruct", device: str = "cuda:0"
    ):
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning(
                "CUDA를 요청했지만 사용 불가능합니다 (torch가 CPU 전용으로 설치됐거나 "
                "드라이버 문제). CPU로 대체합니다 — 3B 모델도 CPU에서는 매우 느리니, "
                "가능하면 CUDA 지원 torch를 재설치하는 걸 권장합니다."
            )
            device = "cpu"
        self.device = device
        self._dtype = torch.float16 if device.startswith("cuda") else torch.float32
        logger.info("Loading tokenizer & model: %s", model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=self._dtype, device_map=self.device
        )
        logger.info("Model loaded successfully on GPU!")

    def _chat(
        self, system_prompt: str, user_prompt: str, max_new_tokens: int = 1024
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate( # type: ignore
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

    @staticmethod
    def _extract_json(text: str) -> dict:
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        raw = match.group(1) if match else text
        return json.loads(raw.strip())

    def generate_answerable(
        self, doc_id: str, chunk: str, n_questions: int = 3
    ) -> list[RawCandidate]:
        if not chunk.strip():
            return []
        response = self._chat(
            QA_SYSTEM_PROMPT,
            QA_USER_PROMPT.format(
                doc_id=doc_id, n_questions=n_questions, clean_text=chunk
            ),
        )
        try:
            data = self._extract_json(response)
            return [
                RawCandidate(
                    doc_id=doc_id,
                    question=qa["question"],
                    answer=qa["answer"],
                    evidence_text=qa["evidence_text"],
                    is_unanswerable=False,
                )
                for qa in data.get("qa_pairs", [])
                if "question" in qa and "answer" in qa and "evidence_text" in qa
            ]
        except Exception as e:  # noqa: BLE001
            logger.warning("JSON 파싱 실패 (%s): %s", doc_id, e)
            return []

    def generate_unanswerable(self, doc_id: str, chunk: str) -> RawCandidate | None:
        if not chunk.strip():
            return None
        response = self._chat(
            UNANSWERABLE_SYSTEM_PROMPT,
            UNANSWERABLE_USER_PROMPT.format(doc_id=doc_id, clean_text=chunk),
            max_new_tokens=256,
        )
        try:
            data = self._extract_json(response)
            question = data.get("question", "").strip()
            if not question:
                return None
            return RawCandidate(
                doc_id=doc_id,
                question=question,
                answer="(정답 없음 - 문서에 명시되지 않은 정보)",
                evidence_text="",
                is_unanswerable=True,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("미답변 질문 생성 파싱 실패 (%s): %s", doc_id, e)
            return None


# --------------------------------------------------------------------------
# 코퍼스 그라운딩
# --------------------------------------------------------------------------
def build_doc_lookup(docs: list[dict]) -> dict[str, str]:
    chunks_by_doc: dict[str, list[str]] = defaultdict(list)
    for d in docs:
        meta = d.get("metadata", {})
        doc_id = meta.get("source") or d.get("source") or d.get("doc_id")
        content = d.get("page_content", "")
        if doc_id and content:
            chunks_by_doc[doc_id].append(content)
    return {doc_id: "\n".join(parts) for doc_id, parts in chunks_by_doc.items()}


def ground_evidence(evidence_text: str, doc_text: str, min_len: int = 6) -> str:
    if not evidence_text or not doc_text:
        return ""
    masked_ev = _NOISE_RE.sub(lambda m: "\x00" * len(m.group()), evidence_text)
    masked_doc = _NOISE_RE.sub(lambda m: "\x01" * len(m.group()), doc_text)
    matcher = difflib.SequenceMatcher(None, masked_ev, masked_doc, autojunk=False)
    match = matcher.find_longest_match(0, len(masked_ev), 0, len(masked_doc))
    if match.size < min_len:
        return ""
    return doc_text[match.b : match.b + match.size].strip()


def ground_in_corpus(
    evidence_text: str,
    doc_id: str,
    doc_lookup: dict[str, str],
    min_len: int = 6,
    ambiguous_min_len: int = 8,
) -> tuple[str, str | None]:
    if not evidence_text or not doc_lookup:
        return "", "코퍼스에없음"
    own_match = ground_evidence(evidence_text, doc_lookup.get(doc_id, ""), min_len)
    if own_match:
        if len(own_match) >= ambiguous_min_len:
            hit_count = sum(1 for text in doc_lookup.values() if own_match in text)
            if hit_count > 1:
                return own_match, "공고특정불가"
        return own_match, None
    best_match = ""
    for text in doc_lookup.values():
        m = ground_evidence(evidence_text, text, min_len)
        if len(m) > len(best_match):
            best_match = m
    if not best_match:
        return "", "코퍼스에없음"
    return best_match, "라벨불일치"


# --------------------------------------------------------------------------
# 유형 분류
# --------------------------------------------------------------------------
def classify_type(question: str, answer: str) -> str:
    if re.search(
        r"[A-Z]{2,4}-\d{2,4}|요구사항|기능|규격|명칭|기능명|SFR|과업|내역|항목",
        question,
    ):
        return "요구사항"
    if re.search(
        r"배점|점수|정성|정량|평가|몇\s*점|비율|가산점|가점|한도|기준|%", question
    ) or re.search(r"\d+점|\d+%", f"{question} {answer}"):
        return "배점"
    return "의역"


# --------------------------------------------------------------------------
# 결함 판정 + 최종 변환
# --------------------------------------------------------------------------
@dataclass
class BuildResult:
    final_items: list[dict]
    defect_items: list[dict]


def build_dataset(
    candidates: list[RawCandidate], doc_lookup: dict[str, str]
) -> BuildResult:
    defect_items: list[dict] = []
    seen_questions: set[str] = set()
    converted: list[dict] = []

    for c in candidates:
        dedup_key = re.sub(r"\s+", "", c.question)
        if dedup_key and dedup_key in seen_questions:
            defect_items.append({**c.__dict__, "defect_reason": "중복질문"})
            continue
        if dedup_key:
            seen_questions.add(dedup_key)

        if c.is_unanswerable:
            converted.append(
                {
                    "question": c.question,
                    "keywords": [],
                    "doc_id": c.doc_id,
                    "type": "없음",
                    "answerable": False,
                }
            )
            continue

        if not c.evidence_text.strip():
            defect_items.append({**c.__dict__, "defect_reason": "빈키워드"})
            continue

        grounded, defect = ground_in_corpus(c.evidence_text, c.doc_id, doc_lookup)
        if defect:
            defect_items.append({**c.__dict__, "defect_reason": defect})
            continue

        converted.append(
            {
                "question": c.question,
                "keywords": [grounded],
                "doc_id": c.doc_id,
                "type": classify_type(c.question, c.answer),
                "answerable": True,
            }
        )

    logger.info(
        "변환 완료: 통과 %d개, 결함 %d개 (%s)",
        len(converted),
        len(defect_items),
        {
            r: sum(1 for d in defect_items if d["defect_reason"] == r)
            for r in {d["defect_reason"] for d in defect_items}
        },
    )
    return BuildResult(final_items=converted, defect_items=defect_items)


def sample_by_quota(
    items: list[dict], quotas: dict[str, int], seed: int = 42
) -> list[dict]:
    random.seed(seed)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        by_type[it["type"]].append(it)

    final: list[dict] = []
    for t, count in quotas.items():
        pool = by_type.get(t, [])
        if len(pool) < count:
            logger.warning(
                "'%s' 유형 %d개 요청했지만 %d개만 확보됨", t, count, len(pool)
            )
        random.shuffle(pool)
        final.extend(pool[:count])
    return final


# --------------------------------------------------------------------------
# 입출력
# --------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    logger.info("Loaded %d items from %s", len(items), path)
    return items


def save_jsonl(items: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info("Saved %d items to %s", len(items), path)


# --------------------------------------------------------------------------
# 전체 파이프라인
# --------------------------------------------------------------------------
def run(
    docs: str | Path,
    out: str | Path,
    defect_out: str | Path | None = None,
    model_id: str = "Qwen/Qwen2.5-3B-Instruct",
    device: str = "cuda:0",
    chunk_size: int = 2500,
    max_chunks_per_doc: int = 2,
    n_questions_per_chunk: int = 3,
    unanswerable_per_doc: int = 1,
    quotas: dict[str, int] | None = None,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """문서 로드 → LLM 생성 → 그라운딩/결함필터 → 쿼터 샘플링까지 한 번에 실행한다.

    노트북 예)
        from generate_final_evalset import run
        final, defects = run(
            docs="Pipeline_Output/Pipeline_v2/cleaned_documents.jsonl",
            out="Eval_Set/eval_set_160.jsonl",
        )
    """
    docs_path, out_path = Path(docs), Path(out)
    defect_path = (
        Path(defect_out) if defect_out else out_path.with_name("defect_items.jsonl")
    )
    quotas = quotas or DEFAULT_QUOTAS

    doc_items = load_jsonl(docs_path)
    if not doc_items:
        raise ValueError(f"{docs_path}가 비어있습니다. 코퍼스 경로를 확인하세요.")
    doc_lookup = build_doc_lookup(doc_items)

    generator = QAGenerator(model_id=model_id, device=device)

    candidates: list[RawCandidate] = []
    for doc in tqdm(doc_items, desc="Golden QA 생성 중"):
        meta = doc.get("metadata", {})
        doc_id = (
            meta.get("source") or meta.get("doc_id") or meta.get("filename", "unknown")
        )
        full_text = doc.get("page_content", "")
        if not full_text.strip():
            continue

        chunks = [
            full_text[i : i + chunk_size] for i in range(0, len(full_text), chunk_size)
        ]
        chunks = chunks[:max_chunks_per_doc]

        for chunk in chunks:
            candidates.extend(
                generator.generate_answerable(doc_id, chunk, n_questions_per_chunk)
            )

        for _ in range(unanswerable_per_doc):
            if chunks:
                cand = generator.generate_unanswerable(doc_id, chunks[0])
                if cand:
                    candidates.append(cand)

    logger.info("원본 후보 %d개 생성", len(candidates))

    result = build_dataset(candidates, doc_lookup)
    final_dataset = sample_by_quota(result.final_items, quotas, seed=seed)
    target = sum(quotas.values())
    if len(final_dataset) < target:
        logger.warning(
            "목표 %d개에 %d개 못 미침 — chunk_size/max_chunks_per_doc/n_questions_per_chunk를 "
            "늘려 원본 후보 풀을 키워야 합니다.",
            target,
            len(final_dataset),
        )

    save_jsonl(final_dataset, out_path)
    save_jsonl(result.defect_items, defect_path)
    return final_dataset, result.defect_items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--defect-out", type=Path, default=None)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-size", type=int, default=2500)
    parser.add_argument("--max-chunks-per-doc", type=int, default=2)
    parser.add_argument("--n-questions-per-chunk", type=int, default=3)
    parser.add_argument("--unanswerable-per-doc", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run(
        docs=args.docs,
        out=args.out,
        defect_out=args.defect_out,
        model_id=args.model_id,
        device=args.device,
        chunk_size=args.chunk_size,
        max_chunks_per_doc=args.max_chunks_per_doc,
        n_questions_per_chunk=args.n_questions_per_chunk,
        unanswerable_per_doc=args.unanswerable_per_doc,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
