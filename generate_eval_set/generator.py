import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """당신은 RFP(입찰제안요청서) 문서를 분석하여 RAG 시스템 평가용 질문-답변 쌍을 구축하는 전문가입니다.
주어진 본문에 명시된 사실만을 근거로 구체적이고 명확한 질문과 정답을 작성하세요.

반드시 다음 JSON 형식으로만 응답하세요:
```json
{
  "qa_pairs": [
    {
      "question": "구체적인 질문 내용",
      "answer": "정답 내용",
      "evidence_text": "답변의 근거가 되는 원문 문장 발췌 (원문 그대로 복사)",
      "question_type": "factual"
    }
  ]
}
```"""

USER_PROMPT_TEMPLATE = """다음은 RFP 문서 "{doc_id}"의 본문 일부입니다.
이 본문 내용만을 바탕으로 {n_questions}개의 질문-답변 쌍을 JSON 형식으로 작성하세요.

--- 본문 내용 ---
{clean_text}
--- 본문 끝 ---"""


@dataclass
class EvalItem:
    doc_id: str
    question: str
    answer: str
    evidence_text: str
    question_type: str


class QAEvalGenerator:
    def __init__(
        self, model_id: str = "Qwen/Qwen2.5-3B-Instruct", device: str = "cuda:0"
    ):
        self.device = device
        logger.info(f"Loading tokenizer & model: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map=self.device
        )
        logger.info("Model loaded successfully on GPU!")

    def _extract_json_block(self, text: str) -> dict:
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        raw_json = match.group(1) if match else text
        return json.loads(raw_json.strip())

    def generate_qa_chunk(
        self, doc_id: str, text_chunk: str, n_questions: int = 3
    ) -> list[EvalItem]:
        if not text_chunk.strip():
            return []

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(
                    doc_id=doc_id, n_questions=n_questions, clean_text=text_chunk
                ),
            },
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(  # type: ignore
                **inputs,
                max_new_tokens=1024,
                temperature=0.2,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        try:
            data = self._extract_json_block(response_text)
            return [
                EvalItem(
                    doc_id=doc_id,
                    question=qa["question"],
                    answer=qa["answer"],
                    evidence_text=qa["evidence_text"],
                    question_type=qa.get("question_type", "factual"),
                )
                for qa in data.get("qa_pairs", [])
                if "question" in qa and "answer" in qa and "evidence_text" in qa
            ]
        except Exception as e:  # noqa: BLE001
            logger.warning(f"JSON 파싱 실패 건너뜀 ({doc_id}): {e}")
            return []

    def generate_from_documents(
        self,
        documents: list[dict],
        n_questions_per_chunk: int = 3,
        chunk_size: int = 2500,
        max_chunks_per_doc: int = 2,
    ) -> list[dict]:
        all_results = []
        for doc in tqdm(documents, desc="Golden QA 생성 중"):
            metadata = doc.get("metadata", {})
            doc_id = (
                metadata.get("doc_id")
                or metadata.get("source")
                or metadata.get("filename", "unknown")
            )
            full_text = doc.get("page_content", "")

            if not full_text.strip():
                continue

            chunks = [
                full_text[i : i + chunk_size]
                for i in range(0, len(full_text), chunk_size)
            ]
            chunks = chunks[:max_chunks_per_doc]

            for chunk in chunks:
                items = self.generate_qa_chunk(
                    doc_id, chunk, n_questions=n_questions_per_chunk
                )
                all_results.extend([asdict(item) for item in items])

        return all_results


def load_documents(input_path: Path) -> list[dict]:
    documents = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                documents.append(json.loads(line))
    logger.info(f"Loaded {len(documents)} documents from {input_path}")
    return documents


def save_jsonl(items: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(items)} items to {output_path}")
