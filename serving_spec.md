# 기능 명세 — Serving

> 담당 범위: **수집~생성 전 구간의 서비스 운영(Serving)**.
> 기준일 2026-09-09

---

## 0. 한눈에

```
수집        crawl.py                  나라장터 API → data/raw/*.hwp + data_list.csv
  ↓
전처리      build.py run_pipeline()    hwp/pdf → 본문 2벌(검색용·생성용) + 메타 병합
  ↓
청킹        run_pipeline(chunk=True)   파이프라인이 자른다 (표 원자성 유지)
                                       → outputs/chunks/<CHUNKS>.jsonl 로 직접
  ↓
색인        lance_store.sync_docs()    Dense (증분)
            pieces.search.BM25         Sparse (기동 시 · 형태소 캐시)
  ↓
검색 1단계  retriever.search_notices() 어떤 공고를 볼지 고른다
검색 2단계  retriever.retrieve()       그 공고 안에서 근거를 찾는다
  ↓
생성        generation.generate_answer() / stream_answer()
  ↓
API/UI      api.py  ←  Next.js(Vercel) route handler
```

---

# 1. Serving

## 1.1 파이프라인 — 파일과 메서드

### ① 수집 — `src/crawl.py`

| 메서드 | 하는 일 |
|---|---|
| `crawl(hours=None, days=None)` | 나라장터 API 를 훑어 새 공고를 받는다 |
| `fetch_page()` | 목록 API 한 쪽 |
| `pick_attachment()` | **어느 첨부가 제안요청서인지 API 가 안 알려준다** — 이름으로 고른다 |
| `download()` | 첨부를 `data/raw/` 로 |
| `to_row()` | API 응답 → CSV 한 줄. `배정예산`/`추정가격` 구분을 여기서 기록 |
| `append_csv()` | `data/metadata/data_list.csv` 에 덧쓴다 |
| `existing_ids()` / `superseded()` | 중복·정정공고 판정. **내용 비교 안 한다 — API 가 알려준다** |
| `upgrade_header()` | CSV 열이 늘 때 한 번 도는 이관 |
| `selftest()` | `python src/crawl.py --selftest` |

산출: `data/raw/*.hwp|pdf` · `data/metadata/data_list.csv`

### ② 전처리 — `src/preprocessing/rfp/build.py`

| 메서드 | 하는 일 |
|---|---|
| `run_pipeline()` | 추출 → 정제 → CSV 병합 → `cleaned_documents.jsonl` |
| `_cached_rows(path)` | **증분의 핵심.** 이미 만든 결과를 읽어 재사용. CSV 열은 떼어낸다 |
| `_cache_cutoff()` | 캐시가 원본보다 오래됐으면 무시 |
| `write_jsonl()` | 본문 2벌(`page_content`, `page_content_for_generation`) 기록 |
| `_build_report()` | 병합 성공률. **90% 아래면 멈춘다** |

추출기: `src/preprocessing/rfp/hwp.py`, `hwpx.py`, `pdf.py`
필드 표: `src/preprocessing/fields.py` (`FIELDS`, `CSV_COLUMNS`, `normalize_columns`)

### ③ 청킹 — `src/chunking.py`

| 메서드 | 하는 일 |
|---|---|
| `split_recursive(size=1500, overlap=250)` | **채택된 기법** |
| `drop_toc_chunks()` | 목차 청크 제거 |
| `save_chunks(name)` / `load_chunks(name)` | `outputs/chunks/{name}.jsonl` |
| `chunk_stats()` | 길이 분포 점검 |

### ④ 색인

| 대상 | 파일 · 메서드 | 증분 |
|---|---|---|
| Dense | `src/lance_store.py :: sync_docs(name, docs, embedder)` | **가능** — 새 공고만 임베딩, 바뀐 건 지우고 다시, 빠진 건 삭제 |
| Dense(구) | `src/vectorstore.py` (FAISS) | 부분 삭제·필터 불가. `STORE` 로 고른다 |
| Sparse | `src/pieces/search.py :: BM25.__init__` | **형태소 캐시로 가능** — 아래 2.3 |

`lance_store` 보조: `_doc_hashes()`(doc_id+본문+메타 해시), `add_chunks()`, `delete_docs()`, `_write_stamp()`

### ⑤ 조율 — `scripts/retrieval/prepare.py`

지금 설정(`.env`)에 필요한 파생물이 다 있는지 보고 **없거나 낡은 것만** 만든다.
크론이 매일 부르는 것도 이 한 줄이다.

```
python scripts/retrieval/prepare.py --build --service
```

| 단계 | 무엇을 보나 | 없거나 낡으면 |
|---|---|---|
| [1] 전처리본 | `data/raw` 의 mtime 이 더 새로운가 | `run_pipeline()` |
| [2] 청크 | `outputs/chunks/<CHUNKS>.jsonl` 이 있나 | `run_pipeline()` (같이 만들어진다) |
| [3] 인덱스 | `chunk_signature` 가 색인의 것과 같은가 | `sync_docs()` |
| [4] 평가 세트 | 도장이 청크와 맞는가 | 다시 만든다 (`--service` 면 건너뛴다) |

**청크는 파이프라인이 직접 쓴다.** `run_pipeline(enable_chunk_output=True)` 가
`settings.CHUNKS / f"{cfg.CHUNKS}.jsonl"` 로 내보내므로, [1] 이 돌면 [2] 도 같이
채워진다. 중간에 파일을 옮기거나 이름을 바꾸는 사람 손이 없다.

`chunking.py` 로 자르지 않는 이유는 **표 원자성** 때문이다. 우리가 다시 자르면
표의 헤더와 값이 다른 청크로 갈린다. 이름에 `__pipeline` 을 박아 두는 것도 같은
이유다 — `recursive` 라고 쓰면 나중에 `chunking.py` 로 다시 만들 수 있다고
착각한다.

#### 예외 경로 — `scripts/retrieval/ingest.py`

외부에서 만들어진 청크 파일을 우리 이름으로 가져올 때만 쓴다. 전처리팀이 파일을
넘겨주던 시절의 반입 도구고, **크론 경로에는 없다.**

```
python scripts/retrieval/ingest.py --src <외부 jsonl>   # 옮기기만
python scripts/retrieval/ingest.py --run                # 파이프라인부터 (prepare 와 같은 일)
```

### ⑥ 생성 — `src/generation.py`

| 메서드 | 하는 일 |
|---|---|
| `generate_answer(model_key, query, context, history)` | 한 번에 받는다. 예외를 안 던지고 고정 스키마로 |
| `stream_answer(...)` | 토큰 단위로 흘린다. 막히면 위 함수로 폴백 |
| `_build_messages()` | system + 히스토리 + `[컨텍스트]/[질문]` |
| `_run_openai()` / `_run_sglang()` / `_stream_chat()` | provider 별 호출 |
| `ask(model_key, system, user)` | 채점용 범용 호출 (프롬프트 틀 없이) |

모델 표: `config/model_config.py :: MODEL_CONFIGS`
`mini`(gpt-5-mini/medium) · `mini-fast`(gpt-5-mini/minimal) · `nano` · `exaone` · `qwen`

## 1.2 API — `src/api.py`

| 메서드 | 경로 | 하는 일 |
|---|---|---|
| `search` | `POST /search` | 1단계. `retriever.search_notices` |
| `ask` | `POST /ask` | 2단계+생성. 한 번에 응답 |
| `ask_stream` | `POST /ask/stream` | 같은 답을 NDJSON 으로. `meta`→`delta`→`done` |
| `reload_index` | `POST /reload` | **새 청크를 무중단 반영. 재시작을 대신한다** |
| `notice_one` | `GET /notice/{doc_id}` | 공고 한 건 |
| `file` | `GET /file/{doc_id}` | 원본 RFP 내려받기 |
| `models` | `GET /models` | 드롭다운 목록 (`ready`, `usd_per_call`, `effort`) |
| `evalsets` / `eval_upload` | `GET /evalsets` · `POST /eval/upload` | 평가 세트 |
| `eval_start` / `eval_list` / `eval_status` / `eval_cancel` | `POST GET /eval …` | 평가 작업 |
| `health` | `GET /health` | **유일하게 토큰 없이 열린다** |

**인증** `guard` 미들웨어. `hmac.compare_digest` 로 `x-api-token` 비교.
`API_TOKEN` 이 비어 있으면 **검사하지 않는다** — 막으면 토큰을 넣기 전에 배포한
순간 화면이 통째로 죽는다. 대신 뜰 때 경고하고 `/health` 에 `auth: false` 로 드러낸다.

**UI 쪽** `app/api/[...path]/route.ts` 가 **서버에서** 토큰을 붙인다.
`RFP_API`·`RFP_TOKEN` 은 `NEXT_PUBLIC_` 이 아니다 — 브라우저로 가면 토큰이 아니다.

## 1.3 무중단 갱신

```
docker/refresh.sh   crawl.py → prepare.py --build --service → POST /reload
                    12,15,18,21,0시 (--hours 4) · 03:30 (--days 2)
                    flock 으로 한 번에 하나만
```

`src/retriever.py :: reload()`

```python
chunks = chunking.load_chunks(CHUNKS)  # 디스크에서 새로
BM25(chunks, k=POOL)  # ← 먼저 짓는다 (_BM25_CACHE 에 들어간다)
_load.cache_clear()  # ← 그 다음에 비운다
_store.cache_clear()
```

**데운 다음에 비운다.** 반대로 하면 직후 요청 하나가 색인 짓는 값을 혼자 낸다.

| | 값 |
|---|---|
| 걸리는 시간 | 1.9초 |
| 인덱스 RAM | 487MB (교체 순간 두 벌 974MB) |
| 형태소 캐시 | `outputs/vectorstore/kiwi_tokens.json.gz` 4.6MB |

**한계** `CHUNKS` 이름이 바뀌면 `/reload` 로 안 된다(import 때 읽은 값).
코퍼스 버전을 올릴 때만 재시작.

## 1.4 평가 실행 — `src/evalrun.py`

| 메서드 | 하는 일 |
|---|---|
| `start(evalset, model, judge, ...)` | 작업을 만들고 스레드에서 돌린다. 작업번호를 즉시 반환 |
| `_run(job)` | 발췌 → 답변 → 채점 세 단계 |
| `_stream(job, args, step)` | 자식 프로세스를 띄우고 출력을 로그에 붙인다 |
| `normalize(rows)` | 업로드된 세트의 필드명·파일명→doc_id 변환 |
| `cancel(job_id)` | 발췌는 예외로, 자식 프로세스는 `terminate()` 로 |
| `read` / `listing` / `estimate` | 상태 조회 · 비용 추산 |
| `sweep()` | 서버 기동 시 1회. 돌던 중 죽은 작업을 `interrupted` 로 |
| `_drop_upload()` | 업로드본 삭제. **접두어와 경로를 둘 다 확인** |

**상태는 파일에 있다** — `outputs/eval_runs/<번호>.json`.
화면을 떠나도, 새로고침해도, API 가 재시작돼도 이어 본다. Vercel 쪽에는 상태가 없다.

## 1.5 자체 점검

| 스크립트 | 무엇을 |
|---|---|
| `check_metadata.py` | 원본 ↔ CSV 대조 (전처리 **전에**) |
| `check_fields.py` | 필드가 전처리 → 청크 → 검색단까지 오나 |
| `check_incremental.py` | 증분 경로 (전처리 캐시 · `sync_docs`) |
| `check_tokencache.py` | 형태소 캐시 적중 · 토크나이저 교체 시 폐기 |
| `check_bm25.py` | BM25 시간·RAM 을 단계별로 |
| `check_chunks.py` | 파이프라인이 자른 청크가 쓸 수 있는 상태인가 (길이·마크업 잔존) |
| `check_budget.py` | CSV 금액이 본문에도 있나 |
| `check_run.py` | 이 실행이 새 코드로 돈 것인가 |

## 1.6 배포와 인프라

```
Vercel (Next.js)  →  route handler(토큰 부착)  →  VM :8010 FastAPI
                                                    ├ :8085 TEI embed   docker
                                                    ├ :8086 TEI rerank  docker
                                                    └ :8087 SGLang gen  docker (모델 1개씩 교체)
```

- VM L4 24GB · RAM 16GB. **벽은 GPU 가 아니라 디스크**(모델 가중치 43GB)
- SGLang 은 모델을 하나만 올린다 → 고르면 컨테이너 교체(1~2분). UI 가 `ready`로 표시
- 시나리오 A(VM 생성) / B(OpenAI)는 **환경변수로 배포 시점에** 정한다.
  원본 RFP 가 NDA 라 고객 선택으로 노출하지 않는다

## 1.7 설정 한 곳

| 파일 | 무엇 |
|---|---|
| `.env` | `CHUNKS` · `POOL` · `NOTICE_POOL` · `TOP_K` · `STORE` · `API_TOKEN` · `OPENAI_API_KEY` |
| `config/retrieval.py` | 위 값을 읽어 `chunk_name()` · `index_name()` 로 |
| `config/settings.py` | 경로 (`DATA`, `CHUNKS`, `VECTORSTORE`, `EVAL_RESULTS` …) |
| `config/model_config.py` | `MODEL_CONFIGS` |

**상수를 코드에 다시 적지 않는다.** 두 번 데였다 — 2단계 `pool 80`, 1단계 `pool 200`
둘 다 폐기된 측정에서 온 값이 코드에 굳어 있었다.

---
