# 서버 띄우기

VM 한 대에 임베딩·리랭커·생성 세 가지를 다 올린다. L4 24GB 면 넉넉하다.

| 서버 | 포트 | 모델 | GPU 메모리 |
|---|---|---|---|
| 임베딩 (TEI) | **8085** | `dragonkue/BGE-m3-ko` | 약 1.3GB |
| 리랭커 (TEI) | **8086** | `dragonkue/bge-reranker-v2-m3-ko` | 약 1.3GB |
| SPLADE (TEI) | **8084** | `telepix/PIXIE-Splade-v1.5` | 약 1GB |
| 생성 (SGLang) | **8087** | 고른 모델 하나 | 4~19GB |

## 포트를 왜 8085부터 쓰나

8000번대 앞쪽을 다른 서비스가 쓰고 있던 시절에 밀려난 자리다. 지금은 비어 있지만
옮길 이유가 없어서 그대로 둔다 — 8085 임베딩, 8086 리랭커, 8087 생성, 8010 API.

띄우기 전에 빈 포트인지 확인한다. 다른 컨테이너가 이미 물고 있으면 조용히 못 뜬다.

```bash
sudo ss -tlnp | grep -E ':(8085|8086|8087|8010) '
```

아무것도 안 나오면 비어 있는 것이다. 이미 쓰이고 있으면 compose 의 포트와
`.env` 의 `TEI_EMBED_URL` / `TEI_RERANK_URL` / `SGLANG_URL` 을 같이 바꾼다.

## 처음 한 번

도커와 NVIDIA 컨테이너 툴킷이 필요하다. 이미 깔려 있으면 건너뛴다.

```bash
# 도커
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER      # 이 줄 실행 후 로그아웃/로그인 한 번

# GPU 를 컨테이너에서 쓰려면
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 확인 — GPU 가 보이면 성공
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

## 임베딩·리랭커

```bash
cd ~/proj-bid-mate/docker
docker compose up -d
docker compose logs -f          # 처음엔 모델 내려받느라 3~5분
```

`Ready` 가 뜨면 확인:

```bash
curl -s localhost:8085/info | python3 -m json.tool
curl -s localhost:8086/info | python3 -m json.tool

# 임베딩 — 앞부분만 보기. -s 를 빼면 진행률 표시와 섞여 지저분하다.
curl -s localhost:8085/embed -X POST -H 'Content-Type: application/json' \
  -d '{"inputs":"과업기간은 계약체결일로부터 120일"}' | head -c 120; echo

curl -s localhost:8086/rerank -X POST -H 'Content-Type: application/json' \
  -d '{"query":"과업기간","texts":["과업기간 120일","점심 메뉴는 김치찌개"]}'
```

리랭커가 첫 번째 문장에 훨씬 높은 점수를 주면 정상이다.

> `curl: Failed writing body` 는 **오류가 아니다.** `head -c 120` 이 120자를 읽고
> 파이프를 닫았는데 curl 이 나머지를 계속 쓰려다 나는 소리다. 위처럼 `-s` 를 붙이면
> 조용해진다. 벡터 값이 보였으면 성공한 것이다.

## 생성 (SGLang) — 한 번에 한 모델만

`docker compose up -d` 에 같이 딸려 온다. 기본 모델은 Kanana-nano-2.1B 다.

```bash
curl -s localhost:8087/v1/models | python3 -m json.tool
```

**모델을 바꾸는 건 컨테이너를 다시 만드는 일이다.**

```bash
GEN_MODEL=Qwen/Qwen2.5-3B-Instruct GEN_MEM=0.45 \
  docker compose up -d --force-recreate gen
```

이 한 줄을 `src/models/sglang.py` 의 `ensure()` 가 대신 친다. UI 에서 모델을
고르면 `/ask` 가 알아서 갈아끼우므로 손으로 칠 일은 디버깅할 때뿐이다.

### 왜 하나만 올리나

L4 24GB 에 TEI 셋이 약 4GB 를 붙박이로 쓴다. 생성용은 20GB 다.

| 키 | 모델 | 가중치(fp16) | `mem` |
|---|---|---|---|
| `exaone` | EXAONE-3.5-2.4B | 4.8GB | 0.38 |
| `qwen` | Qwen2.5-3B (기본) | 6.2GB | 0.45 |
| `kanana8b` | Kanana 1.5 8B | 16GB | 0.78 |
| `luxia8b` | Ko-Llama3-Luxia-8B | 16GB | 0.78 |

작은 것들만 합쳐도 8B 가 낄 자리가 없고, 8B 하나면 20GB 를 꽉 채운다.
그래서 상주 대신 교체다. 교체는 30초~2분, 처음 받는 모델은 몇 분 더 걸린다.

`mem` 은 **GPU 전체 대비 비율**이다(남은 양이 아니다). 0.83 을 넘기면 TEI 가
죽는다. OOM 이 나면 낮추고, KV 캐시가 모자라 느리면 올린다 —
`config/model_config.py` 에서 모델별로 잡는다.

`luxia8b` 는 베이스 모델이라 채팅 템플릿이 없다. `args` 로 Llama-3 템플릿을
씌워 뒀다. 답이 이상하면 이 모델부터 의심할 것.

### Kanana-nano-2.1B 은 빠졌다 (2026-09-02)

이 이미지의 transformers 가 launch 단계에서 거부한다.

```
ValueError: The hidden size (1792) is not a multiple of
            the number of attention heads (24)
```

config 에 `head_dim: 128` 을 명시해서 `hidden_size / num_heads` 와 일부러 다르게
잡은 모델인데(1792 vs 24×128=3072), 검증기가 `head_dim` 을 안 본다.
**우리 설정 문제가 아니다.** 나머지 넷은 `hidden/heads` 가 딱 떨어져서 안 걸린다
(Qwen 2048/16, EXAONE 2560/32, 8B 둘 다 4096/32).

되살리려면 이미지 태그를 바꿔서 **실제로 떠야** 확인된 것이다. 목록에만 되돌리면
사용자가 고른 뒤 에러를 보게 된다.

### 크래시는 안 기다린다

`restart: unless-stopped` 라 launch 가 실패하면 조용히 무한 재시도한다.
`ensure()` 가 컨테이너 재시작 횟수를 같이 보다가 0 을 넘으면 바로 나오면서
로그 꼬리를 에러에 붙인다. 안 그러면 15분(SWAP_TIMEOUT)을 꽉 채우고 나서야 안다.

### 확인

```bash
python scripts/check_gen_store.py --gen qwen exaone
```

## API 를 재부팅에도 살려두기

`docker/bidmate-api.service` 를 그대로 쓴다.

```bash
sudo cp docker/bidmate-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now bidmate-api
journalctl -u bidmate-api -f
```

**가상환경은 activate 하지 않는다.** activate 는 PATH 를 손보는 게 전부라
유닛에서 할 일이 없다. `.venv/bin/uvicorn` 을 절대경로로 부르면 그 안의
파이썬과 패키지를 그대로 쓴다 — uv 로 만들었든 pip 로 만들었든 같다.

도커 컨테이너는 `restart: unless-stopped` 라 알아서 돌아온다. 이 유닛은
`After=docker.service` 라서 API 가 도커보다 늦게 뜬다 — 모델 교체가
`docker compose` 를 부르기 때문이다.

**uvicorn 을 돌리는 계정이 `docker` 그룹에 있어야 한다.**

```bash
groups | grep docker || sudo usermod -aG docker $USER   # 후 재로그인
```

없으면 UI 에서 모델을 바꿀 때 permission denied 로 죽는다.

### OOM 이 반복되면

`Restart=always` 라 다시 올라오지만, 계속 죽으면 원인은 호스트 RAM 이다.
TEI·SGLang 은 GPU 를 쓰지만 uvicorn 프로세스는 FAISS·청크·BM25 를 **호스트
메모리**에 올리고 이 VM 은 16GB 다. `journalctl -u bidmate-api | grep -i oom`
에 뭔가 보이면 워커를 늘리지 말고 `POOL` 이나 청크 수부터 줄인다.

## 전체 세팅 — 무엇이 자동이고 무엇이 손인가

| | 부팅 후 자동? | 왜 |
|---|---|---|
| TEI 임베딩·SPLADE·리랭커 | **O** | compose 의 `restart: unless-stopped` |
| SGLang 생성 | **O** | 같음. 기본 모델(Qwen2.5-3B)로 뜬다 |
| FastAPI :8010 | **O** | `bidmate-api.service` (`enable --now` 했을 때) |
| 크롤링 | **O** | crontab |
| 전처리·색인 | **X** | 증분 경로가 없다. 아래 참고 |
| Vercel | 무관 | VM 과 독립. `RFP_API` 만 맞으면 된다 |

### 처음 한 번

```bash
cd ~/rfp-rag-system/docker && sudo docker compose up -d
sudo cp bidmate-api.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now bidmate-api
crontab crontab.txt
groups | grep docker || sudo usermod -aG docker $USER   # 후 재로그인
```

`enable` 이 부팅 자동을 켠다. `start` 만 하면 재부팅 후 안 뜬다.

### 확인은 한 곳에서

```bash
curl -s localhost:8010/health | python3 -m json.tool
```

```json
{ "ok": true, "embedder": "dragonkue/snowflake-arctic-embed-l-v2.0-ko",
  "reranker": "dragonkue/bge-reranker-v2-m3-ko",
  "generator": "Qwen/Qwen2.5-3B-Instruct",
  "store": "faiss", "index": "cleaned_documents_v8__pipeline_1500_250__tei",
  "chunks": "cleaned_documents_v8__pipeline_1500_250" }
```

`null` 인 칸이 끊긴 칸이다. **`store`/`index` 를 같이 주는 이유** — 배포 사고의
절반이 "어느 코퍼스를 보고 있는지" 가 어긋난 것이었다.

### 재부팅이 진짜 되는지는 재부팅해봐야 안다

```bash
sudo reboot
# 3분 뒤
curl -s localhost:8010/health | python3 -m json.tool
sudo docker compose -f ~/rfp-rag-system/docker/docker-compose.yml ps
```

`systemctl is-enabled docker bidmate-api` 가 둘 다 `enabled` 여야 한다.
기동에 2분 반 걸린다(BM25 형태소 분석 155초) — 바로 안 뜬다고 죽은 게 아니다.

## FAISS 와 LanceDB — 무엇이 다른가

**바꾸는 건 `.env` 한 줄이다.** systemd 유닛은 안 건드린다.

```bash
echo "STORE=lance" >> ~/rfp-rag-system/.env      # 켜기
sudo systemctl restart bidmate-api
```

`settings.load_env()` 가 `config/retrieval.py` 보다 먼저 돌아서 잡힌다.
`os.environ.setdefault` 라 **명령줄이 `.env` 를 덮는다** — 한 번만 되돌리려면
`STORE=faiss uvicorn ...`.

|  | FAISS | LanceDB |
|---|---|---|
| 인덱스 만들기 | `prepare.py --build` | `STORE=lance prepare.py --build` |
| 저장 위치 | `outputs/vectorstore/{이름}/` | `outputs/lancedb/{이름}.lance` |
| 검색 정확도 | — | **겹침 100% · 1위 전부 일치** (실측) |
| 질문당 | 13ms | 75ms (질문 총 3~5초 중 1~2%) |
| RAM | 인덱스를 메모리에 | 디스크에서 읽는다 (인덱스 자체는 37MB라 총량 차이는 거의 없다) |
| 공고 지우기 | 통째로 다시 써야 한다 | `delete_docs()` 한 줄 |
| 새 공고 붙이기 | pickle 을 다시 쓴다 | `add_chunks()` |

**지금은 FAISS 로 둔다.** Lance 의 값어치는 삭제·증분인데 코퍼스가 고정이면
실현되지 않는다. 크롤러가 매일 붓기 시작하는 날 `.env` 한 줄로 바꾼다.

`prepare.py` 가 `STORE` 를 보고 그쪽만 검사한다 — 이름은 양쪽이 같고 폴더만
다르므로, 바꾼 뒤 `prepare.py` 를 한 번 돌리면 없는 쪽을 만들어 준다.

## 노트북에서 쓰기

```python
from bidmate.models import load_embedder, load_reranker, load_llm, check_servers

check_servers()                      # 지금 뭐가 떠 있는지

embedder = load_embedder("tei")
reranker = load_reranker("tei")
llm      = load_llm("vllm")          # SGLang 도 OpenAI 규격이라 그대로 붙는다
```

## 자주 나는 문제

**`Temporary failure in name resolution` / `Could not download model artifacts`**

컨테이너 안에서 `huggingface.co` 를 못 찾은 것이다. **먼저 진짜 DNS 문제인지 확인한다.**

```bash
docker run --rm alpine nslookup huggingface.co
```

**주소가 나오면 — 일시적 실패다.** 두 컨테이너가 동시에 모델을 받으면서 DNS 조회가
몰린 경우가 대부분이다. 그냥 다시 올리면 된다. compose 에 `depends_on` 을 넣어
임베딩이 먼저 뜬 뒤 리랭커가 시작하게 해 뒀다.

```bash
docker compose up -d
docker compose logs -f rerank
```

**주소가 안 나오면 — 진짜 DNS 문제다.** 도커 데몬에 DNS 를 박는다.

```bash
sudo tee /etc/docker/daemon.json > /dev/null <<'EOF'
{ "dns": ["8.8.8.8", "1.1.1.1"] }
EOF
sudo systemctl restart docker
cd ~/proj-bid-mate/docker && docker compose up -d
```

> GCP VM 은 기본 DNS 가 `169.254.169.254`(메타데이터 서버)다. 이게 잘 도는데
> 굳이 8.8.8.8 로 바꾸면 사내망 주소를 못 찾게 될 수 있다. **위 확인이 실패했을 때만** 하자.

**계속 안 되면** — 호스트에서 미리 받아 캐시에 넣어 두고 컨테이너는 그걸 읽게 한다.

```bash
pip install -U "huggingface_hub[cli]"
HF_HOME=~/proj-bid-mate/docker/tei-cache \
  hf download dragonkue/bge-reranker-v2-m3-ko
docker compose up -d rerank
```

한쪽만 실패하는 이유 — 임베딩은 이미 받아서 `tei-cache/` 에 남아 있고, 리랭커만
새로 받으려다 걸린 것이다. 캐시가 있으면 네트워크를 아예 안 탄다.

**`413 Payload Too Large`** — 한 번에 보낸 문장이 너무 많다.
`TEIEmbeddings(batch_size=32)` 로 줄이거나, compose 의
`--max-client-batch-size` 를 키우고 `docker compose up -d` 로 다시 올린다.

**`CUDA out of memory`** — 생성 서버가 메모리를 다 먹었다.
`nvidia-smi` 로 확인하고 `config/model_config.py` 의 그 모델 `mem` 값을 낮춘다.
급하면 `docker compose stop gen` 으로 생성만 내리고 인덱스 작업을 먼저 한다.

**모델을 못 내려받는다** — 게이티드 모델이면 토큰이 필요하다.
compose 의 `environment` 에 `- HF_TOKEN=hf_xxx` 를 추가한다.
`dragonkue` 두 모델은 공개라 필요 없다.

**컨테이너가 계속 재시작한다** — `docker compose logs embed` 로 본다.
GPU 태그가 안 맞는 게 흔한 원인이다. `nvidia-smi` 로 GPU 이름을 확인하고
compose 의 이미지 태그를 맞춘다 (L4 → `89-1.9`).

## 서버를 안 띄우고 개발하기

TEI 없이도 배관은 확인된다.

```python
embedder = load_embedder("fake")     # 가짜 임베딩
reranker = load_reranker("fake")
llm      = load_llm("echo")          # 프롬프트만 되돌려 줌
```

검색 품질은 무의미하지만 부품 조립이 도는지는 알 수 있다.
