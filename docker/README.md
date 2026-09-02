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
| `kanana` | Kanana-nano-2.1B | 4.2GB | 0.35 |
| `exaone` | EXAONE-3.5-2.4B | 4.8GB | 0.38 |
| `qwen` | Qwen2.5-3B | 6.2GB | 0.45 |
| `kanana8b` | Kanana 1.5 8B | 16GB | 0.78 |
| `luxia8b` | Ko-Llama3-Luxia-8B | 16GB | 0.78 |

작은 셋만 합쳐도 15GB 라 8B 가 낄 자리가 없고, 8B 하나면 20GB 를 꽉 채운다.
그래서 상주 대신 교체다. 교체는 30초~2분, 처음 받는 모델은 몇 분 더 걸린다.

`mem` 은 **GPU 전체 대비 비율**이다(남은 양이 아니다). 0.83 을 넘기면 TEI 가
죽는다. OOM 이 나면 낮추고, KV 캐시가 모자라 느리면 올린다 —
`config/model_config.py` 에서 모델별로 잡는다.

`luxia8b` 는 베이스 모델이라 채팅 템플릿이 없다. `args` 로 Llama-3 템플릿을
씌워 뒀다. 답이 이상하면 이 모델부터 의심할 것.

### 확인

```bash
python scripts/check_gen_store.py --gen kanana qwen
```

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
