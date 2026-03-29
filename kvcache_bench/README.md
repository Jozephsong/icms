# KV Cache Tier Benchmark

ShareGPT 실제 대화 데이터를 입력으로 사용하는 **vLLM + LMCache KV 캐시 계층 벤치마크**입니다.
GPU, CPU RAM, SSD 4가지 메모리 계층 시나리오와 0%~100% hit-rate sweep을 통해
TTFT, 처리량, LMCache per-tier 토큰 카운트를 측정합니다.

---

## 목차

1. [아키텍처](#아키텍처)
2. [시나리오](#시나리오)
3. [측정 메트릭](#측정-메트릭)
4. [의존성 설치](#의존성-설치)
5. [ShareGPT 데이터 준비](#sharegpt-데이터-준비)
6. [빠른 시작](#빠른-시작)
7. [CLI 파라미터 전체 참조](#cli-파라미터-전체-참조)
8. [run_bench.sh 사용법](#run_benchsh-사용법)
9. [hit-rate 제어 방식](#hit-rate-제어-방식)
10. [Prometheus 메트릭 수집 방식](#prometheus-메트릭-수집-방식)
11. [출력 결과 형식](#출력-결과-형식)
12. [외부 서버 연결 모드](#외부-서버-연결-모드)
13. [로컬 개발 / 저사양 GPU 환경](#로컬-개발--저사양-gpu-환경)
14. [알려진 제약사항](#알려진-제약사항)

---

## 아키텍처

```
bench.py (클라이언트 프로세스)
  │
  ├─ 시나리오별 vllm serve 서브프로세스 시작
  │    └─ LMCache 환경변수 주입 (LMCACHE_USE_EXPERIMENTAL, LMCACHE_CHUNK_SIZE, …)
  │    └─ VLLM_SERVER_DEV_MODE=1 → /reset_prefix_cache 엔드포인트 활성화
  │
  ├─ hit_rate별 루프
  │    1. Prewarm  → OpenAI Streaming API (asyncio.gather 동시 전송)
  │    2. GPU 캐시 초기화 → POST /reset_prefix_cache  (offload 시나리오만)
  │    3. LMCache settle 대기  (offload_wait_s)
  │    4. Prometheus 스냅샷 (before)  ← http://localhost:9090/metrics
  │    5. Benchmark 배치 전송  → TTFT + 생성 토큰 수 수집
  │    6. Prometheus flush 대기  (stats_flush_wait_s, default 12s)
  │    7. Prometheus 스냅샷 (after) + delta 계산
  │    8. BenchResult 기록
  │    9. GPU 캐시 초기화 (다음 iteration 준비)
  │
  └─ 결과 저장 → bench_results/bench_YYYYMMDD_HHMMSS.{json,csv}
```

**서버-클라이언트 분리 이유**: vLLM V1은 모델 워커를 별도 subprocess에서 실행합니다.
`LMCStatsMonitor` 싱글톤은 해당 subprocess 안에만 존재하므로 클라이언트 프로세스에서
직접 접근할 수 없습니다. 대신 LMCache V1 multiprocess server가 포트 9090에
Prometheus HTTP 서버를 노출하므로, 이 엔드포인트를 scraping하여 메트릭을 수집합니다.

---

## 시나리오

| 이름 | 설명 | LMCache 활성 |
|---|---|---|
| `gpu_only` | vLLM GPU prefix cache만 사용 (순수 baseline) | No — 오버헤드 없음 |
| `gpu_cpu` | GPU + CPU RAM offload (LMCache L1=CPU) | Yes |
| `gpu_ssd` | GPU + SSD offload (LMCache L1=SSD) | Yes |
| `gpu_cpu_ssd` | GPU + CPU + SSD 3계층 (CPU=L2, SSD=L3) | Yes |

> `gpu_only`는 `--kv-transfer-config` 없이 순수 vLLM으로 실행됩니다.
> LMCache 커넥터가 로드되지 않으므로 공정한 baseline이 됩니다.

---

## 측정 메트릭

### 클라이언트 측 (항상 수집)

| 컬럼 | 설명 |
|---|---|
| `ttft_mean_s` | TTFT 평균 (초) — 요청 시작 → 첫 스트리밍 청크 |
| `ttft_p50_s` | TTFT 중앙값 |
| `ttft_p90_s` | TTFT p90 |
| `ttft_p99_s` | TTFT p99 |
| `total_gen_tokens` | 배치 전체 생성 토큰 수 |
| `gen_tps` | 생성 처리량 (tokens/s) = total_gen_tokens / wall_s |
| `wall_ms` | 배치 전체 wall time (ms) |

### LMCache Prometheus 메트릭 (offload 시나리오)

| 컬럼 | 설명 |
|---|---|
| `tok_requested` | LMCache에 조회 요청된 총 토큰 수 (counter delta) |
| `tok_hit` | LMCache에서 hit된 토큰 수 (counter delta) |
| `tok_stored` | LMCache에 저장된 토큰 수 (counter delta) |
| `tok_vllm_hit` | vLLM GPU prefix cache hit 토큰 수 (counter delta) |
| `tok_prompt` | 처리된 총 prompt 토큰 수 (counter delta) |
| `retrieve_hit_rate` | `tok_hit / tok_requested` — 실제 토큰 단위 hit rate |
| `avg_retrieve_s` | LMCache retrieve 평균 지연시간 (s) |
| `avg_store_s` | LMCache store 평균 지연시간 (s) |
| `avg_retrieve_tps` | LMCache retrieve 처리량 (tokens/s) |
| `avg_store_tps` | LMCache store 처리량 (tokens/s) |
| `avg_retr_to_gpu_s` | CPU/SSD → GPU 전송 평균 시간 (s) |
| `avg_store_from_gpu_s` | GPU → CPU/SSD 저장 평균 시간 (s) |
| `cpu_bytes` | CPU RAM 캐시 점유량 (bytes, 최신 gauge) |
| `disk_bytes` | SSD 캐시 점유량 (bytes, 최신 gauge) |

> `retrieve_hit_rate`는 파라미터 `--hit-rates`와 다를 수 있습니다.
> `--hit-rates`는 "warm pool에서 가져올 요청 비율"이고, `retrieve_hit_rate`는
> LMCache가 실제로 청크 단위로 hit한 토큰 비율입니다. 청크 정렬, 프롬프트 길이 분포에 따라 차이가 납니다.

---

## 의존성 설치

```bash
pip install vllm lmcache openai requests numpy
```

vLLM과 LMCache 버전 호환성은 각 프로젝트의 설치 가이드를 따르십시오.
LMCache V1 (experimental) 기준으로 작성되었습니다.

---

## ShareGPT 데이터 준비

[ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) 데이터셋을 사용합니다.

```bash
# Hugging Face에서 다운로드
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

# 또는 datasets 라이브러리
python -c "
from datasets import load_dataset
ds = load_dataset('anon8231489123/ShareGPT_Vicuna_unfiltered', data_files='ShareGPT_V3_unfiltered_cleaned_split.json')
import json; json.dump(list(ds['train']), open('ShareGPT.json','w'))
"
```

파일 형식은 아래 구조의 JSON 배열이어야 합니다:

```json
[
  {
    "conversations": [
      {"from": "human", "value": "질문 내용"},
      {"from": "gpt",   "value": "답변 내용"}
    ]
  }
]
```

`from` 필드 허용 값: `human`, `user`, `gpt`, `chatgpt`, `bing`, `bard`, `assistant`

---

## 빠른 시작

### 1. 전체 시나리오 실행 (권장)

```bash
python bench.py \
  --data /data/ShareGPT.json \
  --scenario all \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct
```

4개 시나리오 × 5개 hit-rate = 20회 벤치마크가 순차 실행됩니다.
각 시나리오마다 vLLM 서버가 한 번 시작되고, hit-rate 변경 시에는 `/reset_prefix_cache`로
GPU 캐시만 초기화합니다 (서버 재시작 없음).

### 2. 단일 시나리오

```bash
python bench.py \
  --data /data/ShareGPT.json \
  --scenario gpu_cpu \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --num-requests 16 \
  --max-output-tokens 128 \
  --chunk-size 256 \
  --max-cpu-gb 20.0
```

### 3. hit-rate 커스텀 지정

```bash
python bench.py \
  --data /data/ShareGPT.json \
  --scenario gpu_cpu_ssd \
  --hit-rates 0.0 0.5 1.0
```

### 4. run_bench.sh 사용

```bash
DATA=/data/ShareGPT.json \
SCENARIO=gpu_cpu \
NUM_REQUESTS=16 \
bash run_bench.sh
```

---

## CLI 파라미터 전체 참조

### Data

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `--data` | *필수* | ShareGPT JSON 파일 경로 |
| `--max-chars` | `4000` | 대화당 최대 문자 수. 이 길이를 초과하는 대화는 필터링됨 |
| `--data-seed` | `42` | 데이터 셔플 랜덤 시드 |

### Scenario

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `--scenario` | *필수* | `gpu_only` / `gpu_cpu` / `gpu_ssd` / `gpu_cpu_ssd` / `all` |

### vLLM Server

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `--model` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | HuggingFace 모델 ID |
| `--port` | `8000` | vLLM 서버 포트 |
| `--gpu-mem` | `0.5` | `gpu_memory_utilization` (0.0 ~ 1.0) |
| `--max-model-len` | `4096` | 최대 시퀀스 길이 (tokens) |
| `--num-gpu-blocks` | *자동* | GPU KV 블록 수 직접 지정 (생략 시 vLLM 자동 계산) |
| `--startup-timeout` | `300` | 서버 ready 대기 최대 시간 (초) |
| `--server-url` | *없음* | 외부 실행 중인 서버 URL. 지정 시 서버 라이프사이클 관리 생략. 예: `http://localhost:8000` |

### Benchmark

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `--num-requests` | `8` | hit-rate 당 benchmark 배치 크기 |
| `--max-output-tokens` | `64` | 요청당 최대 생성 토큰 수 |
| `--hit-rates` | `0.0 0.25 0.50 0.75 1.0` | hit-rate sweep 값 목록 (공백 구분) |
| `--offload-wait` | `5.0` | GPU 캐시 초기화 후 LMCache async store 완료 대기 시간 (초). SSD 시나리오에서는 `15` 이상 권장 |
| `--stats-flush-wait` | `12.0` | 배치 완료 후 Prometheus flush 대기 시간 (초). LMCache `log_interval`(10s)보다 커야 함 |

### LMCache

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `--chunk-size` | `256` | LMCache KV 청크 크기 (tokens) |
| `--max-cpu-gb` | `10.0` | CPU RAM 캐시 최대 용량 (GB). `gpu_cpu`, `gpu_cpu_ssd`에서 사용 |
| `--max-disk-gb` | `10.0` | SSD 캐시 최대 용량 (GB). `gpu_ssd`, `gpu_cpu_ssd`에서 사용 |
| `--disk-path` | `/tmp/kvcache_bench` | SSD 캐시 기본 디렉토리. 시나리오별 하위 디렉토리가 자동 생성됨 |

### Prometheus

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `--prometheus-url` | `http://localhost:9090/metrics` | LMCache Prometheus 엔드포인트. 빈 문자열(`""`)로 지정 시 scraping 비활성화 |

> LMCache V1 multiprocess server는 vLLM API 포트(8000)와 **별도의 포트(9090)** 에
> Prometheus 서버를 시작합니다. vLLM의 `/metrics`와 다른 엔드포인트입니다.

### Output

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `--output-dir` | `./bench_results` | JSON/CSV 결과 저장 디렉토리 |

### Dev

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `--local-dev` | `False` | WSL2/소비자 GPU 저사양 환경용 workaround 활성화. `--enforce-eager`, CPU KV buffer, `--max-num-batched-tokens` 제한 적용 |

---

## run_bench.sh 사용법

환경변수로 파라미터를 제어하는 shell wrapper입니다.

```bash
# 기본 실행 (DATA 필수)
DATA=/data/ShareGPT.json bash run_bench.sh

# 환경변수로 파라미터 지정
DATA=/data/ShareGPT.json \
SCENARIO=gpu_cpu \
MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct \
NUM_REQUESTS=16 \
MAX_OUTPUT_TOKENS=128 \
MAX_CPU_GB=20.0 \
bash run_bench.sh

# 추가 플래그는 직접 전달
DATA=/data/ShareGPT.json bash run_bench.sh --local-dev

# 외부 서버 연결
SERVER_URL=http://localhost:8000 \
DATA=/data/ShareGPT.json \
bash run_bench.sh --scenario gpu_cpu
```

### run_bench.sh 환경변수 전체 목록

| 환경변수 | 기본값 | 대응 CLI 파라미터 |
|---|---|---|
| `DATA` | *필수* | `--data` |
| `SCENARIO` | `all` | `--scenario` |
| `MODEL` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | `--model` |
| `PORT` | `8000` | `--port` |
| `GPU_MEM` | `0.5` | `--gpu-mem` |
| `MAX_MODEL_LEN` | `4096` | `--max-model-len` |
| `NUM_GPU_BLOCKS` | *(자동)* | `--num-gpu-blocks` |
| `STARTUP_TIMEOUT` | `300` | `--startup-timeout` |
| `SERVER_URL` | *(없음)* | `--server-url` |
| `NUM_REQUESTS` | `8` | `--num-requests` |
| `MAX_OUTPUT_TOKENS` | `64` | `--max-output-tokens` |
| `HIT_RATES` | `0.0 0.25 0.50 0.75 1.0` | `--hit-rates` |
| `MAX_CHARS` | `4000` | `--max-chars` |
| `DATA_SEED` | `42` | `--data-seed` |
| `OFFLOAD_WAIT` | `5.0` | `--offload-wait` |
| `STATS_FLUSH_WAIT` | `12.0` | `--stats-flush-wait` |
| `CHUNK_SIZE` | `256` | `--chunk-size` |
| `MAX_CPU_GB` | `10.0` | `--max-cpu-gb` |
| `MAX_DISK_GB` | `10.0` | `--max-disk-gb` |
| `DISK_PATH` | `/tmp/kvcache_bench` | `--disk-path` |
| `PROMETHEUS_URL` | `http://localhost:9090/metrics` | `--prometheus-url` |
| `OUTPUT_DIR` | `./bench_results` | `--output-dir` |

---

## hit-rate 제어 방식

벤치마크는 **warm pool**과 **cold pool** 두 프롬프트 집합으로 hit-rate를 제어합니다.

```
all_prompts  = load_conversations(total_needed)
warm_pool    = all_prompts[:N]          # N = --num-requests
cold_pool    = all_prompts[N:]          # fresh 프롬프트
```

hit_rate = H 일 때:

```
n_hot  = int(N * H)    # warm_pool에서 가져올 요청 수
n_cold = N - n_hot     # cold_pool에서 가져올 요청 수

1. prewarm: warm_pool[:n_hot] 전송 → 캐시 워밍
2. (offload 시나리오) GPU 캐시 초기화 → 이후 히트는 LMCache에서만 발생
3. benchmark batch: warm_pool[:n_hot] + cold_pool[cold_idx:cold_idx+n_cold] 동시 전송
```

| hit_rate | n_hot | n_cold | 기대 동작 |
|---|---|---|---|
| 0.0 | 0 | N | 전부 캐시 미스 (cold start) |
| 0.25 | N×0.25 | N×0.75 | 25% 요청이 캐시 히트 |
| 0.50 | N×0.5 | N×0.5 | 50% 요청이 캐시 히트 |
| 0.75 | N×0.75 | N×0.25 | 75% 요청이 캐시 히트 |
| 1.0 | N | 0 | 전부 캐시 히트 |

> **주의**: 파라미터 `hit_rate`(요청 단위)와 결과의 `retrieve_hit_rate`(토큰 단위)는
> 다릅니다. LMCache는 청크(chunk_size 토큰) 단위로 캐시를 관리하므로, 프롬프트 길이가
> 청크 경계와 정렬되지 않으면 실제 토큰 hit rate가 낮게 측정됩니다.

---

## Prometheus 메트릭 수집 방식

```
[bench.py]
  prom_before = scrape(http://localhost:9090/metrics)
  ─── benchmark batch 전송 ───
  sleep(stats_flush_wait_s)          # LMCache PrometheusController flush 대기
  prom_after  = scrape(http://localhost:9090/metrics)

  # 카운터가 움직이지 않으면 1회 추가 대기 후 재시도
  if delta(num_requested_tokens) == 0:
      sleep(10s)
      prom_after = scrape(...)

  delta = prom_after - prom_before   # interval 메트릭 계산
```

**왜 별도 대기가 필요한가**: LMCache의 `PrometheusController`는 `log_interval=10s` 주기로
Prometheus 카운터를 업데이트합니다. 배치 완료 직후 바로 scraping하면 직전 flush 이후의
데이터가 반영되지 않을 수 있습니다.

`--stats-flush-wait`(default 12s)는 이 주기보다 크게 설정되어 있으며,
카운터가 여전히 변화 없을 경우 10s를 추가로 기다린 뒤 1회 재시도합니다.

**Prometheus 포트**: LMCache V1은 vLLM API 포트(8000)와 별도로 포트 **9090**에
Prometheus HTTP 서버를 시작합니다. 포트가 충돌하면 `--prometheus-url`로 변경하십시오.

---

## 출력 결과 형식

벤치마크 완료 후 `--output-dir`(default `./bench_results`)에 두 파일이 저장됩니다.

### bench_YYYYMMDD_HHMMSS.json

```json
{
  "timestamp": "20240101_120000",
  "config": { "model": "...", "scenario": "all", ... },
  "results": [
    {
      "scenario": "gpu_only",
      "hit_rate": 0.0,
      "num_requests": 8,
      "wall_ms": 1234.5,
      "ttft_mean_s": 0.123,
      "ttft_p50_s": 0.115,
      "ttft_p90_s": 0.198,
      "ttft_p99_s": 0.241,
      "total_gen_tokens": 512,
      "gen_tps": 415.0,
      "tok_requested": 0,
      "tok_hit": 0,
      "retrieve_hit_rate": 0.0,
      "cpu_bytes": 0,
      "disk_bytes": 0,
      ...
    }
  ]
}
```

### bench_YYYYMMDD_HHMMSS.csv

JSON과 동일한 데이터를 CSV로 저장합니다. 컬럼은 `BenchResult` 필드와 1:1 대응됩니다.

### 터미널 요약 출력 예시

```
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
  BENCHMARK SUMMARY
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
  Scenario       HitRate   TTFT_mean  TTFT_p90    GenTPS  LMC_HitRate    vLLM_hit   LMC_hit    Stored  CPU_MB  Disk_MB  RetrSpd
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  gpu_only           0%      234.1ms   312.5ms     410.2        0.000           0         0         0       0        0        0
  gpu_only         100%       45.3ms    52.1ms     892.4        0.000       8,192         0         0       0        0        0

  gpu_cpu            0%      241.8ms   325.0ms     398.7        0.000           0         0     8,192      12        0        0
  gpu_cpu          100%       38.2ms    44.6ms     978.1        0.943           0     7,714     8,192    4096        0   245000
```

---

## 외부 서버 연결 모드

이미 실행 중인 vLLM+LMCache 서버에 연결할 때 `--server-url`을 사용합니다.

```bash
# 외부 서버 직접 실행 (LMCache 환경변수 수동 설정 필요)
export LMCACHE_USE_EXPERIMENTAL=True
export LMCACHE_CHUNK_SIZE=256
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=10
export VLLM_SERVER_DEV_MODE=1
vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
  --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'

# bench.py를 외부 서버에 연결
python bench.py \
  --data /data/ShareGPT.json \
  --scenario gpu_cpu \
  --server-url http://localhost:8000 \
  --prometheus-url http://localhost:9090/metrics
```

**주의**: `--server-url`과 `--scenario all`을 함께 사용하면 4개 시나리오 레이블이
모두 같은 서버를 측정합니다. 외부 서버 모드는 단일 시나리오 테스트에만 사용하십시오.

---

## 로컬 개발 / 저사양 GPU 환경

WSL2, 소비자 GPU(VRAM 8~12GB) 환경에서는 `--local-dev`를 사용합니다.

```bash
python bench.py \
  --data /data/ShareGPT.json \
  --scenario gpu_cpu \
  --model facebook/opt-125m \
  --local-dev \
  --gpu-mem 0.45 \
  --max-model-len 512 \
  --max-output-tokens 32 \
  --num-requests 4 \
  --chunk-size 64 \
  --max-cpu-gb 2.0
```

`--local-dev` 활성화 시 적용되는 옵션:

| 옵션 | 값 | 효과 |
|---|---|---|
| `--enforce-eager` | — | CUDA graph 비활성화 → VRAM 절약 |
| `--max-num-batched-tokens` | `max_model_len` | 배치 크기 제한 |
| `kv_buffer_device` | `cpu` | KV transfer buffer를 CPU에 할당 |
| `kv_buffer_size` | `200MB` | KV buffer 크기 제한 |

---

## 알려진 제약사항

1. **SSD `offload_wait` 부족**: SATA SSD는 write throughput이 낮아 `--offload-wait 5`(default)으로는 async store가 완료되지 않을 수 있습니다. `gpu_ssd`, `gpu_cpu_ssd` 시나리오에서 `retrieve_hit_rate`이 예상보다 낮게 나오면 `--offload-wait 15` 이상으로 늘리십시오.

2. **서버 로그 숨김**: 기본적으로 vLLM 서버의 stdout/stderr는 `/dev/null`로 redirect됩니다. 서버 크래시 디버깅이 필요하면 `bench.py`의 `server_session` 함수에서 `stdout=subprocess.DEVNULL`을 `stdout=None`으로 임시 변경하십시오.

3. **`--server-url` + `--scenario all` 비권장**: 외부 서버 모드에서 `all`을 지정하면 4개 시나리오가 동일 서버를 측정합니다. 결과 레이블은 다르지만 실제로는 같은 LMCache 설정을 반복 측정합니다.

4. **요청 단위 hit_rate vs 토큰 단위 retrieve_hit_rate**: 파라미터 `--hit-rates`는 요청(request) 단위 warm 비율이고, 결과의 `retrieve_hit_rate`는 LMCache가 청크 단위로 hit한 실제 토큰 비율입니다. 두 값은 항상 다릅니다.

5. **Prometheus multiprocess 환경변수**: LMCache V1은 `PROMETHEUS_MULTIPROC_DIR` 환경변수를 사용하여 cross-process 메트릭 집계를 합니다. 이 변수가 이미 설정된 환경에서는 충돌이 발생할 수 있습니다.
