# ICMS — KV Cache Offloading Benchmark

`icms/` 디렉토리의 vLLM + LMCache 기반 KV 캐시 오프로딩 벤치마크 앱입니다.

## 시나리오

| 시나리오  | 설명                                      | 데이터 경로     |
|-----------|-------------------------------------------|-----------------|
| `cpu`     | GPU → CPU 메모리 오프로딩                 | System RAM      |
| `ssd`     | GPU → Disk 오프로딩 (CPU staging bypass)  | `--disk-path`   |
| `cpu_ssd` | GPU → CPU → Disk 계층형 (CPU=L2, SSD=L3) | RAM + `--disk-path` |

## 실행 방법

```bash
# CPU 오프로딩 테스트
python -m icms --scenario cpu

# SSD 오프로딩 테스트
python -m icms --scenario ssd --disk-path /mnt/nvme/kvcache

# CPU+SSD 계층형 테스트
python -m icms --scenario cpu_ssd --num-requests 1 4 16

# 세 시나리오 모두 실행 (비교 리포트 생성)
python -m icms --scenario all --prefix-len 2000 --suffix-len 500
```

## 주요 CLI 파라미터

| 파라미터             | 기본값                                   | 설명                                        |
|----------------------|------------------------------------------|---------------------------------------------|
| `--scenario`         | (필수)                                   | `cpu`, `ssd`, `cpu_ssd`, `all`              |
| `--model`            | `meta-llama/Meta-Llama-3.1-8B-Instruct`  | 사용할 모델                                 |
| `--gpu-mem`          | `0.5`                                    | vLLM GPU 메모리 사용률                      |
| `--max-model-len`    | `8000`                                   | 최대 시퀀스 길이 (토큰)                     |
| `--num-requests`     | `1 4 16`                                 | 테스트할 동시 요청 수 목록                  |
| `--prefix-len`       | `1500`                                   | 공유 prefix 토큰 길이                       |
| `--suffix-len`       | `500`                                    | 요청별 고유 suffix 토큰 길이                |
| `--max-output-tokens`| `1`                                      | 생성할 최대 토큰 수                         |
| `--chunk-size`       | `256`                                    | LMCache 청크 크기 (토큰)                    |
| `--max-cpu-gb`       | `10.0`                                   | CPU 캐시 최대 크기 (GB)                     |
| `--max-disk-gb`      | `10.0`                                   | Disk 캐시 최대 크기 (GB)                    |
| `--disk-path`        | `/tmp/icms_kvcache`                      | SSD 캐시 경로                               |
| `--offload-wait`     | `5.0`                                    | GPU 캐시 리셋 후 대기 시간 (초)             |
| `--output-dir`       | `./icms_results`                         | JSON/CSV 결과 저장 경로                     |
| `--no-warmup`        | (flag)                                   | 워밍업 생략                                 |

## 벤치마크 3단계

각 `--num-requests` 값에 대해 3단계를 순서대로 실행합니다:

1. **cold** — 캐시 없이 KV 계산 후 오프로드 타겟에 저장
2. **gpu_hit** — 동일 프롬프트, vLLM GPU prefix cache 히트
3. **offload_hit** — GPU 캐시 초기화 후 LMCache(CPU/SSD)에서 KV 복원

## 수집 메트릭 (LMCStatsMonitor)

- Retrieve/Store/Lookup 요청 수 및 토큰 수
- Hit Rate (retrieve, lookup)
- Retrieve/Store 속도 (tokens/sec) 및 시간
- 파이프라인 세부 타이밍 (process_tokens, broadcast, to_gpu, from_gpu, put 등)
- 캐시 사용량 (CPU bytes, disk bytes, remote bytes)
- Eviction 카운트 (count, keys, failed, forced_unpin)
- Active/Pinned 메모리 객체 수
- P2P 전송 통계
- 요청별 lookup hit-rate 분포
- 캐시 lifespan 평균

## 출력

- **콘솔**: Phase Timing 요약 + 상세 메트릭 테이블 + 시나리오 간 비교 (offload_hit 기준)
- **JSON**: `<output-dir>/icms_results_<timestamp>.json` — config + 전체 snapshot
- **CSV**: `<output-dir>/icms_results_<timestamp>.csv` — 플랫 테이블 (row = snapshot)

## 의존성

- vLLM (V1 KV connector 지원 버전, `LMCacheConnectorV1`)
- LMCache (`lmcache` 패키지, `lmcache.v1.cache_engine` 포함)
- GPU 및 모델 접근 필요
