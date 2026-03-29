# LMCache 메트릭 모니터링이 동작하지 않는 이유와 해결 방법

## 문제

`icms`를 실행하면 phase timing(wall time)은 정상 수집되지만, LMCache 세부 메트릭
(저장/조회 토큰 수, hit rate, retrieve 속도 등)이 모두 0으로 표시된다.

```
│  retrieve/store/lookup req  0 / 0 / 0
│  tokens: requested=0  hit=0  stored=0  prompt=0
│  cache: cpu=0 MB  disk=0 MB  remote=0 MB
```

## 원인

vLLM **V1 엔진** (vllm ≥ 0.6.x)은 `python -m multiprocessing` spawn 방식으로
별도의 **EngineCore subprocess**를 띄운다.

```
[Parent process]          [EngineCore_DP0 subprocess]
  icms/__main__.py    →       vllm V1 engine
  Collector.record()          LMCache engine
  LMCStatsMonitor ←×          LMCStatsMonitor  ← 실제 stats 여기
```

`Collector.record()`가 호출하는 `LMCStatsMonitor.GetOrCreate()`는 **parent
프로세스**의 인스턴스를 반환한다. 실제 LMCache 연산은 subprocess에서 일어나므로
parent의 monitor에는 아무 데이터도 쌓이지 않는다.

---

## 해결 방법

### 방법 1 — LMCache Internal API Server 활용 (권장)

LMCache에는 subprocess 내부에서 HTTP API 서버를 띄우는 기능이 있다.
이를 활성화하면 parent 프로세스에서 HTTP로 stats를 폴링할 수 있다.

#### 1-1. LMCache config에서 활성화

환경변수 또는 config 파일로 활성화한다.

```bash
export LMCACHE_INTERNAL_API_SERVER_ENABLED=true
export LMCACHE_INTERNAL_API_SERVER_PORT_START=6999   # 기본값
```

또는 `KVTransferConfig`의 `kv_connector_extra_config`에 전달:

```python
from vllm.config import KVTransferConfig

ktc = KVTransferConfig(
    kv_connector="LMCacheConnectorV1",
    kv_role="kv_both",
    kv_connector_extra_config={
        "internal_api_server_enabled": True,
        "internal_api_server_port_start": 6999,
    },
)
```

#### 1-2. Collector를 HTTP 폴링 방식으로 교체

`metrics.py`의 `Collector.record()`를 subprocess-safe하게 수정한다.

```python
import requests

class Collector:
    LMCACHE_API = "http://localhost:6999"   # internal_api_server_port_start

    def record(self, scenario, phase, num_requests, wall_ms):
        # LMCache internal API에서 stats 가져오기
        resp = requests.get(f"{self.LMCACHE_API}/stats/and/clear", timeout=5)
        raw = resp.json()
        # raw를 Snapshot으로 변환 (LMCache API 스펙에 맞게)
        snap = Snapshot(scenario=scenario, phase=phase,
                        num_requests=num_requests, wall_ms=wall_ms,
                        tok_stored=raw.get("interval_stored_tokens", 0),
                        ...)
        self._rows.append(snap)
        return snap
```

> LMCache internal API 엔드포인트는 버전에 따라 다를 수 있으므로
> `lmcache/v1/internal_api_server/` 소스 또는 공식 문서 확인 필요.

---

### 방법 2 — vLLM API 서버 모드 사용

`python -m icms`의 offline LLM 대신 vLLM을 **API 서버**로 띄우고
OpenAI-compatible API로 요청을 보내면, LMCache engine이 서버 프로세스 내에 있어
별도의 stats 수집이 가능하다.

```bash
# 터미널 1: vLLM + LMCache 서버 기동
LMCACHE_LOCAL_CPU=True LMCACHE_MAX_LOCAL_CPU_SIZE=10 \
vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
  --enable-prefix-caching

# 터미널 2: icms를 HTTP 클라이언트 모드로 실행 (별도 구현 필요)
python -m icms --scenario cpu --server-url http://localhost:8000
```

이 경우 `runner.py`를 HTTP 클라이언트로 완전히 재작성해야 한다.

---

### 방법 3 — vLLM V0 엔진 사용 (단기 우회)

vLLM V0 엔진은 단일 프로세스로 동작하므로 `LMCStatsMonitor`가 같은 프로세스에 있다.

```bash
VLLM_USE_V1=0 python -m icms --scenario cpu
```

단, V0 엔진은 deprecated이며 향후 제거될 예정이므로 임시 방편으로만 사용.

---

## 정리

| 방법 | 난이도 | 안정성 | 비고 |
|------|--------|--------|------|
| LMCache Internal API (방법 1) | 중 | 높음 | 권장. 코드 수정 최소화 |
| vLLM API 서버 (방법 2) | 높 | 높음 | runner.py 재작성 필요 |
| VLLM_USE_V1=0 (방법 3) | 낮 | 낮음 | V0 deprecated, 임시 우회용 |

**권장 경로**: 방법 1 — LMCache internal API server 활성화 후
`Collector.record()`를 HTTP 폴링으로 교체.
