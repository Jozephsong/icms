"""
ShareGPT-based KV Cache Tier Benchmark  (server-client mode)

Architecture
────────────
  Server : vllm serve  — bench.py spawns one vLLM server per scenario.
           LMCache is configured via environment variables before the server
           starts.  VLLM_SERVER_DEV_MODE=1 is set to enable the
           POST /reset_prefix_cache endpoint used between hit-rate levels.

  Client : bench.py — uses the OpenAI-compatible streaming API to send
           requests and measure TTFT (first streaming chunk) per request.

  Metrics: LMCache exposes a Prometheus HTTP server on a separate port
           (default 9090).  bench.py scrapes it before and after each
           benchmark run to get interval-delta counter values.

Scenarios
─────────
  Scenario 1: gpu_only    — GPU prefix cache only (no LMCache offload)
  Scenario 2: gpu_cpu     — GPU + CPU RAM offload
  Scenario 3: gpu_ssd     — GPU + SSD offload
  Scenario 4: gpu_cpu_ssd — GPU + CPU + SSD tiered (CPU=L2, SSD=L3)

Hit-rate control
────────────────
  For each hit_rate H in the sweep:
    1. Prewarm:  send warm_pool[:N*H] to populate GPU cache (or LMCache for
                 offload scenarios).
    2. Reset GPU prefix cache via POST /reset_prefix_cache (offload only).
    3. Wait offload_wait_s for LMCache async stores to settle.
    4. Scrape Prometheus → baseline counter snapshot.
    5. Benchmark: send N*H warm + N*(1-H) fresh prompts concurrently.
    6. Wait stats_flush_wait_s for LMCache→Prometheus flush (log_interval≈10s).
    7. Scrape Prometheus → after-snapshot; compute delta.
    8. Record BenchResult.
    9. Reset GPU prefix cache (cleanup for next iteration).

  The same warm_pool and cold_pool are reused across all scenarios so that
  results are directly comparable.

Metrics collected per (scenario, hit_rate)
──────────────────────────────────────────
  Client-side (always):
    TTFT      mean, p50, p90, p99  (first streaming chunk − request start)
    Gen TPS   total generated tokens / wall time

  LMCache via Prometheus (when prometheus endpoint is reachable):
    tok_requested, tok_hit, tok_stored, tok_vllm_hit, tok_prompt
    retrieve_hit_rate  (computed from counter delta)
    avg_retrieve_s, avg_retrieve_tps   (from histogram _sum / _count delta)
    avg_store_s,    avg_store_tps
    cpu_bytes, disk_bytes             (gauges, latest value)

Usage
─────
  # Full run (bench.py manages vllm serve lifecycle):
  python bench.py --data ShareGPT.json --scenario all \\
      --model meta-llama/Meta-Llama-3.1-8B-Instruct

  # Single scenario with custom params:
  python bench.py --data ShareGPT.json --scenario gpu_cpu \\
      --num-requests 8 --max-output-tokens 128 \\
      --chunk-size 256 --max-cpu-gb 20.0

  # Connect to existing running server (skip lifecycle management):
  python bench.py --data ShareGPT.json --scenario gpu_cpu \\
      --server-url http://localhost:8000/v1 \\
      --prometheus-url http://localhost:9090/metrics

  # WSL2 / local dev:
  python bench.py --data ShareGPT.json --scenario gpu_cpu \\
      --model facebook/opt-125m --local-dev \\
      --gpu-mem 0.45 --max-model-len 512 --max-output-tokens 32
"""

import argparse
import asyncio
import contextlib
import csv
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import openai
import requests


# ── Scenario registry ─────────────────────────────────────────────────────────

SCENARIO_LABELS: Dict[str, str] = {
    "gpu_only":    "Scenario 1 — GPU only (no offload)",
    "gpu_cpu":     "Scenario 2 — GPU + CPU RAM offload",
    "gpu_ssd":     "Scenario 3 — GPU + SSD offload",
    "gpu_cpu_ssd": "Scenario 4 — GPU + CPU + SSD tiered",
}
ALL_SCENARIOS = list(SCENARIO_LABELS.keys())


# ── Scenario config ───────────────────────────────────────────────────────────

@dataclass
class ScenarioConfig:
    name: str
    label: str
    chunk_size: int
    cpu_enabled: bool
    max_cpu_gb: float
    disk_enabled: bool
    disk_dir: str
    max_disk_gb: float


def make_scenario(
    name: str,
    chunk_size: int,
    max_cpu_gb: float,
    disk_base: str,
    max_disk_gb: float,
) -> ScenarioConfig:
    if name not in SCENARIO_LABELS:
        raise ValueError(f"Unknown scenario '{name}'. Choose: {ALL_SCENARIOS}")
    cpu_on  = name in ("gpu_cpu", "gpu_cpu_ssd")
    disk_on = name in ("gpu_ssd", "gpu_cpu_ssd")
    disk_dir = os.path.join(disk_base, f"lmcache_{name}") if disk_on else ""
    return ScenarioConfig(
        name=name,
        label=SCENARIO_LABELS[name],
        chunk_size=chunk_size,
        cpu_enabled=cpu_on,
        max_cpu_gb=max_cpu_gb,
        disk_enabled=disk_on,
        disk_dir=disk_dir,
        max_disk_gb=max_disk_gb,
    )


def build_lmcache_env(sc: ScenarioConfig) -> Dict[str, str]:
    """
    Return the LMCache environment variables for the given scenario.
    These are merged into the vllm serve subprocess environment.

    gpu_only returns an empty dict — LMCache is not loaded at all for that
    scenario so the baseline reflects pure vLLM GPU prefix-cache performance.
    """
    if sc.name == "gpu_only":
        return {}  # pure vLLM; no LMCache connector overhead

    env: Dict[str, str] = {
        "LMCACHE_USE_EXPERIMENTAL": "True",
        "LMCACHE_CHUNK_SIZE": str(sc.chunk_size),
        "LMCACHE_LOCAL_CPU": "True" if sc.cpu_enabled else "False",
    }
    if sc.cpu_enabled:
        env["LMCACHE_MAX_LOCAL_CPU_SIZE"] = str(sc.max_cpu_gb)
    if sc.disk_enabled:
        os.makedirs(sc.disk_dir, exist_ok=True)
        env["LMCACHE_LOCAL_DISK"] = f"file://{sc.disk_dir}/"
        env["LMCACHE_MAX_LOCAL_DISK_SIZE"] = str(sc.max_disk_gb)
    return env


# ── Data loading ──────────────────────────────────────────────────────────────

def _build_prompt(entry: dict) -> Optional[str]:
    """Flatten a ShareGPT entry into a single text prompt."""
    parts = []
    for turn in entry.get("conversations", []):
        role = turn.get("from", "")
        value = (turn.get("value") or "").strip()
        if not value:
            continue
        if role in ("human", "user"):
            parts.append(f"Human: {value}")
        elif role in ("gpt", "chatgpt", "bing", "bard", "assistant"):
            parts.append(f"Assistant: {value}")
    return "\n\n".join(parts) if len(parts) >= 2 else None


def load_conversations(
    data_path: str,
    num_needed: int,
    max_chars: int = 4000,
    seed: int = 42,
) -> List[str]:
    """Load and filter ShareGPT conversations; return text prompts."""
    print(f"[data] Loading {data_path} ...")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rng = random.Random(seed)
    rng.shuffle(data)

    prompts: List[str] = []
    for entry in data:
        prompt = _build_prompt(entry)
        if prompt and len(prompt) <= max_chars:
            prompts.append(prompt)
        if len(prompts) >= num_needed:
            break

    if len(prompts) < num_needed:
        raise ValueError(
            f"Not enough valid conversations: need {num_needed}, "
            f"found {len(prompts)} from {len(data)} entries.\n"
            f"  Try: fewer --num-requests, larger --max-chars, or bigger dataset."
        )

    print(f"[data] Loaded {len(prompts)} prompts (max_chars={max_chars}).")
    return prompts


# ── Prometheus metrics scraping ───────────────────────────────────────────────

# Pattern matches: metric_name{labels...} value
# Also handles names without labels and histogram/summary suffixes.
_PROM_LINE = re.compile(
    r'^([a-zA-Z_:][a-zA-Z0-9_:]*(?:_total|_sum|_count|_bucket)?)'
    r'(?:\{[^}]*\})?\s+([\d.eE+\-]+)',
    re.MULTILINE,
)


def scrape_prometheus(url: str, timeout: float = 5.0) -> Dict[str, float]:
    """
    Scrape a Prometheus /metrics endpoint and return a flat name→value dict.

    Counter names retain the `_total` suffix (so deltas can be computed).
    Returns an empty dict if the endpoint is unreachable.
    """
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [prom] WARNING: cannot scrape {url}: {e}")
        return {}

    metrics: Dict[str, float] = {}
    for m in _PROM_LINE.finditer(resp.text):
        name, val_str = m.group(1), m.group(2)
        try:
            metrics[name] = float(val_str)
        except ValueError:
            pass
    return metrics


def _delta(after: Dict[str, float], before: Dict[str, float], key: str) -> float:
    """Return the positive delta for a counter key (0 if missing)."""
    return max(0.0, after.get(key, 0.0) - before.get(key, 0.0))


def _scrape_after_flush(
    prom_url: str,
    before: Dict[str, float],
    wait_s: float,
    prom_log_interval: float = 10.0,
) -> Dict[str, float]:
    """
    Wait wait_s, then scrape Prometheus.

    If the Prometheus endpoint was reachable during the 'before' snapshot but
    lmcache:num_requested_tokens_total has not advanced yet (flush not emitted),
    wait one more prom_log_interval and retry once.  This guards against the
    worst-case where the PrometheusController flush timer fires just *before*
    the benchmark batch starts, meaning the next flush is up to log_interval
    seconds after the batch completes — potentially beyond wait_s.
    """
    time.sleep(wait_s)
    if not prom_url:
        return {}
    after = scrape_prometheus(prom_url)
    key = "lmcache:num_requested_tokens_total"
    # Only retry if both snapshots were non-empty (prometheus reachable) AND
    # the counter hasn't moved at all.
    if before and after and _delta(after, before, key) == 0:
        print(
            f"  [prom] Counter unchanged after {wait_s:.0f}s; "
            f"waiting {prom_log_interval:.0f}s more for next flush cycle ..."
        )
        time.sleep(prom_log_interval)
        after = scrape_prometheus(prom_url)
    return after


def extract_lmc_metrics(
    before: Dict[str, float],
    after: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute interval-delta LMCache metrics from two Prometheus snapshots.

    Counter metrics:  delta = after − before  (cumulative Prometheus counters)
    Gauge metrics:    latest value from 'after' snapshot
    Histogram avg:    delta_sum / delta_count  (0 when count delta == 0)
    """
    def _hist_avg(name: str) -> float:
        s = _delta(after, before, f"lmcache:{name}_sum")
        c = _delta(after, before, f"lmcache:{name}_count")
        return s / c if c > 0 else 0.0

    tok_requested = _delta(after, before, "lmcache:num_requested_tokens_total")
    tok_hit       = _delta(after, before, "lmcache:num_hit_tokens_total")

    return {
        "tok_requested":    tok_requested,
        "tok_hit":          tok_hit,
        "tok_stored":       _delta(after, before, "lmcache:num_stored_tokens_total"),
        "tok_vllm_hit":     _delta(after, before, "lmcache:num_vllm_hit_tokens_total"),
        "tok_prompt":       _delta(after, before, "lmcache:num_prompt_tokens_total"),
        "retrieve_hit_rate": (tok_hit / tok_requested) if tok_requested > 0 else 0.0,
        # Histogram averages
        "avg_retrieve_s":     _hist_avg("time_to_retrieve"),
        "avg_store_s":        _hist_avg("time_to_store"),
        "avg_retrieve_tps":   _hist_avg("retrieve_speed"),
        "avg_store_tps":      _hist_avg("store_speed"),
        "avg_retr_to_gpu_s":  _hist_avg("retrieve_to_gpu_time"),
        "avg_store_from_gpu_s": _hist_avg("store_from_gpu_time"),
        # Gauges (latest value)
        "cpu_bytes":  int(after.get("lmcache:local_cache_usage",  0.0)),
        "disk_bytes": int(after.get("lmcache:local_storage_usage", 0.0)),
    }


# ── Results ───────────────────────────────────────────────────────────────────

def _avg(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _pct(xs: List[float], p: float) -> float:
    return float(np.percentile(xs, p)) if xs else 0.0


@dataclass
class BenchResult:
    # Identity
    scenario:     str
    hit_rate:     float
    num_requests: int
    wall_ms:      float
    ts: float = field(default_factory=time.time)

    # TTFT (seconds, client-side streaming measurement)
    ttft_mean_s: float = 0.0
    ttft_p50_s:  float = 0.0
    ttft_p90_s:  float = 0.0
    ttft_p99_s:  float = 0.0

    # Throughput
    total_gen_tokens: int   = 0
    gen_tps:          float = 0.0

    # LMCache token counters (from Prometheus delta)
    tok_requested: int = 0
    tok_hit:       int = 0
    tok_stored:    int = 0
    tok_vllm_hit:  int = 0
    tok_prompt:    int = 0

    # Hit rate (computed from counter delta)
    retrieve_hit_rate: float = 0.0

    # Latency / throughput (from histogram delta)
    avg_retrieve_s:     float = 0.0
    avg_store_s:        float = 0.0
    avg_retrieve_tps:   float = 0.0
    avg_store_tps:      float = 0.0
    avg_retr_to_gpu_s:  float = 0.0
    avg_store_from_gpu_s: float = 0.0

    # Cache occupancy (gauges)
    cpu_bytes:  int = 0
    disk_bytes: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


class ResultsCollector:
    def __init__(self):
        self._rows: List[BenchResult] = []

    def record(
        self,
        scenario: str,
        hit_rate: float,
        num_requests: int,
        wall_ms: float,
        ttfts: List[float],
        total_gen_tokens: int,
        lmc: Dict[str, float],
    ) -> BenchResult:
        wall_s = wall_ms / 1000.0
        r = BenchResult(
            scenario=scenario,
            hit_rate=hit_rate,
            num_requests=num_requests,
            wall_ms=wall_ms,
            ttft_mean_s=_avg(ttfts),
            ttft_p50_s=_pct(ttfts, 50),
            ttft_p90_s=_pct(ttfts, 90),
            ttft_p99_s=_pct(ttfts, 99),
            total_gen_tokens=total_gen_tokens,
            gen_tps=total_gen_tokens / wall_s if wall_s > 0 else 0.0,
            tok_requested=int(lmc.get("tok_requested", 0)),
            tok_hit=int(lmc.get("tok_hit", 0)),
            tok_stored=int(lmc.get("tok_stored", 0)),
            tok_vllm_hit=int(lmc.get("tok_vllm_hit", 0)),
            tok_prompt=int(lmc.get("tok_prompt", 0)),
            retrieve_hit_rate=lmc.get("retrieve_hit_rate", 0.0),
            avg_retrieve_s=lmc.get("avg_retrieve_s", 0.0),
            avg_store_s=lmc.get("avg_store_s", 0.0),
            avg_retrieve_tps=lmc.get("avg_retrieve_tps", 0.0),
            avg_store_tps=lmc.get("avg_store_tps", 0.0),
            avg_retr_to_gpu_s=lmc.get("avg_retr_to_gpu_s", 0.0),
            avg_store_from_gpu_s=lmc.get("avg_store_from_gpu_s", 0.0),
            cpu_bytes=int(lmc.get("cpu_bytes", 0)),
            disk_bytes=int(lmc.get("disk_bytes", 0)),
        )
        self._rows.append(r)
        return r

    @property
    def all(self) -> List[BenchResult]:
        return list(self._rows)


# ── Async streaming client ────────────────────────────────────────────────────

@dataclass
class RequestResponse:
    ttft_s: float           # time to first token (seconds)
    gen_tokens: int         # number of generated tokens
    wall_s: float           # total request time


async def _send_one(
    client: openai.AsyncOpenAI,
    model: str,
    prompt: str,
    max_tokens: int,
) -> RequestResponse:
    """
    Send one streaming chat request and return TTFT + token count.
    TTFT is measured as the wall time from request start to the first
    non-empty content chunk.
    """
    start = time.perf_counter()
    first_token_t: Optional[float] = None
    gen_tokens = 0

    stream = await client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        max_tokens=max_tokens,
        temperature=0.0,
        stream=True,
        stream_options={"include_usage": True},
    )

    async for chunk in stream:
        if chunk.choices:
            content = chunk.choices[0].delta.content
            if content and first_token_t is None:
                first_token_t = time.perf_counter()
        # last chunk carries usage
        if hasattr(chunk, "usage") and chunk.usage is not None:
            gen_tokens = chunk.usage.completion_tokens or gen_tokens

    end = time.perf_counter()
    ttft = (first_token_t - start) if first_token_t is not None else 0.0
    return RequestResponse(ttft_s=ttft, gen_tokens=gen_tokens, wall_s=end - start)


async def _send_batch_async(
    base_url: str,
    model: str,
    prompts: List[str],
    max_tokens: int,
) -> Tuple[List[float], int, float]:
    """
    Send all prompts concurrently (asyncio.gather) to the server.
    Returns: (ttfts, total_gen_tokens, wall_s)
    """
    client = openai.AsyncOpenAI(api_key="EMPTY", base_url=base_url)
    t0 = time.perf_counter()
    responses = await asyncio.gather(*[
        _send_one(client, model, p, max_tokens) for p in prompts
    ])
    wall_s = time.perf_counter() - t0

    ttfts      = [r.ttft_s for r in responses if r.ttft_s > 0]
    total_gen  = sum(r.gen_tokens for r in responses)
    return ttfts, total_gen, wall_s


def send_batch(
    base_url: str,
    model: str,
    prompts: List[str],
    max_tokens: int,
) -> Tuple[List[float], int, float]:
    """Synchronous wrapper around _send_batch_async."""
    return asyncio.run(_send_batch_async(base_url, model, prompts, max_tokens))


# ── vLLM server management ────────────────────────────────────────────────────

def _build_kv_transfer_config(local_dev: bool) -> str:
    """Build the --kv-transfer-config JSON string for vllm serve."""
    cfg: dict = {
        "kv_connector": "LMCacheConnectorV1",
        "kv_role":      "kv_both",
    }
    if local_dev:
        cfg["kv_buffer_device"] = "cpu"
        cfg["kv_buffer_size"]   = 200_000_000
    return json.dumps(cfg)


def _build_server_cmd(
    model: str,
    port: int,
    gpu_mem: float,
    max_model_len: int,
    num_gpu_blocks: Optional[int],
    local_dev: bool,
    use_lmcache: bool = True,
) -> List[str]:
    """
    Build the vllm serve command.

    use_lmcache=False (gpu_only scenario) omits --kv-transfer-config so that
    the server runs as pure vLLM with GPU prefix cache only.
    """
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model",                    model,
        "--port",                     str(port),
        "--gpu-memory-utilization",   str(gpu_mem),
        "--max-model-len",            str(max_model_len),
        "--enable-prefix-caching",
    ]
    if use_lmcache:
        cmd += ["--kv-transfer-config", _build_kv_transfer_config(local_dev)]
    if num_gpu_blocks is not None:
        cmd += ["--num-gpu-blocks-override", str(num_gpu_blocks)]
    if local_dev:
        cmd += [
            "--enforce-eager",
            "--max-num-batched-tokens", str(max_model_len),
        ]
    return cmd


@contextlib.contextmanager
def server_session(
    sc: ScenarioConfig,
    model: str,
    port: int,
    gpu_mem: float,
    max_model_len: int,
    num_gpu_blocks: Optional[int],
    local_dev: bool,
    startup_timeout: float = 300.0,
):
    """
    Context manager that starts `vllm serve` for the given scenario,
    waits until the server is ready, then yields.
    The server is killed (SIGTERM) on exit.

    VLLM_SERVER_DEV_MODE=1 is included so that POST /reset_prefix_cache
    is available within the session.
    """
    env = os.environ.copy()
    env.update(build_lmcache_env(sc))
    env["VLLM_SERVER_DEV_MODE"] = "1"

    cmd = _build_server_cmd(model, port, gpu_mem, max_model_len,
                            num_gpu_blocks, local_dev,
                            use_lmcache=sc.name != "gpu_only")
    print(f"\n[server] Starting vllm serve for scenario '{sc.name}' on port {port}")
    print(f"[server] LMCache env: {build_lmcache_env(sc)}")

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base_url = f"http://localhost:{port}"
    try:
        _wait_server_ready(base_url, timeout=startup_timeout)
        yield base_url
    finally:
        print(f"\n[server] Stopping server (pid={proc.pid}) ...")
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        print("[server] Server stopped.")


def _wait_server_ready(base_url: str, timeout: float = 300.0):
    """Poll GET /health until the server returns 200 or timeout expires."""
    url = f"{base_url}/health"
    deadline = time.time() + timeout
    last_err = ""
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                print(f"[server] Ready at {base_url}")
                return
        except Exception as e:
            last_err = str(e)
        time.sleep(3)
    raise TimeoutError(
        f"Server at {base_url} did not become ready within {timeout}s. "
        f"Last error: {last_err}"
    )


def reset_prefix_cache(base_url: str, reset_external: bool = False):
    """
    Call POST /reset_prefix_cache on the vLLM server.
    Requires VLLM_SERVER_DEV_MODE=1 to be set in the server environment.
    reset_external=False clears GPU prefix cache only (keeps LMCache content).
    """
    url = f"{base_url}/reset_prefix_cache"
    params = {"reset_external": str(reset_external).lower()}
    try:
        resp = requests.post(url, params=params, timeout=10)
        if resp.status_code == 200:
            print(f"  [cache] GPU prefix cache reset (reset_external={reset_external})")
        else:
            print(f"  [cache] WARNING: reset_prefix_cache returned {resp.status_code}")
    except Exception as e:
        print(f"  [cache] WARNING: reset_prefix_cache failed: {e}")


# ── Benchmark runner ──────────────────────────────────────────────────────────

def run_scenario(
    sc: ScenarioConfig,
    prompts_warm: List[str],
    prompts_cold: List[str],
    *,
    model: str,
    port: int,
    gpu_mem: float,
    max_model_len: int,
    num_requests: int,
    max_output_tokens: int,
    hit_rates: List[float],
    offload_wait_s: float,
    stats_flush_wait_s: float,
    prometheus_url: str,
    collector: ResultsCollector,
    external_server_url: Optional[str] = None,
    local_dev: bool = False,
    num_gpu_blocks: Optional[int] = None,
    startup_timeout: float = 300.0,
) -> None:
    """
    Run the full hit-rate sweep for one scenario.

    If external_server_url is given, connects to that server without starting
    one (the user must ensure it is running with the correct LMCache config).
    Otherwise, spawns a vllm serve subprocess.
    """
    print(f"\n{'━' * 64}")
    print(f"  {sc.label}")
    print(f"{'━' * 64}")

    def _run(base_url: str):
        cold_idx = 0

        for hr in hit_rates:
            n_hot  = int(num_requests * hr)
            n_cold = num_requests - n_hot

            print(f"\n  {'─' * 60}")
            print(f"  {sc.name}  hit_rate={hr:.0%}  "
                  f"hot={n_hot}  cold={n_cold}  total={num_requests}")
            print(f"  {'─' * 60}")

            # ── Phase 1: prewarm ──────────────────────────────────────────
            if n_hot > 0:
                print(f"  [prewarm]  sending {n_hot} conversations → "
                      f"populate {'GPU cache' if sc.name == 'gpu_only' else 'LMCache'}")
                send_batch(base_url + "/v1", model, prompts_warm[:n_hot],
                           max_output_tokens)

            # ── Phase 2: reset GPU prefix cache (offload scenarios) ───────
            if sc.name != "gpu_only":
                reset_prefix_cache(base_url, reset_external=False)
                print(f"  [wait]     GPU cache cleared, "
                      f"waiting {offload_wait_s}s for LMCache settle ...")
                time.sleep(offload_wait_s)

            # ── Phase 3: baseline Prometheus snapshot ─────────────────────
            prom_before = scrape_prometheus(prometheus_url)

            # ── Phase 4: benchmark batch ──────────────────────────────────
            cold_batch = prompts_cold[cold_idx : cold_idx + n_cold]
            cold_idx  += n_cold
            batch      = prompts_warm[:n_hot] + cold_batch
            random.shuffle(batch)

            t0 = time.perf_counter()
            ttfts, total_gen, wall_s = send_batch(
                base_url + "/v1", model, batch, max_output_tokens
            )
            wall_ms = wall_s * 1e3
            print(f"  [bench]    batch done in {wall_ms:.0f}ms, "
                  f"waiting {stats_flush_wait_s}s for Prometheus flush ...")

            # ── Phase 5+6: wait for LMCache → Prometheus flush, then scrape ─
            # _scrape_after_flush retries once if counters haven't moved yet.
            prom_after = _scrape_after_flush(
                prometheus_url, prom_before, stats_flush_wait_s
            )
            lmc = extract_lmc_metrics(prom_before, prom_after)

            result = collector.record(
                sc.name, hr, num_requests, wall_ms, ttfts, total_gen, lmc
            )
            _print_result(result)

            # ── Phase 7: cleanup — reset GPU cache for next iteration ─────
            reset_prefix_cache(base_url, reset_external=False)
            time.sleep(1.0)

        print(f"\n  ✓ scenario '{sc.name}' complete\n")

    if external_server_url is not None:
        print(f"  [server] Using external server at {external_server_url}")
        _base = external_server_url.rstrip("/")
        _run(_base[:-3] if _base.endswith("/v1") else _base)
    else:
        with server_session(
            sc, model, port, gpu_mem, max_model_len,
            num_gpu_blocks, local_dev, startup_timeout
        ) as base_url:
            _run(base_url)


def _print_result(r: BenchResult) -> None:
    ttft_str = (
        f"ttft_mean={r.ttft_mean_s * 1000:.1f}ms "
        f"ttft_p90={r.ttft_p90_s * 1000:.1f}ms"
        if r.ttft_mean_s > 0 else "ttft=n/a"
    )
    print(f"  [result]  wall={r.wall_ms:.0f}ms  {ttft_str}  "
          f"gen_tps={r.gen_tps:.1f}tok/s")
    print(f"            lmc_hit_rate={r.retrieve_hit_rate:.3f}  "
          f"vllm_hit={r.tok_vllm_hit:,}tok  "
          f"lmc_hit={r.tok_hit:,}tok  "
          f"stored={r.tok_stored:,}tok")
    print(f"            cpu={r.cpu_bytes // 2**20}MB  "
          f"disk={r.disk_bytes // 2**20}MB  "
          f"retr={r.avg_retrieve_tps:.0f}tok/s  "
          f"store={r.avg_store_tps:.0f}tok/s")


# ── Output ────────────────────────────────────────────────────────────────────

def print_summary(collector: ResultsCollector) -> None:
    rows = collector.all
    if not rows:
        return

    W = 120
    print("\n" + "═" * W)
    print("  BENCHMARK SUMMARY")
    print("═" * W)
    print(
        f"  {'Scenario':<14} {'HitRate':>7}  "
        f"{'TTFT_mean':>10} {'TTFT_p90':>9}  "
        f"{'GenTPS':>7}  "
        f"{'LMC_HitRate':>11}  "
        f"{'vLLM_hit':>9} {'LMC_hit':>9} {'Stored':>9}  "
        f"{'CPU_MB':>7} {'Disk_MB':>8}  "
        f"{'RetrSpd':>8}"
    )
    print("─" * W)
    prev_sc = None
    for r in rows:
        if prev_sc is not None and r.scenario != prev_sc:
            print()
        prev_sc = r.scenario
        ttft_m = f"{r.ttft_mean_s * 1000:>9.1f}ms" if r.ttft_mean_s else f"{'n/a':>10}"
        ttft_p = f"{r.ttft_p90_s * 1000:>8.1f}ms"  if r.ttft_p90_s  else f"{'n/a':>9}"
        print(
            f"  {r.scenario:<14} {r.hit_rate:>6.0%}  "
            f"{ttft_m} {ttft_p}  "
            f"{r.gen_tps:>7.1f}  "
            f"{r.retrieve_hit_rate:>11.3f}  "
            f"{r.tok_vllm_hit:>9,} {r.tok_hit:>9,} {r.tok_stored:>9,}  "
            f"{r.cpu_bytes // 2**20:>7,} {r.disk_bytes // 2**20:>8,}  "
            f"{r.avg_retrieve_tps:>7.0f}"
        )
    print("═" * W)
    print("  TTFT in ms | GenTPS tok/s | LMC_HitRate [0-1] | "
          "token counts | cache MB | LMCache retrieve tok/s")


def save_results(
    collector: ResultsCollector,
    cfg: dict,
    output_dir: str,
) -> None:
    out   = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    rows  = collector.all

    json_path = out / f"bench_{stamp}.json"
    with open(json_path, "w") as f:
        json.dump(
            {"timestamp": stamp, "config": cfg,
             "results": [r.as_dict() for r in rows]},
            f, indent=2, default=str,
        )

    csv_path = out / f"bench_{stamp}.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].as_dict().keys())
            w.writeheader()
            for r in rows:
                w.writerow(r.as_dict())

    print(f"\n[output] Results saved to {output_dir}/")
    print(f"         JSON : {json_path.name}")
    print(f"         CSV  : {csv_path.name}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python bench.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    g = p.add_argument_group("Data")
    g.add_argument("--data", required=True,
                   help="Path to ShareGPT JSON file")
    g.add_argument("--max-chars", type=int, default=4000,
                   help="Max characters per conversation (default: 4000)")
    g.add_argument("--data-seed", type=int, default=42)

    g = p.add_argument_group("Scenario")
    g.add_argument("--scenario", required=True,
                   choices=ALL_SCENARIOS + ["all"],
                   help="Memory tier scenario(s) to run")

    g = p.add_argument_group("vLLM server")
    g.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    g.add_argument("--port", type=int, default=8000,
                   help="vLLM server port (default: 8000)")
    g.add_argument("--gpu-mem", type=float, default=0.5,
                   help="vLLM gpu_memory_utilization (default: 0.5)")
    g.add_argument("--max-model-len", type=int, default=4096)
    g.add_argument("--num-gpu-blocks", type=int, default=None,
                   help="Override vLLM num_gpu_blocks (optional)")
    g.add_argument("--startup-timeout", type=float, default=300.0,
                   help="Seconds to wait for server ready (default: 300)")
    g.add_argument(
        "--server-url",
        default=None,
        metavar="URL",
        help=(
            "Connect to an already-running vLLM server instead of spawning one. "
            "Example: http://localhost:8000  "
            "When set, --port / --gpu-mem / --max-model-len are ignored for "
            "server startup, and the scenario LMCache env must be configured "
            "externally before starting the server."
        ),
    )

    g = p.add_argument_group("Benchmark")
    g.add_argument("--num-requests", type=int, default=8,
                   help="Requests per benchmark run (default: 8)")
    g.add_argument("--max-output-tokens", type=int, default=64)
    g.add_argument("--hit-rates", type=float, nargs="+",
                   default=[0.0, 0.25, 0.50, 0.75, 1.0], metavar="H")
    g.add_argument("--offload-wait", type=float, default=5.0,
                   help="Seconds to wait after GPU cache reset (default: 5.0)")
    g.add_argument("--stats-flush-wait", type=float, default=12.0,
                   help=(
                       "Seconds to wait after benchmark batch before scraping "
                       "Prometheus (should be > LMCache prometheus log_interval, "
                       "default 10s). Default: 12.0"
                   ))

    g = p.add_argument_group("LMCache")
    g.add_argument("--chunk-size", type=int, default=256)
    g.add_argument("--max-cpu-gb", type=float, default=10.0)
    g.add_argument("--max-disk-gb", type=float, default=10.0)
    g.add_argument("--disk-path", default="/tmp/kvcache_bench")

    g = p.add_argument_group("Prometheus")
    g.add_argument("--prometheus-url",
                   default="http://localhost:9090/metrics",
                   help=(
                       "LMCache Prometheus metrics endpoint. "
                       "LMCache exposes this on a separate port (default 9090). "
                       "Set to empty string to disable Prometheus scraping."
                   ))

    g = p.add_argument_group("Output")
    g.add_argument("--output-dir", default="./bench_results")

    g = p.add_argument_group("Dev")
    g.add_argument(
        "--local-dev", action="store_true",
        help=(
            "Low-VRAM workarounds for WSL2/consumer GPU: "
            "CPU KV buffer, eager mode, Triton backend. "
            "Remove on production servers."
        ),
    )

    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse()

    scenarios = ALL_SCENARIOS if args.scenario == "all" else [args.scenario]
    hit_rates = sorted(set(args.hit_rates))
    n         = args.num_requests

    cold_needed  = sum(n - int(n * hr) for hr in hit_rates)
    total_needed = n + cold_needed

    prometheus_url = args.prometheus_url.strip() if args.prometheus_url else ""

    print("\n" + "═" * 64)
    print("  ShareGPT KV Cache Tier Benchmark  [server-client mode]")
    print("═" * 64)
    print(f"  model           {args.model}")
    print(f"  scenarios       {', '.join(scenarios)}")
    print(f"  hit rates       {[f'{h:.0%}' for h in hit_rates]}")
    print(f"  requests/run    {n}")
    print(f"  output tokens   {args.max_output_tokens}")
    print(f"  chunk size      {args.chunk_size}")
    print(f"  max CPU         {args.max_cpu_gb} GB")
    print(f"  max SSD         {args.max_disk_gb} GB  ({args.disk_path})")
    print(f"  offload wait    {args.offload_wait}s")
    print(f"  stats flush wait {args.stats_flush_wait}s")
    print(f"  prometheus      {prometheus_url or '(disabled)'}")
    print(f"  warm pool       {n}  cold pool {cold_needed}  total {total_needed}")
    if args.server_url:
        print(f"  external server {args.server_url}")
    if args.local_dev:
        print(f"  *** local-dev mode (low-VRAM workarounds active) ***")
    print("═" * 64 + "\n")

    all_prompts  = load_conversations(
        args.data, total_needed,
        max_chars=args.max_chars,
        seed=args.data_seed,
    )
    prompts_warm = all_prompts[:n]
    prompts_cold = all_prompts[n:]
    print(f"[data] warm_pool={len(prompts_warm)}  cold_pool={len(prompts_cold)}\n")

    collector = ResultsCollector()

    for sc_name in scenarios:
        sc = make_scenario(
            name=sc_name,
            chunk_size=args.chunk_size,
            max_cpu_gb=args.max_cpu_gb,
            disk_base=args.disk_path,
            max_disk_gb=args.max_disk_gb,
        )
        run_scenario(
            sc,
            prompts_warm=prompts_warm,
            prompts_cold=prompts_cold,
            model=args.model,
            port=args.port,
            gpu_mem=args.gpu_mem,
            max_model_len=args.max_model_len,
            num_requests=n,
            max_output_tokens=args.max_output_tokens,
            hit_rates=hit_rates,
            offload_wait_s=args.offload_wait,
            stats_flush_wait_s=args.stats_flush_wait,
            prometheus_url=prometheus_url,
            collector=collector,
            external_server_url=args.server_url,
            local_dev=args.local_dev,
            num_gpu_blocks=args.num_gpu_blocks,
            startup_timeout=args.startup_timeout,
        )

    print_summary(collector)
    save_results(collector, vars(args), args.output_dir)


if __name__ == "__main__":
    main()
