"""
Benchmark runner.

Drives the three-phase test loop for a single scenario:

  cold        — first inference; KV computed and pushed to offload target
  gpu_hit     — same prompts again while vLLM's GPU prefix cache is warm
  offload_hit — GPU cache wiped; KV retrieved from CPU/SSD via LMCache

Query shape follows InferenceX: a single shared prefix (maximises cache
sharing across requests) plus a per-request unique suffix.
"""

import contextlib
import time
from typing import List, Optional

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.observability import LMCStatsMonitor
from lmcache.v1.cache_engine import LMCacheEngineBuilder

from icms.env import ScenarioConfig, apply
from icms.metrics import Collector
from icms.workload import WorkloadBatch, make_batch, make_warmup_batch


# ── LLM lifecycle ─────────────────────────────────────────────────────────────

@contextlib.contextmanager
def _llm_session(
    model: str,
    gpu_mem: float,
    max_len: int,
    local_dev: bool = False,
):
    """Spin up vLLM+LMCache, yield the LLM, then destroy LMCache state.

    local_dev=True applies a set of workarounds for low-VRAM environments
    (e.g. WSL2 + consumer GPU).  Remove --local-dev when running on a
    production server with sufficient VRAM.

    Workarounds applied when local_dev=True
    ────────────────────────────────────────
    kv_buffer_device="cpu"          LMCache transfer buffer on CPU RAM instead
                                    of GPU (default 1 GB GPU buffer → OOM on
                                    small GPUs)
    kv_buffer_size=200_000_000      Shrink buffer to 200 MB
    enforce_eager=True              Skip CUDA-graph capture (saves ~200 MB VRAM)
    max_num_batched_tokens=max_len  Shrink profile-run batch so the memory
                                    probe uses less activation memory
    kv_cache_memory_bytes=200 MB    Bypass profile-based VRAM estimation
                                    (estimation returns 0 on tiny GPUs)
    attention_config TRITON_ATTN    FlashInfer JIT warmup crashes on SM < 8.0;
                                    Triton backend works on SM 7.5
    """
    kv_kwargs: dict = {}
    llm_kwargs: dict = {}

    if local_dev:
        kv_kwargs = {
            "kv_buffer_device": "cpu",
            "kv_buffer_size": 200_000_000,
        }
        llm_kwargs = {
            "enforce_eager": True,
            "max_num_batched_tokens": max_len,
            "kv_cache_memory_bytes": 200_000_000,
            "attention_config": {"backend": "TRITON_ATTN"},
        }

    ktc = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
        **kv_kwargs,
    )
    print(f"  [llm] loading {model}  gpu_mem={gpu_mem}  max_len={max_len}"
          + ("  [local-dev]" if local_dev else ""))
    llm = LLM(
        model=model,
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_len,
        enable_prefix_caching=True,
        kv_transfer_config=ktc,
        **llm_kwargs,
    )
    try:
        yield llm
    finally:
        print("  [llm] shutting down LMCache engine")
        LMCacheEngineBuilder.destroy(ENGINE_NAME)
        LMCStatsMonitor.DestroyInstance()


# ── per-scenario run ──────────────────────────────────────────────────────────

def run_scenario(
    sc: ScenarioConfig,
    *,
    model: str,
    gpu_mem: float,
    max_model_len: int,
    request_counts: List[int],
    prefix_len: int,
    suffix_len: int,
    max_output_tokens: int,
    offload_wait_s: float,
    warmup: bool,
    collector: Collector,
    local_dev: bool = False,
) -> None:
    """
    Run all three phases for every value in request_counts under scenario sc.

    Phase sequence per (scenario, num_requests):
      1. cold        — cold-cache prefill; KV stored to offload target
      2. gpu_hit     — same prompts while GPU prefix cache is hot
      3. offload_hit — GPU cache reset; KV loaded back from CPU/SSD
    """
    print(f"\n{'━' * 64}")
    print(f"  {sc.label}")
    print(f"{'━' * 64}")

    apply(sc)   # write env vars before LLM init

    sp = SamplingParams(max_tokens=max_output_tokens, temperature=0.0)
    monitor = LMCStatsMonitor.GetOrCreate()

    with _llm_session(model, gpu_mem, max_model_len, local_dev=local_dev) as llm:

        # optional warmup (first batch only)
        if warmup:
            print("  [warmup] running single warmup generation...")
            w = make_warmup_batch(prefix_len, suffix_len)
            llm.generate(w.prompts, sp, use_tqdm=False)
            llm.reset_prefix_cache()
            monitor.get_stats_and_clear()
            time.sleep(2.0)

        for i, n in enumerate(request_counts):
            batch: WorkloadBatch = make_batch(
                num_requests=n,
                prefix_len=prefix_len,
                suffix_len=suffix_len,
                seed=100 * (i + 1),   # different seed per concurrency level
            )

            _header(sc.name, n, prefix_len, suffix_len)

            # ── Phase 1: cold ────────────────────────────────────────────
            print("  [cold]        prefill + store KV to offload target")
            monitor.get_stats_and_clear()
            t0 = time.perf_counter()
            llm.generate(batch.prompts, sp, use_tqdm=False)
            wall = (time.perf_counter() - t0) * 1e3
            snap = collector.record(sc.name, "cold", n, wall)
            print(f"                wall={wall:.0f}ms  stored={snap.tok_stored} tok")

            # ── Phase 2: GPU hit ─────────────────────────────────────────
            print("  [gpu_hit]     same prompts — GPU prefix cache hot")
            t0 = time.perf_counter()
            llm.generate(batch.prompts, sp, use_tqdm=False)
            wall = (time.perf_counter() - t0) * 1e3
            snap = collector.record(sc.name, "gpu_hit", n, wall)
            print(f"                wall={wall:.0f}ms  "
                  f"vllm_hit={snap.tok_vllm_hit} tok  "
                  f"hit_rate={snap.retrieve_hit_rate:.3f}")

            # ── reset GPU cache, let offload tasks drain ─────────────────
            llm.reset_prefix_cache()
            print(f"  [wait]        GPU cache cleared, waiting {offload_wait_s}s...")
            time.sleep(offload_wait_s)

            # ── Phase 3: offload hit ──────────────────────────────────────
            print(f"  [offload_hit] KV retrieve from {sc.name.upper()} storage")
            t0 = time.perf_counter()
            llm.generate(batch.prompts, sp, use_tqdm=False)
            wall = (time.perf_counter() - t0) * 1e3
            snap = collector.record(sc.name, "offload_hit", n, wall)
            print(f"                wall={wall:.0f}ms  "
                  f"hit_rate={snap.retrieve_hit_rate:.3f}  "
                  f"retrieve_speed={snap.avg_retrieve_tps:.0f} tok/s  "
                  f"cpu={snap.cpu_bytes // 2**20}MB  "
                  f"disk={snap.disk_bytes // 2**20}MB")

            # reset before next concurrency level
            llm.reset_prefix_cache()
            time.sleep(1.0)

    print(f"\n  ✓ scenario '{sc.name}' done\n")


# ── helpers ───────────────────────────────────────────────────────────────────

def _header(scenario: str, n: int, prefix: int, suffix: int):
    print(f"\n  {'─' * 60}")
    print(f"  scenario={scenario}  requests={n}  "
          f"prompt={prefix}+{suffix}={prefix+suffix} tok")
    print(f"  {'─' * 60}")
