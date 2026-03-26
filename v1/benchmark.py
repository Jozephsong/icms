"""
ICMS KV Cache Offloading Benchmark

Tests LMCache KV cache offloading via vLLM across three hardware scenarios:
  Case 1 — cpu:     GPU + CPU memory only
  Case 2 — ssd:     GPU + SSD only (bypasses CPU)
  Case 3 — cpu_ssd: GPU + CPU + SSD (tiered offloading)

For each scenario and each num_requests value, three phases are measured:
  cold:         Fresh inference, KV cache stored to offload target
  gpu_hit:      Same prompts re-run while GPU prefix cache is warm
  offload_hit:  GPU cache cleared, KV cache retrieved from offload target

Queries follow the InferenceX pattern: all requests share a common prefix
(prefix_len tokens) to maximise KV cache sharing, then each has a unique
suffix (unique_len tokens).

Usage:
    python -m icms --scenario cpu
    python -m icms --scenario ssd --disk-path /mnt/nvme/cache
    python -m icms --scenario cpu_ssd
    python -m icms --scenario all --num-requests 1 4 16
"""

import argparse
import contextlib
import os
import sys
import time
from dataclasses import asdict
from typing import List

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.observability import LMCStatsMonitor
from lmcache.v1.cache_engine import LMCacheEngineBuilder

from icms.config import BenchmarkConfig
from icms.metrics_collector import MetricsCollector
from icms.queries import make_shared_prefix_prompts, make_warmup_prompt
from icms.report import print_full_report, save_results_json, save_results_csv


# ──────────────────────────────────────────────
# Environment Setup
# ──────────────────────────────────────────────

_LMCACHE_ENV_KEYS = [
    "LMCACHE_LOCAL_CPU",
    "LMCACHE_LOCAL_DISK",
    "LMCACHE_MAX_LOCAL_CPU_SIZE",
    "LMCACHE_MAX_LOCAL_DISK_SIZE",
    "LMCACHE_GDS_PATH",
    "LMCACHE_CHUNK_SIZE",
    "LMCACHE_USE_EXPERIMENTAL",
]


def _clear_lmcache_env():
    for key in _LMCACHE_ENV_KEYS:
        os.environ.pop(key, None)


def setup_env_cpu(cfg: BenchmarkConfig):
    """Case 1: GPU + CPU memory offloading."""
    _clear_lmcache_env()
    os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"
    os.environ["LMCACHE_CHUNK_SIZE"] = str(cfg.chunk_size)
    os.environ["LMCACHE_LOCAL_CPU"] = "True"
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = str(cfg.max_cpu_size_gb)
    print(f"[ENV] Case 1 — CPU offloading: chunk_size={cfg.chunk_size}, "
          f"max_cpu={cfg.max_cpu_size_gb} GB")


def setup_env_ssd(cfg: BenchmarkConfig):
    """Case 2: GPU + SSD offloading (CPU bypassed)."""
    _clear_lmcache_env()
    os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"
    os.environ["LMCACHE_CHUNK_SIZE"] = str(cfg.chunk_size)
    os.environ["LMCACHE_LOCAL_CPU"] = "False"
    disk_path = os.path.join(cfg.disk_path, "lmcache_ssd")
    os.makedirs(disk_path, exist_ok=True)
    os.environ["LMCACHE_LOCAL_DISK"] = f"file://{disk_path}/"
    os.environ["LMCACHE_MAX_LOCAL_DISK_SIZE"] = str(cfg.max_disk_size_gb)
    print(f"[ENV] Case 2 — SSD offloading: chunk_size={cfg.chunk_size}, "
          f"disk_path={disk_path}, max_disk={cfg.max_disk_size_gb} GB")


def setup_env_cpu_ssd(cfg: BenchmarkConfig):
    """Case 3: GPU + CPU + SSD tiered offloading."""
    _clear_lmcache_env()
    os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"
    os.environ["LMCACHE_CHUNK_SIZE"] = str(cfg.chunk_size)
    os.environ["LMCACHE_LOCAL_CPU"] = "True"
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = str(cfg.max_cpu_size_gb)
    disk_path = os.path.join(cfg.disk_path, "lmcache_cpu_ssd")
    os.makedirs(disk_path, exist_ok=True)
    os.environ["LMCACHE_LOCAL_DISK"] = f"file://{disk_path}/"
    os.environ["LMCACHE_MAX_LOCAL_DISK_SIZE"] = str(cfg.max_disk_size_gb)
    print(f"[ENV] Case 3 — CPU+SSD offloading: chunk_size={cfg.chunk_size}, "
          f"max_cpu={cfg.max_cpu_size_gb} GB, disk_path={disk_path}, "
          f"max_disk={cfg.max_disk_size_gb} GB")


_SCENARIO_SETUP = {
    "cpu":     setup_env_cpu,
    "ssd":     setup_env_ssd,
    "cpu_ssd": setup_env_cpu_ssd,
}

_SCENARIO_LABELS = {
    "cpu":     "Case 1: GPU + CPU",
    "ssd":     "Case 2: GPU + SSD",
    "cpu_ssd": "Case 3: GPU + CPU + SSD",
}


# ──────────────────────────────────────────────
# LLM Lifecycle
# ──────────────────────────────────────────────

@contextlib.contextmanager
def build_llm(cfg: BenchmarkConfig):
    """Build and teardown a vLLM + LMCache LLM instance."""
    ktc = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
    )
    print(f"\n[LLM] Initializing model: {cfg.model}")
    print(f"[LLM]   gpu_memory_utilization={cfg.gpu_memory_utilization}, "
          f"max_model_len={cfg.max_model_len}")

    llm = LLM(
        model=cfg.model,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        max_model_len=cfg.max_model_len,
        enable_prefix_caching=True,
        kv_transfer_config=ktc,
    )
    try:
        yield llm
    finally:
        print("[LLM] Cleaning up LMCache engine...")
        LMCacheEngineBuilder.destroy(ENGINE_NAME)
        LMCStatsMonitor.DestroyInstance()


# ──────────────────────────────────────────────
# Benchmark Execution
# ──────────────────────────────────────────────

def run_single_scenario(
    scenario: str,
    cfg: BenchmarkConfig,
    collector: MetricsCollector,
):
    """
    Run a full benchmark for one scenario (cpu / ssd / cpu_ssd).

    For each num_requests value:
      Phase 1 — cold:         Fresh inference, KV cache stored to offload target.
      Phase 2 — gpu_hit:      Same prompts re-run while GPU prefix cache is warm.
      Phase 3 — offload_hit:  GPU cache cleared; KV cache fetched from offload target.

    Query pattern follows InferenceX: shared prefix_len tokens + per-request
    unique_len tokens to maximise prefix KV cache sharing.
    """
    label = _SCENARIO_LABELS.get(scenario, scenario.upper())
    print(f"\n{'━' * 60}")
    print(f"  Starting scenario: {label}")
    print(f"{'━' * 60}")

    _SCENARIO_SETUP[scenario](cfg)

    sampling_params = SamplingParams(
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )

    # Drain any stale metrics before starting
    monitor = LMCStatsMonitor.GetOrCreate()
    monitor.get_stats_and_clear()

    with build_llm(cfg) as llm:
        for req_idx, num_reqs in enumerate(cfg.num_requests):
            print(f"\n{'─' * 60}")
            print(f"  {label} | Requests={num_reqs} | "
                  f"prefix={cfg.prefix_len} + unique={cfg.unique_len} tokens")
            print(f"{'─' * 60}")

            # InferenceX-style prompts: shared prefix + unique suffix
            prompts = make_shared_prefix_prompts(
                num_requests=num_reqs,
                prefix_len=cfg.prefix_len,
                unique_len=cfg.unique_len,
                seed=42 + req_idx,
            )

            # ─── Warmup (first iteration only) ───
            if cfg.warmup and req_idx == 0:
                print("  [Warmup] Running warmup generation...")
                warmup = make_warmup_prompt(cfg.prefix_len, cfg.unique_len)
                llm.generate(warmup, sampling_params, use_tqdm=False)
                llm.reset_prefix_cache()
                monitor.get_stats_and_clear()
                time.sleep(2.0)

            # ─── Phase 1: Cold ───
            print("  [Phase 1] Cold — prefill & store KV to offload target...")
            monitor.get_stats_and_clear()
            t0 = time.perf_counter()
            llm.generate(prompts, sampling_params, use_tqdm=False)
            wall_ms = (time.perf_counter() - t0) * 1000
            snap = collector.collect_snapshot(scenario, "cold", num_reqs, wall_ms)
            print(f"    Wall: {wall_ms:.1f} ms | Stored: {snap.stored_tokens} tokens")

            # ─── Phase 2: GPU Hit ───
            print("  [Phase 2] GPU Hit — reading from vLLM prefix cache...")
            t0 = time.perf_counter()
            llm.generate(prompts, sampling_params, use_tqdm=False)
            wall_ms = (time.perf_counter() - t0) * 1000
            snap = collector.collect_snapshot(scenario, "gpu_hit", num_reqs, wall_ms)
            print(f"    Wall: {wall_ms:.1f} ms | vLLM hit tokens: {snap.vllm_hit_tokens} | "
                  f"Hit rate: {snap.retrieve_hit_rate:.4f}")

            # ─── Clear GPU prefix cache, wait for offload ───
            print("  [Clear] Resetting GPU prefix cache...")
            llm.reset_prefix_cache()
            print(f"  [Wait]  Waiting {cfg.offload_wait_secs}s for offload to complete...")
            time.sleep(cfg.offload_wait_secs)

            # ─── Phase 3: Offload Hit ───
            print(f"  [Phase 3] Offload Hit — fetching KV from {scenario.upper()} storage...")
            t0 = time.perf_counter()
            llm.generate(prompts, sampling_params, use_tqdm=False)
            wall_ms = (time.perf_counter() - t0) * 1000
            snap = collector.collect_snapshot(scenario, "offload_hit", num_reqs, wall_ms)
            print(f"    Wall: {wall_ms:.1f} ms | Hit rate: {snap.retrieve_hit_rate:.4f} | "
                  f"Retrieve speed: {snap.avg_retrieve_speed:.1f} tok/s | "
                  f"CPU cache: {snap.local_cache_usage_bytes // 1024 // 1024} MB | "
                  f"Disk cache: {snap.local_storage_usage_bytes // 1024 // 1024} MB")

            # Reset for next num_requests iteration
            llm.reset_prefix_cache()
            time.sleep(1.0)

    print(f"\n✅ Scenario '{scenario}' completed.\n")


# ──────────────────────────────────────────────
# CLI & Main
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ICMS KV Cache Offloading Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m icms --scenario cpu
  python -m icms --scenario ssd --disk-path /mnt/nvme/cache
  python -m icms --scenario cpu_ssd --num-requests 1 4 16
  python -m icms --scenario all --gpu-memory-utilization 0.6
        """,
    )
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        choices=["cpu", "ssd", "cpu_ssd", "all"],
        help="Offloading scenario: cpu / ssd / cpu_ssd / all",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-model-len", type=int, default=8000)
    parser.add_argument(
        "--num-requests",
        type=int,
        nargs="+",
        default=[1, 4, 16],
        help="List of concurrent request counts to test",
    )
    parser.add_argument(
        "--prefix-len",
        type=int,
        default=1500,
        help="Shared prefix token length (InferenceX-style, default: 1500)",
    )
    parser.add_argument(
        "--unique-len",
        type=int,
        default=500,
        help="Per-request unique suffix token length (default: 500)",
    )
    parser.add_argument("--max-cpu-size", type=float, default=10.0, help="Max CPU cache (GB)")
    parser.add_argument("--max-disk-size", type=float, default=10.0, help="Max disk cache (GB)")
    parser.add_argument("--disk-path", type=str, default="/tmp/icms_ssd_cache")
    parser.add_argument("--chunk-size", type=int, default=256, help="LMCache chunk size (tokens)")
    parser.add_argument("--offload-wait", type=float, default=5.0, help="Seconds to wait for offload")
    parser.add_argument("--output-dir", type=str, default="./icms_results")
    parser.add_argument("--no-warmup", action="store_true", help="Skip warmup generation")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = BenchmarkConfig(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        num_requests=args.num_requests,
        prefix_len=args.prefix_len,
        unique_len=args.unique_len,
        max_cpu_size_gb=args.max_cpu_size,
        max_disk_size_gb=args.max_disk_size,
        disk_path=args.disk_path,
        chunk_size=args.chunk_size,
        offload_wait_secs=args.offload_wait,
        warmup=not args.no_warmup,
        output_dir=args.output_dir,
    )

    print(cfg.describe())

    if args.scenario == "all":
        scenarios = ["cpu", "ssd", "cpu_ssd"]
    else:
        scenarios = [args.scenario]

    collector = MetricsCollector()

    for scenario in scenarios:
        run_single_scenario(scenario, cfg, collector)

    print_full_report(collector.snapshots, scenarios)

    save_results_json(collector.snapshots, asdict(cfg), cfg.output_dir)
    save_results_csv(collector.snapshots, cfg.output_dir)


if __name__ == "__main__":
    main()
