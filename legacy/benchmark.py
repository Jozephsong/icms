"""
ICMS KV Cache Offloading Benchmark

Main entry point for benchmarking KV cache offloading in vLLM with LMCache.
Supports two scenarios:
  1. CPU: GPU → CPU memory offloading
  2. SSD: GPU → Disk offloading (bypassing CPU)

Usage:
    python -m icms.benchmark --scenario cpu --num-requests 1 4 16
    python -m icms.benchmark --scenario ssd --num-requests 1 4 --disk-path /mnt/nvme/cache
    python -m icms.benchmark --scenario all --num-requests 1 4 16
"""

import argparse
import contextlib
import os
import sys
import time
from dataclasses import asdict
from typing import List

from vllm import LLM, SamplingParams, TokensPrompt
from vllm.config import KVTransferConfig

from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.observability import LMCStatsMonitor
from lmcache.v1.cache_engine import LMCacheEngineBuilder

from icms.config import BenchmarkConfig
from icms.metrics_collector import MetricsCollector
from icms.report import print_full_report, save_results_json


# ──────────────────────────────────────────────
# Environment Setup
# ──────────────────────────────────────────────

def _clear_lmcache_env():
    """Remove all LMCache-related environment variables to prevent leaking."""
    keys_to_clear = [
        "LMCACHE_LOCAL_CPU",
        "LMCACHE_LOCAL_DISK",
        "LMCACHE_MAX_LOCAL_CPU_SIZE",
        "LMCACHE_MAX_LOCAL_DISK_SIZE",
        "LMCACHE_GDS_PATH",
        "LMCACHE_CHUNK_SIZE",
        "LMCACHE_USE_EXPERIMENTAL",
    ]
    for key in keys_to_clear:
        os.environ.pop(key, None)


def setup_env_cpu(cfg: BenchmarkConfig):
    """Configure environment for GPU + CPU offloading."""
    _clear_lmcache_env()
    os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"
    os.environ["LMCACHE_CHUNK_SIZE"] = str(cfg.chunk_size)
    os.environ["LMCACHE_LOCAL_CPU"] = "True"
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = str(cfg.max_cpu_size_gb)
    print(f"[ENV] CPU offloading: chunk_size={cfg.chunk_size}, "
          f"max_cpu_size={cfg.max_cpu_size_gb} GB")


def setup_env_ssd(cfg: BenchmarkConfig):
    """Configure environment for GPU + SSD offloading (bypassing CPU)."""
    _clear_lmcache_env()
    os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"
    os.environ["LMCACHE_CHUNK_SIZE"] = str(cfg.chunk_size)
    os.environ["LMCACHE_LOCAL_CPU"] = "False"
    disk_path = os.path.join(cfg.disk_path, "lmcache_ssd")
    os.makedirs(disk_path, exist_ok=True)
    os.environ["LMCACHE_LOCAL_DISK"] = f"file://{disk_path}/"
    os.environ["LMCACHE_MAX_LOCAL_DISK_SIZE"] = str(cfg.max_disk_size_gb)
    print(f"[ENV] SSD offloading: chunk_size={cfg.chunk_size}, "
          f"disk_path={disk_path}, max_disk_size={cfg.max_disk_size_gb} GB")


# ──────────────────────────────────────────────
# LLM Lifecycle
# ──────────────────────────────────────────────

@contextlib.contextmanager
def build_llm(cfg: BenchmarkConfig):
    """Build and manage the lifecycle of vLLM + LMCache LLM instance."""
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
# Prompt Generation
# ──────────────────────────────────────────────

def make_prompts(num: int, length: int) -> List[TokensPrompt]:
    """Generate unique token prompts of given length."""
    return [
        TokensPrompt(prompt_token_ids=[i + 1] + [0] * (length - 1))
        for i in range(num)
    ]


# ──────────────────────────────────────────────
# Benchmark Execution
# ──────────────────────────────────────────────

def run_single_scenario(
    scenario: str,
    cfg: BenchmarkConfig,
    collector: MetricsCollector,
):
    """
    Run a full benchmark for a single scenario (cpu or ssd).

    For each num_requests value, executes three phases:
      1. Cold Run:        Generate with no prior cache
      2. GPU Hit Run:     Re-run same prompts (GPU prefix cache hit)
      3. Offload Hit Run: Reset GPU cache, wait for offload, re-run (LMCache hit)
    """
    # Setup environment
    if scenario == "cpu":
        setup_env_cpu(cfg)
    elif scenario == "ssd":
        setup_env_ssd(cfg)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    sampling_params = SamplingParams(
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )

    # Drain any stale metrics before starting
    monitor = LMCStatsMonitor.GetOrCreate()
    monitor.get_stats_and_clear()

    with build_llm(cfg) as llm:
        for req_idx, num_reqs in enumerate(cfg.num_requests):
            print(f"\n{'━' * 60}")
            print(f"  Scenario={scenario.upper()}, Requests={num_reqs}, "
                  f"prompt_len={cfg.prompt_len}")
            print(f"{'━' * 60}")

            prompts = make_prompts(num_reqs, cfg.prompt_len)

            # ─── Warmup (only on first iteration) ───
            if cfg.warmup and req_idx == 0:
                print("  [Warmup] Running warmup generation...")
                warmup_prompt = [TokensPrompt(
                    prompt_token_ids=[9999] + [0] * (cfg.prompt_len - 1)
                )]
                llm.generate(warmup_prompt, sampling_params, use_tqdm=False)
                llm.reset_prefix_cache()
                # Drain warmup metrics
                monitor.get_stats_and_clear()
                time.sleep(2.0)

            # ─── Phase 1: Cold Run ───
            print("  [Phase 1] Cold Run — prefill & store KV cache...")
            # Drain metrics before phase
            monitor.get_stats_and_clear()
            start = time.perf_counter()
            llm.generate(prompts, sampling_params, use_tqdm=False)
            wall_ms = (time.perf_counter() - start) * 1000
            snapshot = collector.collect_snapshot(scenario, "cold", num_reqs, wall_ms)
            print(f"    Time: {wall_ms:.2f} ms | "
                  f"Stored: {snapshot.stored_tokens} tokens")

            # ─── Phase 2: GPU Hit Run ───
            print("  [Phase 2] GPU Hit Run — reading from GPU prefix cache...")
            start = time.perf_counter()
            llm.generate(prompts, sampling_params, use_tqdm=False)
            wall_ms = (time.perf_counter() - start) * 1000
            snapshot = collector.collect_snapshot(
                scenario, "gpu_hit", num_reqs, wall_ms
            )
            print(f"    Time: {wall_ms:.2f} ms | "
                  f"Hit Rate: {snapshot.retrieve_hit_rate:.4f}")

            # ─── Clear GPU prefix cache ───
            print("  [Clear] Resetting GPU prefix cache...")
            llm.reset_prefix_cache()

            print(f"  [Wait] Waiting {cfg.offload_wait_secs}s for offload "
                  f"tasks to complete...")
            time.sleep(cfg.offload_wait_secs)

            # ─── Phase 3: Offload Hit Run ───
            print(f"  [Phase 3] Offload Hit Run — reading from "
                  f"{scenario.upper()} storage...")
            start = time.perf_counter()
            llm.generate(prompts, sampling_params, use_tqdm=False)
            wall_ms = (time.perf_counter() - start) * 1000
            snapshot = collector.collect_snapshot(
                scenario, "offload_hit", num_reqs, wall_ms
            )
            print(f"    Time: {wall_ms:.2f} ms | "
                  f"Hit Rate: {snapshot.retrieve_hit_rate:.4f} | "
                  f"Retrieve Speed: {snapshot.avg_retrieve_speed:.1f} tok/s")

            # Reset for next iteration
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
  python -m icms.benchmark --scenario cpu --num-requests 1 4
  python -m icms.benchmark --scenario ssd --disk-path /mnt/nvme/cache
  python -m icms.benchmark --scenario all --gpu-memory-utilization 0.6
        """,
    )
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        choices=["cpu", "ssd", "all"],
        help="Offloading scenario to test: cpu, ssd, or all (both)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Model to benchmark (default: Meta-Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.5,
        help="GPU memory utilization for vLLM (default: 0.5)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8000,
        help="Maximum model sequence length (default: 8000)",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        nargs="+",
        default=[1, 4, 16],
        help="List of request counts to test (default: 1 4 16)",
    )
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=2000,
        help="Prompt length in tokens (default: 2000)",
    )
    parser.add_argument(
        "--max-cpu-size",
        type=float,
        default=10.0,
        help="Maximum CPU cache size in GB (default: 10.0)",
    )
    parser.add_argument(
        "--max-disk-size",
        type=float,
        default=10.0,
        help="Maximum disk cache size in GB (default: 10.0)",
    )
    parser.add_argument(
        "--disk-path",
        type=str,
        default="/tmp/icms_ssd_cache",
        help="Path for SSD/disk cache storage (default: /tmp/icms_ssd_cache)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="LMCache chunk size in tokens (default: 256)",
    )
    parser.add_argument(
        "--offload-wait",
        type=float,
        default=5.0,
        help="Seconds to wait for offload completion (default: 5.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./icms_results",
        help="Directory to save JSON results (default: ./icms_results)",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip warmup generation",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = BenchmarkConfig(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        num_requests=args.num_requests,
        prompt_len=args.prompt_len,
        max_cpu_size_gb=args.max_cpu_size,
        max_disk_size_gb=args.max_disk_size,
        disk_path=args.disk_path,
        chunk_size=args.chunk_size,
        offload_wait_secs=args.offload_wait,
        warmup=not args.no_warmup,
        output_dir=args.output_dir,
    )

    print(cfg.describe())

    scenarios: List[str] = []
    if args.scenario == "all":
        scenarios = ["cpu", "ssd"]
    else:
        scenarios = [args.scenario]

    collector = MetricsCollector()

    for scenario in scenarios:
        run_single_scenario(scenario, cfg, collector)

    # ─── Report ───
    print_full_report(collector.snapshots, scenarios)

    # ─── Save JSON ───
    save_results_json(
        collector.snapshots,
        asdict(cfg),
        cfg.output_dir,
    )


if __name__ == "__main__":
    main()
