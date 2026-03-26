"""
ICMS — KV Cache Offloading Benchmark

Tests LMCache KV cache offloading under three hardware configurations:

  Case 1  cpu      GPU + CPU memory
  Case 2  ssd      GPU + SSD disk (CPU staging bypassed)
  Case 3  cpu_ssd  GPU + CPU + SSD (tiered: CPU=L2, SSD=L3)

Queries follow the InferenceX pattern:
  all N requests share a common prefix (prefix_len tokens) then each
  appends a unique suffix (suffix_len tokens).  This maximises KV cache
  sharing and gives realistic cache-hit behaviour.

Three phases per (scenario × num_requests):
  cold         fresh prefill; KV stored to offload target
  gpu_hit      same prompts while vLLM GPU prefix cache is warm
  offload_hit  GPU cache cleared; KV fetched back via LMCache

Usage:
  python -m icms --scenario cpu
  python -m icms --scenario ssd --disk-path /mnt/nvme/kvcache
  python -m icms --scenario cpu_ssd --num-requests 1 4 16
  python -m icms --scenario all --prefix-len 2000 --suffix-len 500

  # Low-VRAM / WSL2 local dev (applies memory-saving workarounds):
  python -m icms --scenario cpu --local-dev --model facebook/opt-125m \\
      --gpu-mem 0.45 --max-model-len 512 --max-cpu-gb 1.0
"""

import argparse
from dataclasses import asdict, dataclass
from typing import List

from icms.env import make_scenario
from icms.metrics import Collector
from icms.output import print_report, save
from icms.runner import run_scenario


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m icms",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--scenario",
        required=True,
        choices=["cpu", "ssd", "cpu_ssd", "all"],
        help="Which offloading scenario(s) to run.",
    )
    p.add_argument("--model",              default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--gpu-mem",            type=float, default=0.5,
                   help="vLLM gpu_memory_utilization (default 0.5)")
    p.add_argument("--max-model-len",      type=int,   default=8000)
    p.add_argument("--num-requests",       type=int,   nargs="+", default=[1, 4, 16],
                   metavar="N",
                   help="Concurrent request counts to test (default: 1 4 16)")
    p.add_argument("--prefix-len",         type=int,   default=1500,
                   help="Shared prefix token length (default: 1500)")
    p.add_argument("--suffix-len",         type=int,   default=500,
                   help="Per-request unique suffix token length (default: 500)")
    p.add_argument("--max-output-tokens",  type=int,   default=1)
    p.add_argument("--chunk-size",         type=int,   default=256,
                   help="LMCache chunk size in tokens (default: 256)")
    p.add_argument("--max-cpu-gb",         type=float, default=10.0)
    p.add_argument("--max-disk-gb",        type=float, default=10.0)
    p.add_argument("--disk-path",          default="/tmp/icms_kvcache")
    p.add_argument("--offload-wait",       type=float, default=5.0,
                   help="Seconds to wait after GPU cache reset (default: 5.0)")
    p.add_argument("--output-dir",         default="./icms_results")
    p.add_argument("--no-warmup",          action="store_true")
    p.add_argument(
        "--local-dev",
        action="store_true",
        help=(
            "Apply low-VRAM workarounds for local / WSL2 development: "
            "CPU KV buffer, smaller KV memory, Triton attention backend, "
            "eager mode. Remove this flag on production servers."
        ),
    )
    return p.parse_args()


# ── config snapshot for JSON output ──────────────────────────────────────────

@dataclass
class RunConfig:
    model: str
    gpu_mem: float
    max_model_len: int
    num_requests: List[int]
    prefix_len: int
    suffix_len: int
    max_output_tokens: int
    chunk_size: int
    max_cpu_gb: float
    max_disk_gb: float
    disk_path: str
    offload_wait: float
    scenarios: List[str]
    warmup: bool
    output_dir: str
    local_dev: bool


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse()

    scenarios = ["cpu", "ssd", "cpu_ssd"] if args.scenario == "all" else [args.scenario]

    cfg = RunConfig(
        model=args.model,
        gpu_mem=args.gpu_mem,
        max_model_len=args.max_model_len,
        num_requests=args.num_requests,
        prefix_len=args.prefix_len,
        suffix_len=args.suffix_len,
        max_output_tokens=args.max_output_tokens,
        chunk_size=args.chunk_size,
        max_cpu_gb=args.max_cpu_gb,
        max_disk_gb=args.max_disk_gb,
        disk_path=args.disk_path,
        offload_wait=args.offload_wait,
        scenarios=scenarios,
        warmup=not args.no_warmup,
        output_dir=args.output_dir,
        local_dev=args.local_dev,
    )

    # print config summary
    print("\n" + "═" * 50)
    print("  ICMS — KV Cache Offloading Benchmark")
    print("═" * 50)
    print(f"  model          {cfg.model}")
    print(f"  scenarios      {', '.join(scenarios)}")
    print(f"  requests       {cfg.num_requests}")
    print(f"  prompt shape   {cfg.prefix_len} (shared) + {cfg.suffix_len} (unique) "
          f"= {cfg.prefix_len + cfg.suffix_len} tokens")
    print(f"  output tokens  {cfg.max_output_tokens}")
    print(f"  chunk size     {cfg.chunk_size}")
    print(f"  CPU max        {cfg.max_cpu_gb} GB")
    print(f"  disk max       {cfg.max_disk_gb} GB  ({cfg.disk_path})")
    print(f"  offload wait   {cfg.offload_wait}s")
    if cfg.local_dev:
        print(f"  *** local-dev mode ON (low-VRAM workarounds active) ***")
    print("═" * 50 + "\n")

    collector = Collector()

    for sc_name in scenarios:
        sc = make_scenario(
            name=sc_name,
            chunk_size=cfg.chunk_size,
            max_cpu_gb=cfg.max_cpu_gb,
            disk_base=cfg.disk_path,
            max_disk_gb=cfg.max_disk_gb,
        )
        run_scenario(
            sc,
            model=cfg.model,
            gpu_mem=cfg.gpu_mem,
            max_model_len=cfg.max_model_len,
            request_counts=cfg.num_requests,
            prefix_len=cfg.prefix_len,
            suffix_len=cfg.suffix_len,
            max_output_tokens=cfg.max_output_tokens,
            offload_wait_s=cfg.offload_wait,
            warmup=cfg.warmup,
            collector=collector,
            local_dev=cfg.local_dev,
        )

    print_report(scenarios, collector)
    save(collector, asdict(cfg), cfg.output_dir)


if __name__ == "__main__":
    main()
