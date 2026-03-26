"""
ICMS Benchmark Configuration

Dataclass-based configuration for KV cache offloading benchmarks.
All parameters can be overridden via CLI arguments in benchmark.py.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class BenchmarkConfig:
    """Configuration for KV cache offloading benchmark."""

    # Model settings
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gpu_memory_utilization: float = 0.5
    max_model_len: int = 8000

    # Benchmark parameters
    num_requests: List[int] = field(default_factory=lambda: [1, 4, 16])
    prompt_len: int = 2000
    max_tokens: int = 1
    temperature: float = 0.0

    # LMCache settings
    chunk_size: int = 256
    max_cpu_size_gb: float = 10.0
    max_disk_size_gb: float = 10.0
    disk_path: str = "/tmp/icms_ssd_cache"

    # Timing
    offload_wait_secs: float = 5.0
    warmup: bool = True

    # Output
    output_dir: str = "./icms_results"

    def describe(self) -> str:
        """Return a human-readable description of the configuration."""
        lines = [
            "=== Benchmark Configuration ===",
            f"  Model:                {self.model}",
            f"  GPU Mem Utilization:  {self.gpu_memory_utilization}",
            f"  Max Model Len:        {self.max_model_len}",
            f"  Num Requests:         {self.num_requests}",
            f"  Prompt Length:         {self.prompt_len}",
            f"  Max Tokens:           {self.max_tokens}",
            f"  Chunk Size:           {self.chunk_size}",
            f"  Max CPU Size (GB):    {self.max_cpu_size_gb}",
            f"  Max Disk Size (GB):   {self.max_disk_size_gb}",
            f"  Disk Path:            {self.disk_path}",
            f"  Offload Wait (sec):   {self.offload_wait_secs}",
            f"  Output Dir:           {self.output_dir}",
            "=" * 32,
        ]
        return "\n".join(lines)
