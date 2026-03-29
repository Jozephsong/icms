"""
LMCache environment configuration for each offloading scenario.

Three cases:
  cpu     — GPU KV cache spills to CPU memory (RAM)
  ssd     — GPU KV cache spills directly to disk (no CPU staging)
  cpu_ssd — GPU → CPU → disk, tiered (CPU serves as L2, disk as L3)
"""

import os
from dataclasses import dataclass


@dataclass
class ScenarioConfig:
    name: str          # "cpu" | "ssd" | "cpu_ssd"
    label: str         # human-readable
    chunk_size: int
    cpu_enabled: bool
    max_cpu_gb: float
    disk_enabled: bool
    disk_dir: str      # absolute path, empty string if disabled
    max_disk_gb: float


def _wipe():
    """Remove all LMCache env vars so runs don't bleed into each other."""
    for key in [
        "LMCACHE_USE_EXPERIMENTAL",
        "LMCACHE_CHUNK_SIZE",
        "LMCACHE_LOCAL_CPU",
        "LMCACHE_MAX_LOCAL_CPU_SIZE",
        "LMCACHE_LOCAL_DISK",
        "LMCACHE_MAX_LOCAL_DISK_SIZE",
    ]:
        os.environ.pop(key, None)


def apply(sc: ScenarioConfig) -> None:
    """
    Apply the given scenario's settings to the process environment.
    Must be called before vLLM / LMCache initialisation.
    """
    _wipe()

    os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"
    os.environ["LMCACHE_CHUNK_SIZE"] = str(sc.chunk_size)
    os.environ["LMCACHE_LOCAL_CPU"] = "True" if sc.cpu_enabled else "False"

    if sc.cpu_enabled:
        os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = str(sc.max_cpu_gb)

    if sc.disk_enabled:
        os.makedirs(sc.disk_dir, exist_ok=True)
        os.environ["LMCACHE_LOCAL_DISK"] = f"file://{sc.disk_dir}/"
        os.environ["LMCACHE_MAX_LOCAL_DISK_SIZE"] = str(sc.max_disk_gb)

    parts = [f"chunk={sc.chunk_size}"]
    if sc.cpu_enabled:
        parts.append(f"cpu_max={sc.max_cpu_gb}GB")
    if sc.disk_enabled:
        parts.append(f"disk={sc.disk_dir}  max={sc.max_disk_gb}GB")
    print(f"[env] {sc.label}: {', '.join(parts)}")


def make_scenario(
    name: str,
    chunk_size: int,
    max_cpu_gb: float,
    disk_base: str,
    max_disk_gb: float,
) -> ScenarioConfig:
    """
    Factory that builds a ScenarioConfig from the four shared knobs.

    name must be one of: "cpu", "ssd", "cpu_ssd"
    """
    labels = {
        "cpu":     "Case 1 — GPU + CPU",
        "ssd":     "Case 2 — GPU + SSD",
        "cpu_ssd": "Case 3 — GPU + CPU + SSD",
    }
    if name not in labels:
        raise ValueError(f"Unknown scenario '{name}'. Choose: cpu, ssd, cpu_ssd")

    cpu_on     = name in ("cpu", "cpu_ssd")
    disk_on    = name in ("ssd", "cpu_ssd")
    disk_dir   = os.path.join(disk_base, f"lmcache_{name}") if disk_on else ""

    return ScenarioConfig(
        name=name,
        label=labels[name],
        chunk_size=chunk_size,
        cpu_enabled=cpu_on,
        max_cpu_gb=max_cpu_gb,
        disk_enabled=disk_on,
        disk_dir=disk_dir,
        max_disk_gb=max_disk_gb,
    )
