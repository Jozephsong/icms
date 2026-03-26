"""
ICMS Metrics Collector

Wraps LMCStatsMonitor to collect and store per-phase metrics snapshots
from the LMCache observability system.
"""

import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from lmcache.observability import LMCStatsMonitor, LMCacheStats


@dataclass
class MetricsSnapshot:
    """A snapshot of LMCache metrics captured at a specific phase."""

    scenario: str  # "cpu" or "ssd"
    phase: str  # "cold", "gpu_hit", "offload_hit"
    num_requests: int
    wall_time_ms: float  # wall-clock time for the phase
    timestamp: float = field(default_factory=time.time)

    # Core counters
    retrieve_requests: int = 0
    store_requests: int = 0
    requested_tokens: int = 0
    hit_tokens: int = 0
    stored_tokens: int = 0

    # Hit rates
    retrieve_hit_rate: float = 0.0

    # Timing distributions (averages in seconds)
    avg_time_to_retrieve: float = 0.0
    avg_time_to_store: float = 0.0
    avg_retrieve_speed: float = 0.0  # tokens/sec
    avg_store_speed: float = 0.0  # tokens/sec

    # Granular profiling (averages in seconds)
    avg_retrieve_process_tokens_time: float = 0.0
    avg_retrieve_to_gpu_time: float = 0.0
    avg_store_from_gpu_time: float = 0.0
    avg_store_put_time: float = 0.0

    # Cache usage
    local_cache_usage_bytes: int = 0
    local_storage_usage_bytes: int = 0

    # Eviction metrics
    cpu_evict_count: int = 0
    cpu_evict_keys_count: int = 0

    # Memory objects
    active_memory_objs_count: int = 0
    pinned_memory_objs_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _safe_avg(values: List[float]) -> float:
    """Compute average, returning 0.0 for empty lists."""
    if not values:
        return 0.0
    return sum(values) / len(values)


class MetricsCollector:
    """Collects metrics snapshots from LMCStatsMonitor after each benchmark phase."""

    def __init__(self):
        self.snapshots: List[MetricsSnapshot] = []

    def collect_snapshot(
        self,
        scenario: str,
        phase: str,
        num_requests: int,
        wall_time_ms: float,
    ) -> MetricsSnapshot:
        """
        Collect a metrics snapshot from the current LMCStatsMonitor state.

        Calls get_stats_and_clear() so each snapshot captures only the metrics
        accumulated since the last collection.
        """
        monitor = LMCStatsMonitor.GetOrCreate()
        stats: LMCacheStats = monitor.get_stats_and_clear()

        snapshot = MetricsSnapshot(
            scenario=scenario,
            phase=phase,
            num_requests=num_requests,
            wall_time_ms=wall_time_ms,
            # Core counters
            retrieve_requests=stats.interval_retrieve_requests,
            store_requests=stats.interval_store_requests,
            requested_tokens=stats.interval_requested_tokens,
            hit_tokens=stats.interval_hit_tokens,
            stored_tokens=stats.interval_stored_tokens,
            # Hit rates
            retrieve_hit_rate=stats.retrieve_hit_rate,
            # Timing distributions
            avg_time_to_retrieve=_safe_avg(stats.time_to_retrieve),
            avg_time_to_store=_safe_avg(stats.time_to_store),
            avg_retrieve_speed=_safe_avg(stats.retrieve_speed),
            avg_store_speed=_safe_avg(stats.store_speed),
            # Granular profiling
            avg_retrieve_process_tokens_time=_safe_avg(
                stats.retrieve_process_tokens_time
            ),
            avg_retrieve_to_gpu_time=_safe_avg(stats.retrieve_to_gpu_time),
            avg_store_from_gpu_time=_safe_avg(stats.store_from_gpu_time),
            avg_store_put_time=_safe_avg(stats.store_put_time),
            # Cache usage
            local_cache_usage_bytes=stats.local_cache_usage_bytes,
            local_storage_usage_bytes=stats.local_storage_usage_bytes,
            # Eviction metrics
            cpu_evict_count=stats.interval_local_cpu_evict_count,
            cpu_evict_keys_count=stats.interval_local_cpu_evict_keys_count,
            # Memory objects
            active_memory_objs_count=stats.active_memory_objs_count,
            pinned_memory_objs_count=stats.pinned_memory_objs_count,
        )

        self.snapshots.append(snapshot)
        return snapshot

    def get_snapshots(
        self,
        scenario: Optional[str] = None,
        phase: Optional[str] = None,
    ) -> List[MetricsSnapshot]:
        """Filter and return snapshots by scenario and/or phase."""
        results = self.snapshots
        if scenario is not None:
            results = [s for s in results if s.scenario == scenario]
        if phase is not None:
            results = [s for s in results if s.phase == phase]
        return results

    def get_all_as_dicts(self) -> List[dict]:
        """Return all snapshots as list of dicts (for JSON export)."""
        return [s.to_dict() for s in self.snapshots]

    def clear(self):
        """Clear all collected snapshots."""
        self.snapshots.clear()
