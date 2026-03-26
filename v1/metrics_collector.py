"""
ICMS Metrics Collector

Wraps LMCStatsMonitor to collect and store per-phase metrics snapshots
from the LMCache observability system. Captures all available fields
from LMCacheStats including lookup, remote, P2P, and eviction metrics.
"""

import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from lmcache.observability import LMCStatsMonitor, LMCacheStats


def _safe_avg(values: List[float]) -> float:
    """Compute average, returning 0.0 for empty lists."""
    return sum(values) / len(values) if values else 0.0


@dataclass
class MetricsSnapshot:
    """
    A snapshot of all available LMCache metrics captured at a specific phase.
    Fields map directly to LMCacheStats from lmcache/observability.py.
    """

    # ── Identification ────────────────────────────────────────────────────
    scenario: str    # "cpu", "ssd", or "cpu_ssd"
    phase: str       # "cold", "gpu_hit", or "offload_hit"
    num_requests: int
    wall_time_ms: float
    timestamp: float = field(default_factory=time.time)

    # ── Core request counters ──────────────────────────────────────────────
    retrieve_requests: int = 0
    store_requests: int = 0
    lookup_requests: int = 0

    # ── Token counters ─────────────────────────────────────────────────────
    requested_tokens: int = 0      # total tokens requested in retrieve
    hit_tokens: int = 0            # tokens retrieved from LMCache
    stored_tokens: int = 0         # tokens stored into LMCache
    lookup_tokens: int = 0         # total tokens sent to lookup
    lookup_hits: int = 0           # tokens found in lookup
    vllm_hit_tokens: int = 0       # tokens hit in vLLM's own prefix cache
    prompt_tokens: int = 0         # total prompt tokens processed

    # ── Hit rates ──────────────────────────────────────────────────────────
    retrieve_hit_rate: float = 0.0
    lookup_hit_rate: float = 0.0

    # ── Slow retrieval counters ────────────────────────────────────────────
    num_slow_retrieval_by_time: int = 0
    num_slow_retrieval_by_speed: int = 0

    # ── Timing distributions (averages in seconds) ─────────────────────────
    avg_time_to_retrieve: float = 0.0
    avg_time_to_store: float = 0.0
    avg_time_to_lookup: float = 0.0
    avg_retrieve_speed: float = 0.0   # tokens/sec
    avg_store_speed: float = 0.0      # tokens/sec

    # ── Granular profiling (averages in seconds) ───────────────────────────
    avg_retrieve_process_tokens_time: float = 0.0
    avg_retrieve_broadcast_time: float = 0.0
    avg_retrieve_to_gpu_time: float = 0.0
    avg_remote_backend_batched_get_blocking_time: float = 0.0
    avg_instrumented_connector_batched_get_time: float = 0.0
    avg_store_process_tokens_time: float = 0.0
    avg_store_from_gpu_time: float = 0.0
    avg_store_put_time: float = 0.0

    # ── Cache usage ────────────────────────────────────────────────────────
    local_cache_usage_bytes: int = 0
    remote_cache_usage_bytes: int = 0
    local_storage_usage_bytes: int = 0

    # ── Memory objects ─────────────────────────────────────────────────────
    active_memory_objs_count: int = 0
    pinned_memory_objs_count: int = 0

    # ── Eviction metrics ───────────────────────────────────────────────────
    cpu_evict_count: int = 0
    cpu_evict_keys_count: int = 0
    cpu_evict_failed_count: int = 0
    forced_unpin_count: int = 0

    # ── Remote backend metrics ─────────────────────────────────────────────
    remote_read_requests: int = 0
    remote_read_bytes: int = 0
    remote_write_requests: int = 0
    remote_write_bytes: int = 0
    avg_remote_time_to_get: float = 0.0
    avg_remote_time_to_put: float = 0.0
    avg_remote_time_to_get_sync: float = 0.0
    remote_ping_latency_ms: float = 0.0
    remote_ping_errors: int = 0
    remote_ping_success: int = 0

    # ── P2P transfer metrics ───────────────────────────────────────────────
    p2p_requests: int = 0
    p2p_transferred_tokens: int = 0
    avg_p2p_time_to_transfer: float = 0.0
    avg_p2p_transfer_speed: float = 0.0   # tokens/sec

    # ── Per-request lookup hit rate distribution ───────────────────────────
    avg_lookup_hit_rate_non_zero: float = 0.0   # avg over requests with >0 hit
    lookup_0_hit_requests: int = 0

    # ── Cache lifespan ─────────────────────────────────────────────────────
    avg_request_cache_lifespan_min: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class MetricsCollector:
    """Collects MetricsSnapshot from LMCStatsMonitor after each benchmark phase."""

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
        Drain the current LMCStatsMonitor state into a MetricsSnapshot.

        Calls get_stats_and_clear() so each snapshot captures only the metrics
        accumulated since the last collection (per-phase delta).
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
            lookup_requests=stats.interval_lookup_requests,
            # Token counters
            requested_tokens=stats.interval_requested_tokens,
            hit_tokens=stats.interval_hit_tokens,
            stored_tokens=stats.interval_stored_tokens,
            lookup_tokens=stats.interval_lookup_tokens,
            lookup_hits=stats.interval_lookup_hits,
            vllm_hit_tokens=stats.interval_vllm_hit_tokens,
            prompt_tokens=stats.interval_prompt_tokens,
            # Hit rates
            retrieve_hit_rate=stats.retrieve_hit_rate,
            lookup_hit_rate=stats.lookup_hit_rate,
            # Slow retrieval
            num_slow_retrieval_by_time=stats.interval_num_slow_retrieval_by_time,
            num_slow_retrieval_by_speed=stats.interval_num_slow_retrieval_by_speed,
            # Timing distributions
            avg_time_to_retrieve=_safe_avg(stats.time_to_retrieve),
            avg_time_to_store=_safe_avg(stats.time_to_store),
            avg_time_to_lookup=_safe_avg(stats.time_to_lookup),
            avg_retrieve_speed=_safe_avg(stats.retrieve_speed),
            avg_store_speed=_safe_avg(stats.store_speed),
            # Granular profiling
            avg_retrieve_process_tokens_time=_safe_avg(stats.retrieve_process_tokens_time),
            avg_retrieve_broadcast_time=_safe_avg(stats.retrieve_broadcast_time),
            avg_retrieve_to_gpu_time=_safe_avg(stats.retrieve_to_gpu_time),
            avg_remote_backend_batched_get_blocking_time=_safe_avg(
                stats.remote_backend_batched_get_blocking_time
            ),
            avg_instrumented_connector_batched_get_time=_safe_avg(
                stats.instrumented_connector_batched_get_time
            ),
            avg_store_process_tokens_time=_safe_avg(stats.store_process_tokens_time),
            avg_store_from_gpu_time=_safe_avg(stats.store_from_gpu_time),
            avg_store_put_time=_safe_avg(stats.store_put_time),
            # Cache usage
            local_cache_usage_bytes=stats.local_cache_usage_bytes,
            remote_cache_usage_bytes=stats.remote_cache_usage_bytes,
            local_storage_usage_bytes=stats.local_storage_usage_bytes,
            # Memory objects
            active_memory_objs_count=stats.active_memory_objs_count,
            pinned_memory_objs_count=stats.pinned_memory_objs_count,
            # Eviction
            cpu_evict_count=stats.interval_local_cpu_evict_count,
            cpu_evict_keys_count=stats.interval_local_cpu_evict_keys_count,
            cpu_evict_failed_count=stats.interval_local_cpu_evict_failed_count,
            forced_unpin_count=stats.interval_forced_unpin_count,
            # Remote backend
            remote_read_requests=stats.interval_remote_read_requests,
            remote_read_bytes=stats.interval_remote_read_bytes,
            remote_write_requests=stats.interval_remote_write_requests,
            remote_write_bytes=stats.interval_remote_write_bytes,
            avg_remote_time_to_get=_safe_avg(stats.interval_remote_time_to_get),
            avg_remote_time_to_put=_safe_avg(stats.interval_remote_time_to_put),
            avg_remote_time_to_get_sync=_safe_avg(stats.interval_remote_time_to_get_sync),
            remote_ping_latency_ms=stats.interval_remote_ping_latency,
            remote_ping_errors=stats.interval_remote_ping_errors,
            remote_ping_success=stats.interval_remote_ping_success,
            # P2P transfer
            p2p_requests=stats.interval_p2p_requests,
            p2p_transferred_tokens=stats.interval_p2p_transferred_tokens,
            avg_p2p_time_to_transfer=_safe_avg(stats.p2p_time_to_transfer),
            avg_p2p_transfer_speed=_safe_avg(stats.p2p_transfer_speed),
            # Lookup hit rate distribution
            avg_lookup_hit_rate_non_zero=_safe_avg(stats.interval_lookup_hit_rates),
            lookup_0_hit_requests=stats.interval_lookup_0_hit_requests,
            # Cache lifespan
            avg_request_cache_lifespan_min=_safe_avg(
                list(stats.interval_request_cache_lifespan.values())
                if isinstance(stats.interval_request_cache_lifespan, dict)
                else stats.interval_request_cache_lifespan
            ),
        )

        self.snapshots.append(snapshot)
        return snapshot

    def get_snapshots(
        self,
        scenario: Optional[str] = None,
        phase: Optional[str] = None,
    ) -> List[MetricsSnapshot]:
        """Filter snapshots by scenario and/or phase."""
        results = self.snapshots
        if scenario is not None:
            results = [s for s in results if s.scenario == scenario]
        if phase is not None:
            results = [s for s in results if s.phase == phase]
        return results

    def get_all_as_dicts(self) -> List[dict]:
        return [s.to_dict() for s in self.snapshots]

    def clear(self):
        self.snapshots.clear()
