"""
LMCache metrics snapshot.

Maps every field of LMCacheStats (lmcache/observability.py) into a flat,
serialisable dataclass so nothing is lost between collection and output.

LMCacheStats fields covered
────────────────────────────
Interval counters:
  interval_retrieve_requests, interval_store_requests,
  interval_lookup_requests,
  interval_requested_tokens, interval_hit_tokens, interval_stored_tokens,
  interval_lookup_tokens, interval_lookup_hits,
  interval_vllm_hit_tokens, interval_prompt_tokens,
  interval_num_slow_retrieval_by_time, interval_num_slow_retrieval_by_speed,
  interval_remote_read_requests, interval_remote_read_bytes,
  interval_remote_write_requests, interval_remote_write_bytes,
  interval_remote_ping_latency, interval_remote_ping_errors,
  interval_remote_ping_success, interval_remote_ping_error_code,
  interval_local_cpu_evict_count, interval_local_cpu_evict_keys_count,
  interval_local_cpu_evict_failed_count,
  interval_forced_unpin_count,
  interval_p2p_requests, interval_p2p_transferred_tokens,
  interval_lookup_0_hit_requests

Snapshot (real-time) values:
  retrieve_hit_rate, lookup_hit_rate,
  local_cache_usage_bytes, remote_cache_usage_bytes,
  local_storage_usage_bytes,
  active_memory_objs_count, pinned_memory_objs_count

Distribution averages (List[float] → single avg):
  time_to_retrieve, time_to_store, time_to_lookup,
  retrieve_speed, store_speed,
  retrieve_process_tokens_time, retrieve_broadcast_time, retrieve_to_gpu_time,
  remote_backend_batched_get_blocking_time,
  instrumented_connector_batched_get_time,
  store_process_tokens_time, store_from_gpu_time, store_put_time,
  interval_remote_time_to_get, interval_remote_time_to_put,
  interval_remote_time_to_get_sync,
  p2p_time_to_transfer, p2p_transfer_speed,
  interval_lookup_hit_rates,
  interval_request_cache_lifespan
"""

import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from lmcache.observability import LMCacheStats, LMCStatsMonitor


def _avg(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _lifespan_values(v) -> List[float]:
    """LMCacheStats.interval_request_cache_lifespan is List[float] in the dataclass."""
    if isinstance(v, dict):
        return list(v.values())
    return list(v) if v else []


@dataclass
class Snapshot:
    # ── identity ──────────────────────────────────────────────────────────
    scenario: str        # "cpu" | "ssd" | "cpu_ssd"
    phase: str           # "cold" | "gpu_hit" | "offload_hit"
    num_requests: int
    wall_ms: float
    ts: float = field(default_factory=time.time)

    # ── request counters ──────────────────────────────────────────────────
    n_retrieve: int = 0
    n_store:    int = 0
    n_lookup:   int = 0

    # ── token counters ─────────────────────────────────────────────────────
    tok_requested:  int = 0   # tokens sent to retrieve
    tok_hit:        int = 0   # tokens returned by retrieve (LMCache hit)
    tok_stored:     int = 0   # tokens saved to LMCache
    tok_lookup:     int = 0   # tokens sent to lookup
    tok_lookup_hit: int = 0   # tokens found by lookup
    tok_vllm_hit:   int = 0   # tokens already in vLLM GPU prefix cache
    tok_prompt:     int = 0   # total prompt tokens

    # ── hit rates ─────────────────────────────────────────────────────────
    retrieve_hit_rate: float = 0.0
    lookup_hit_rate:   float = 0.0

    # ── slow-retrieval flags ───────────────────────────────────────────────
    slow_by_time:  int = 0
    slow_by_speed: int = 0

    # ── latency averages (seconds) ────────────────────────────────────────
    avg_retrieve_s:  float = 0.0
    avg_store_s:     float = 0.0
    avg_lookup_s:    float = 0.0

    # ── throughput averages (tokens/s) ───────────────────────────────────
    avg_retrieve_tps: float = 0.0
    avg_store_tps:    float = 0.0

    # ── granular pipeline timings (seconds) ──────────────────────────────
    avg_retr_process_tok_s:  float = 0.0   # token processing inside retrieve
    avg_retr_broadcast_s:    float = 0.0   # broadcast to other workers
    avg_retr_to_gpu_s:       float = 0.0   # memcpy → GPU
    avg_remote_batch_get_s:  float = 0.0   # remote backend batched-get blocking
    avg_connector_batch_s:   float = 0.0   # instrumented connector batched-get
    avg_store_process_tok_s: float = 0.0   # token processing inside store
    avg_store_from_gpu_s:    float = 0.0   # memcpy GPU →
    avg_store_put_s:         float = 0.0   # put into storage backend

    # ── cache occupancy ───────────────────────────────────────────────────
    cpu_bytes:    int = 0   # local CPU cache used
    disk_bytes:   int = 0   # local disk/SSD cache used
    remote_bytes: int = 0   # remote cache used

    # ── memory object tracking ────────────────────────────────────────────
    mem_active: int = 0
    mem_pinned: int = 0

    # ── eviction ──────────────────────────────────────────────────────────
    evict_count:        int = 0
    evict_keys:         int = 0
    evict_failed:       int = 0
    forced_unpin:       int = 0

    # ── remote backend I/O ────────────────────────────────────────────────
    remote_read_reqs:   int   = 0
    remote_read_bytes:  int   = 0
    remote_write_reqs:  int   = 0
    remote_write_bytes: int   = 0
    avg_remote_get_s:   float = 0.0
    avg_remote_put_s:   float = 0.0
    avg_remote_get_sync_s: float = 0.0
    ping_latency_ms:    float = 0.0
    ping_errors:        int   = 0
    ping_ok:            int   = 0

    # ── P2P transfer ──────────────────────────────────────────────────────
    p2p_reqs:        int   = 0
    p2p_tok:         int   = 0
    avg_p2p_s:       float = 0.0
    avg_p2p_tps:     float = 0.0

    # ── per-request lookup hit-rate distribution ──────────────────────────
    lookup_0hit_reqs:    int   = 0
    avg_lookup_hit_rate: float = 0.0   # average over requests with > 0 hit

    # ── cache lifespan ────────────────────────────────────────────────────
    avg_lifespan_min: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


class Collector:
    """
    Drains LMCStatsMonitor at the end of each phase and stores the result.
    Each call to record() captures only the delta since the last call.
    """

    def __init__(self):
        self._rows: List[Snapshot] = []

    def drain(self) -> None:
        """Discard accumulated stats without recording (use before timed sections)."""
        LMCStatsMonitor.GetOrCreate().get_stats_and_clear()

    def record(
        self,
        scenario: str,
        phase: str,
        num_requests: int,
        wall_ms: float,
    ) -> Snapshot:
        """Drain the monitor and convert its state into a Snapshot."""
        s: LMCacheStats = LMCStatsMonitor.GetOrCreate().get_stats_and_clear()

        snap = Snapshot(
            scenario=scenario,
            phase=phase,
            num_requests=num_requests,
            wall_ms=wall_ms,
            # request counters
            n_retrieve=s.interval_retrieve_requests,
            n_store=s.interval_store_requests,
            n_lookup=s.interval_lookup_requests,
            # token counters
            tok_requested=s.interval_requested_tokens,
            tok_hit=s.interval_hit_tokens,
            tok_stored=s.interval_stored_tokens,
            tok_lookup=s.interval_lookup_tokens,
            tok_lookup_hit=s.interval_lookup_hits,
            tok_vllm_hit=s.interval_vllm_hit_tokens,
            tok_prompt=s.interval_prompt_tokens,
            # hit rates
            retrieve_hit_rate=s.retrieve_hit_rate,
            lookup_hit_rate=s.lookup_hit_rate,
            # slow retrieval
            slow_by_time=s.interval_num_slow_retrieval_by_time,
            slow_by_speed=s.interval_num_slow_retrieval_by_speed,
            # latency
            avg_retrieve_s=_avg(s.time_to_retrieve),
            avg_store_s=_avg(s.time_to_store),
            avg_lookup_s=_avg(s.time_to_lookup),
            # throughput
            avg_retrieve_tps=_avg(s.retrieve_speed),
            avg_store_tps=_avg(s.store_speed),
            # granular pipeline
            avg_retr_process_tok_s=_avg(s.retrieve_process_tokens_time),
            avg_retr_broadcast_s=_avg(s.retrieve_broadcast_time),
            avg_retr_to_gpu_s=_avg(s.retrieve_to_gpu_time),
            avg_remote_batch_get_s=_avg(s.remote_backend_batched_get_blocking_time),
            avg_connector_batch_s=_avg(s.instrumented_connector_batched_get_time),
            avg_store_process_tok_s=_avg(s.store_process_tokens_time),
            avg_store_from_gpu_s=_avg(s.store_from_gpu_time),
            avg_store_put_s=_avg(s.store_put_time),
            # cache occupancy
            cpu_bytes=s.local_cache_usage_bytes,
            disk_bytes=s.local_storage_usage_bytes,
            remote_bytes=s.remote_cache_usage_bytes,
            # memory objects
            mem_active=s.active_memory_objs_count,
            mem_pinned=s.pinned_memory_objs_count,
            # eviction
            evict_count=s.interval_local_cpu_evict_count,
            evict_keys=s.interval_local_cpu_evict_keys_count,
            evict_failed=s.interval_local_cpu_evict_failed_count,
            forced_unpin=s.interval_forced_unpin_count,
            # remote I/O
            remote_read_reqs=s.interval_remote_read_requests,
            remote_read_bytes=s.interval_remote_read_bytes,
            remote_write_reqs=s.interval_remote_write_requests,
            remote_write_bytes=s.interval_remote_write_bytes,
            avg_remote_get_s=_avg(s.interval_remote_time_to_get),
            avg_remote_put_s=_avg(s.interval_remote_time_to_put),
            avg_remote_get_sync_s=_avg(s.interval_remote_time_to_get_sync),
            ping_latency_ms=s.interval_remote_ping_latency,
            ping_errors=s.interval_remote_ping_errors,
            ping_ok=s.interval_remote_ping_success,
            # P2P
            p2p_reqs=s.interval_p2p_requests,
            p2p_tok=s.interval_p2p_transferred_tokens,
            avg_p2p_s=_avg(s.p2p_time_to_transfer),
            avg_p2p_tps=_avg(s.p2p_transfer_speed),
            # lookup hit-rate distribution
            lookup_0hit_reqs=s.interval_lookup_0_hit_requests,
            avg_lookup_hit_rate=_avg(s.interval_lookup_hit_rates),
            # cache lifespan
            avg_lifespan_min=_avg(_lifespan_values(s.interval_request_cache_lifespan)),
        )

        self._rows.append(snap)
        return snap

    # ── accessors ─────────────────────────────────────────────────────────

    @property
    def all(self) -> List[Snapshot]:
        return self._rows

    def for_scenario(self, scenario: str) -> List[Snapshot]:
        return [r for r in self._rows if r.scenario == scenario]

    def for_phase(self, phase: str) -> List[Snapshot]:
        return [r for r in self._rows if r.phase == phase]

    def find(self, scenario: str, phase: str, num_requests: int) -> Optional[Snapshot]:
        for r in self._rows:
            if r.scenario == scenario and r.phase == phase and r.num_requests == num_requests:
                return r
        return None
