"""
ICMS Report Generator

Formats benchmark results as console tables and saves them as JSON and CSV files.
Covers all three scenarios: cpu, ssd, cpu_ssd.
"""

import csv
import json
import os
import time
from typing import Dict, List, Optional

from icms.metrics_collector import MetricsSnapshot


# ──────────────────────────────────────────────
# Formatters
# ──────────────────────────────────────────────

def _fmt_bytes(value: int) -> str:
    if value == 0:
        return "0 B"
    for unit in ["B", "KB", "MB", "GB"]:
        if value < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def _fmt_ms(secs: float) -> str:
    return f"{secs * 1000:.2f} ms" if secs else "—"


def _fmt_speed(toks_per_sec: float) -> str:
    return f"{toks_per_sec:.1f} tok/s" if toks_per_sec else "—"


# ──────────────────────────────────────────────
# Console Tables
# ──────────────────────────────────────────────

def print_phase_timing_table(scenario: str, snapshots: List[MetricsSnapshot]):
    """Wall-clock time per phase and request count for one scenario."""
    by_reqs: Dict[int, Dict[str, MetricsSnapshot]] = {}
    for s in snapshots:
        if s.scenario != scenario:
            continue
        by_reqs.setdefault(s.num_requests, {})[s.phase] = s

    if not by_reqs:
        return

    label = scenario.upper()
    print(f"\n{'=' * 72}")
    print(f"  PHASE TIMING — {label}")
    print(f"{'=' * 72}")
    print(f"{'Reqs':<6} | {'Cold (ms)':<13} | {'GPU Hit (ms)':<14} | {'Offload Hit (ms)'}")
    print("-" * 72)

    for nr in sorted(by_reqs.keys()):
        p = by_reqs[nr]
        c  = f"{p['cold'].wall_time_ms:<13.1f}"     if "cold"        in p else f"{'—':<13}"
        g  = f"{p['gpu_hit'].wall_time_ms:<14.1f}"  if "gpu_hit"     in p else f"{'—':<14}"
        oh = f"{p['offload_hit'].wall_time_ms:.1f}" if "offload_hit" in p else "—"
        print(f"{nr:<6} | {c} | {g} | {oh}")


def print_lmcache_metrics_table(scenario: str, snapshots: List[MetricsSnapshot]):
    """Detailed LMCache metrics for each phase in one scenario."""
    filtered = [s for s in snapshots if s.scenario == scenario]
    if not filtered:
        return

    print(f"\n{'=' * 88}")
    print(f"  LMCACHE METRICS — {scenario.upper()}")
    print(f"{'=' * 88}")

    for s in filtered:
        print(f"\n  ── Reqs={s.num_requests}  Phase={s.phase} ──")
        # Core
        print(f"    Retrieve Requests:         {s.retrieve_requests}")
        print(f"    Store Requests:            {s.store_requests}")
        print(f"    Lookup Requests:           {s.lookup_requests}")
        print(f"    Requested Tokens:          {s.requested_tokens}")
        print(f"    Hit Tokens:                {s.hit_tokens}")
        print(f"    Stored Tokens:             {s.stored_tokens}")
        print(f"    Lookup Tokens:             {s.lookup_tokens}")
        print(f"    Lookup Hits:               {s.lookup_hits}")
        print(f"    vLLM Hit Tokens:           {s.vllm_hit_tokens}")
        print(f"    Prompt Tokens:             {s.prompt_tokens}")
        # Hit rates
        print(f"    Retrieve Hit Rate:         {s.retrieve_hit_rate:.4f}")
        print(f"    Lookup Hit Rate:           {s.lookup_hit_rate:.4f}")
        # Slow retrieval
        print(f"    Slow Retrieve (by time):   {s.num_slow_retrieval_by_time}")
        print(f"    Slow Retrieve (by speed):  {s.num_slow_retrieval_by_speed}")
        # Timing
        print(f"    Avg Retrieve Time:         {_fmt_ms(s.avg_time_to_retrieve)}")
        print(f"    Avg Store Time:            {_fmt_ms(s.avg_time_to_store)}")
        print(f"    Avg Lookup Time:           {_fmt_ms(s.avg_time_to_lookup)}")
        print(f"    Avg Retrieve Speed:        {_fmt_speed(s.avg_retrieve_speed)}")
        print(f"    Avg Store Speed:           {_fmt_speed(s.avg_store_speed)}")
        # Granular
        print(f"    Avg Retrieve→GPU:          {_fmt_ms(s.avg_retrieve_to_gpu_time)}")
        print(f"    Avg Retrieve Broadcast:    {_fmt_ms(s.avg_retrieve_broadcast_time)}")
        print(f"    Avg Retrieve Process Tok:  {_fmt_ms(s.avg_retrieve_process_tokens_time)}")
        print(f"    Avg Store GPU→Mem:         {_fmt_ms(s.avg_store_from_gpu_time)}")
        print(f"    Avg Store Process Tok:     {_fmt_ms(s.avg_store_process_tokens_time)}")
        print(f"    Avg Store Put:             {_fmt_ms(s.avg_store_put_time)}")
        # Cache usage
        print(f"    Local CPU Cache Usage:     {_fmt_bytes(s.local_cache_usage_bytes)}")
        print(f"    Local Disk Cache Usage:    {_fmt_bytes(s.local_storage_usage_bytes)}")
        print(f"    Remote Cache Usage:        {_fmt_bytes(s.remote_cache_usage_bytes)}")
        # Memory objects
        print(f"    Active Mem Objects:        {s.active_memory_objs_count}")
        print(f"    Pinned Mem Objects:        {s.pinned_memory_objs_count}")
        # Eviction
        print(f"    CPU Evict Count:           {s.cpu_evict_count}")
        print(f"    CPU Evict Keys:            {s.cpu_evict_keys_count}")
        print(f"    CPU Evict Failed:          {s.cpu_evict_failed_count}")
        print(f"    Forced Unpin Count:        {s.forced_unpin_count}")
        # P2P (usually 0 in local-only scenarios)
        if s.p2p_requests:
            print(f"    P2P Requests:              {s.p2p_requests}")
            print(f"    P2P Transferred Tokens:    {s.p2p_transferred_tokens}")
            print(f"    Avg P2P Transfer Speed:    {_fmt_speed(s.avg_p2p_transfer_speed)}")
        # Lookup hit rate distribution
        print(f"    Lookup 0-hit Requests:     {s.lookup_0_hit_requests}")
        if s.avg_lookup_hit_rate_non_zero:
            print(f"    Avg Lookup Hit Rate (>0):  {s.avg_lookup_hit_rate_non_zero:.4f}")


def print_cross_scenario_comparison(snapshots: List[MetricsSnapshot], scenarios: List[str]):
    """Side-by-side offload_hit comparison across all active scenarios."""
    offload = {sc: {s.num_requests: s for s in snapshots
                    if s.scenario == sc and s.phase == "offload_hit"}
               for sc in scenarios}

    common_reqs = sorted(
        set.intersection(*[set(d.keys()) for d in offload.values()])
        if len(offload) > 1 else set(next(iter(offload.values())).keys())
    )
    if not common_reqs:
        return

    print(f"\n{'=' * 100}")
    print(f"  OFFLOAD HIT COMPARISON — {' vs '.join(s.upper() for s in scenarios)}")
    print(f"{'=' * 100}")

    col = 14
    header = f"{'Reqs':<6}"
    for sc in scenarios:
        header += f" | {(sc.upper() + ' Time(ms)'):<{col}} | {(sc.upper() + ' Speed'):<{col}} | {(sc.upper() + ' Hit Rate'):<12}"
    print(header)
    print("-" * 100)

    for nr in common_reqs:
        row = f"{nr:<6}"
        for sc in scenarios:
            s = offload[sc].get(nr)
            if s:
                row += (f" | {s.wall_time_ms:<{col}.1f}"
                        f" | {s.avg_retrieve_speed:<{col}.1f}"
                        f" | {s.retrieve_hit_rate:<12.4f}")
            else:
                row += f" | {'—':<{col}} | {'—':<{col}} | {'—':<12}"
        print(row)


# ──────────────────────────────────────────────
# Full Report
# ──────────────────────────────────────────────

def print_full_report(snapshots: List[MetricsSnapshot], scenarios: List[str]):
    for scenario in scenarios:
        print_phase_timing_table(scenario, snapshots)
        print_lmcache_metrics_table(scenario, snapshots)

    if len(scenarios) > 1:
        print_cross_scenario_comparison(snapshots, scenarios)


# ──────────────────────────────────────────────
# File Output
# ──────────────────────────────────────────────

def save_results_json(
    snapshots: List[MetricsSnapshot],
    config_dict: dict,
    output_dir: str,
) -> str:
    """Save all snapshots + config to a timestamped JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"icms_results_{ts}.json")

    data = {
        "timestamp": ts,
        "config": config_dict,
        "snapshots": [s.to_dict() for s in snapshots],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    print(f"\n[Output] JSON saved → {path}")
    return path


def save_results_csv(
    snapshots: List[MetricsSnapshot],
    output_dir: str,
) -> str:
    """Save all snapshots to a flat CSV file for easy analysis."""
    os.makedirs(output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"icms_results_{ts}.csv")

    if not snapshots:
        return path

    fieldnames = list(snapshots[0].to_dict().keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in snapshots:
            writer.writerow(s.to_dict())

    print(f"[Output] CSV saved  → {path}")
    return path
