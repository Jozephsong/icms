"""
ICMS Report Generator

Formats and outputs benchmark results as console tables and JSON files.
"""

import json
import os
import time
from typing import Dict, List, Optional

from icms.metrics_collector import MetricsSnapshot


def _format_bytes(value: int) -> str:
    """Format bytes into human-readable string."""
    if value == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB"]
    idx = 0
    fval = float(value)
    while fval >= 1024 and idx < len(units) - 1:
        fval /= 1024
        idx += 1
    return f"{fval:.2f} {units[idx]}"


def _format_time_ms(value: float) -> str:
    """Format seconds to ms string."""
    if value == 0:
        return "—"
    return f"{value * 1000:.2f} ms"


def _format_speed(value: float) -> str:
    """Format tokens/sec."""
    if value == 0:
        return "—"
    return f"{value:.1f} tok/s"


def print_phase_summary_table(
    scenario: str,
    snapshots: List[MetricsSnapshot],
):
    """Print a summary table of wall-clock times for each phase and num_requests."""
    # Group snapshots by num_requests
    by_reqs: Dict[int, Dict[str, MetricsSnapshot]] = {}
    for s in snapshots:
        if s.scenario != scenario:
            continue
        by_reqs.setdefault(s.num_requests, {})[s.phase] = s

    if not by_reqs:
        print(f"  No data for scenario '{scenario}'")
        return

    print(f"\n{'=' * 70}")
    print(f"  PHASE TIMING SUMMARY — Scenario: {scenario.upper()}")
    print(f"{'=' * 70}")
    header = f"{'Reqs':<6} | {'Cold (ms)':<12} | {'GPU Hit (ms)':<14} | {'Offload Hit (ms)':<18}"
    print(header)
    print("-" * 70)

    for num_reqs in sorted(by_reqs.keys()):
        phases = by_reqs[num_reqs]
        cold = phases.get("cold")
        gpu = phases.get("gpu_hit")
        offload = phases.get("offload_hit")

        cold_str = f"{cold.wall_time_ms:<12.2f}" if cold else f"{'—':<12}"
        gpu_str = f"{gpu.wall_time_ms:<14.2f}" if gpu else f"{'—':<14}"
        offload_str = f"{offload.wall_time_ms:<18.2f}" if offload else f"{'—':<18}"

        print(f"{num_reqs:<6} | {cold_str} | {gpu_str} | {offload_str}")


def print_detailed_metrics_table(
    scenario: str,
    snapshots: List[MetricsSnapshot],
):
    """Print detailed LMCache metrics for each phase."""
    filtered = [s for s in snapshots if s.scenario == scenario]
    if not filtered:
        return

    print(f"\n{'=' * 90}")
    print(f"  DETAILED LMCACHE METRICS — Scenario: {scenario.upper()}")
    print(f"{'=' * 90}")

    for s in filtered:
        print(f"\n  ── Reqs={s.num_requests}, Phase={s.phase} ──")
        print(f"    Wall Time:            {s.wall_time_ms:.2f} ms")
        print(f"    Retrieve Requests:    {s.retrieve_requests}")
        print(f"    Store Requests:       {s.store_requests}")
        print(f"    Requested Tokens:     {s.requested_tokens}")
        print(f"    Hit Tokens:           {s.hit_tokens}")
        print(f"    Stored Tokens:        {s.stored_tokens}")
        print(f"    Retrieve Hit Rate:    {s.retrieve_hit_rate:.4f}")
        print(f"    Avg Retrieve Time:    {_format_time_ms(s.avg_time_to_retrieve)}")
        print(f"    Avg Store Time:       {_format_time_ms(s.avg_time_to_store)}")
        print(f"    Avg Retrieve Speed:   {_format_speed(s.avg_retrieve_speed)}")
        print(f"    Avg Store Speed:      {_format_speed(s.avg_store_speed)}")
        print(f"    Avg Retrieve→GPU:     {_format_time_ms(s.avg_retrieve_to_gpu_time)}")
        print(f"    Avg Store GPU→Mem:    {_format_time_ms(s.avg_store_from_gpu_time)}")
        print(f"    Avg Store Put:        {_format_time_ms(s.avg_store_put_time)}")
        print(f"    Local Cache Usage:    {_format_bytes(s.local_cache_usage_bytes)}")
        print(f"    Local Storage Usage:  {_format_bytes(s.local_storage_usage_bytes)}")
        print(f"    CPU Evict Count:      {s.cpu_evict_count}")
        print(f"    Active Mem Objects:   {s.active_memory_objs_count}")
        print(f"    Pinned Mem Objects:   {s.pinned_memory_objs_count}")


def print_comparison_table(snapshots: List[MetricsSnapshot]):
    """Print side-by-side comparison of CPU vs SSD for offload_hit phase."""
    cpu_offload = [
        s for s in snapshots if s.scenario == "cpu" and s.phase == "offload_hit"
    ]
    ssd_offload = [
        s for s in snapshots if s.scenario == "ssd" and s.phase == "offload_hit"
    ]

    if not cpu_offload or not ssd_offload:
        return

    cpu_by_reqs = {s.num_requests: s for s in cpu_offload}
    ssd_by_reqs = {s.num_requests: s for s in ssd_offload}
    all_reqs = sorted(set(cpu_by_reqs.keys()) & set(ssd_by_reqs.keys()))

    if not all_reqs:
        return

    print(f"\n{'=' * 90}")
    print(f"  OFFLOAD HIT COMPARISON — CPU vs SSD")
    print(f"{'=' * 90}")
    header = (
        f"{'Reqs':<6} | "
        f"{'CPU Time(ms)':<14} | {'SSD Time(ms)':<14} | "
        f"{'CPU Spd(tok/s)':<16} | {'SSD Spd(tok/s)':<16} | "
        f"{'CPU Hit Rate':<13} | {'SSD Hit Rate':<13}"
    )
    print(header)
    print("-" * 90)

    for nr in all_reqs:
        c = cpu_by_reqs[nr]
        s = ssd_by_reqs[nr]
        print(
            f"{nr:<6} | "
            f"{c.wall_time_ms:<14.2f} | {s.wall_time_ms:<14.2f} | "
            f"{c.avg_retrieve_speed:<16.1f} | {s.avg_retrieve_speed:<16.1f} | "
            f"{c.retrieve_hit_rate:<13.4f} | {s.retrieve_hit_rate:<13.4f}"
        )


def save_results_json(
    snapshots: List[MetricsSnapshot],
    config_dict: dict,
    output_dir: str,
):
    """Save all snapshots and config to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"benchmark_results_{timestamp}.json")

    data = {
        "timestamp": timestamp,
        "config": config_dict,
        "snapshots": [s.to_dict() for s in snapshots],
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)

    print(f"\n✅ Results saved to: {filepath}")
    return filepath


def print_full_report(
    snapshots: List[MetricsSnapshot],
    scenarios: List[str],
):
    """Print the full benchmark report for all scenarios."""
    for scenario in scenarios:
        print_phase_summary_table(scenario, snapshots)
        print_detailed_metrics_table(scenario, snapshots)

    if "cpu" in scenarios and "ssd" in scenarios:
        print_comparison_table(snapshots)
