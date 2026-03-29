"""
Result output: console tables, JSON, and CSV.

save() writes two files per run:
  icms_results_<timestamp>.json  — full structured dump (config + all snapshots)
  icms_results_<timestamp>.csv   — flat table, one row per snapshot

Console output mirrors the three phases and produces a cross-scenario
comparison table when more than one scenario was tested.
"""

import csv
import json
import os
import time
from typing import Dict, List

from icms.metrics import Collector, Snapshot


# ── formatting helpers ────────────────────────────────────────────────────────

def _ms(secs: float) -> str:
    return f"{secs * 1e3:.1f}ms" if secs else "—"

def _tps(v: float) -> str:
    return f"{v:.0f} tok/s" if v else "—"

def _mb(b: int) -> str:
    return f"{b / 2**20:.1f} MB" if b else "0 MB"


# ── console tables ────────────────────────────────────────────────────────────

def print_phase_table(scenario: str, rows: List[Snapshot]) -> None:
    """Wall-clock timing for the three phases, grouped by num_requests."""
    by_n: Dict[int, Dict[str, Snapshot]] = {}
    for r in rows:
        if r.scenario != scenario:
            continue
        by_n.setdefault(r.num_requests, {})[r.phase] = r

    if not by_n:
        return

    W = 70
    print(f"\n{'═' * W}")
    print(f"  PHASE TIMING  ·  {scenario.upper()}")
    print(f"{'═' * W}")
    print(f"  {'Reqs':<6}  {'Cold':>10}  {'GPU Hit':>10}  {'Offload Hit':>12}  {'Speedup*':>9}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*9}")
    for n in sorted(by_n):
        p = by_n[n]
        cold    = p.get("cold")
        gpu     = p.get("gpu_hit")
        offload = p.get("offload_hit")
        c_ms  = f"{cold.wall_ms:>9.0f}ms"    if cold    else f"{'—':>10}"
        g_ms  = f"{gpu.wall_ms:>9.0f}ms"     if gpu     else f"{'—':>10}"
        o_ms  = f"{offload.wall_ms:>11.0f}ms" if offload else f"{'—':>12}"
        # speedup = cold / offload (how much faster offload_hit is vs cold)
        speedup = "—"
        if cold and offload and offload.wall_ms > 0:
            speedup = f"{cold.wall_ms / offload.wall_ms:>7.2f}×"
        print(f"  {n:<6}  {c_ms}  {g_ms}  {o_ms}  {speedup:>9}")
    print(f"  * cold_time / offload_hit_time")


def print_metrics_detail(scenario: str, rows: List[Snapshot]) -> None:
    """Full metric dump for every (scenario, phase, num_requests) triple."""
    filtered = [r for r in rows if r.scenario == scenario]
    if not filtered:
        return

    W = 86
    print(f"\n{'═' * W}")
    print(f"  LMCACHE METRICS  ·  {scenario.upper()}")
    print(f"{'═' * W}")

    for r in filtered:
        print(f"\n  ┌── {scenario}  requests={r.num_requests}  phase={r.phase} ──")
        # request & token counts
        print(f"  │  retrieve/store/lookup req  {r.n_retrieve} / {r.n_store} / {r.n_lookup}")
        print(f"  │  tokens: requested={r.tok_requested}  hit={r.tok_hit}  "
              f"stored={r.tok_stored}  prompt={r.tok_prompt}")
        print(f"  │  lookup: tokens={r.tok_lookup}  hits={r.tok_lookup_hit}  "
              f"vllm_hit={r.tok_vllm_hit}")
        # hit rates
        print(f"  │  retrieve_hit_rate={r.retrieve_hit_rate:.4f}  "
              f"lookup_hit_rate={r.lookup_hit_rate:.4f}")
        # latency / throughput
        print(f"  │  retrieve  {_ms(r.avg_retrieve_s)}  @  {_tps(r.avg_retrieve_tps)}")
        print(f"  │  store     {_ms(r.avg_store_s)}  @  {_tps(r.avg_store_tps)}")
        print(f"  │  lookup    {_ms(r.avg_lookup_s)}")
        # pipeline breakdown
        print(f"  │  pipeline (retrieve): process_tok={_ms(r.avg_retr_process_tok_s)}  "
              f"broadcast={_ms(r.avg_retr_broadcast_s)}  to_gpu={_ms(r.avg_retr_to_gpu_s)}")
        print(f"  │  pipeline (store):    process_tok={_ms(r.avg_store_process_tok_s)}  "
              f"from_gpu={_ms(r.avg_store_from_gpu_s)}  put={_ms(r.avg_store_put_s)}")
        # slow retrieval
        if r.slow_by_time or r.slow_by_speed:
            print(f"  │  slow retrieval: by_time={r.slow_by_time}  by_speed={r.slow_by_speed}")
        # cache occupancy
        print(f"  │  cache: cpu={_mb(r.cpu_bytes)}  disk={_mb(r.disk_bytes)}  "
              f"remote={_mb(r.remote_bytes)}")
        # memory objects
        print(f"  │  mem objects: active={r.mem_active}  pinned={r.mem_pinned}")
        # eviction
        print(f"  │  eviction: count={r.evict_count}  keys={r.evict_keys}  "
              f"failed={r.evict_failed}  forced_unpin={r.forced_unpin}")
        # P2P (typically zero in local-only scenarios)
        if r.p2p_reqs:
            print(f"  │  P2P: reqs={r.p2p_reqs}  tokens={r.p2p_tok}  "
                  f"time={_ms(r.avg_p2p_s)}  speed={_tps(r.avg_p2p_tps)}")
        # lookup hit-rate distribution
        print(f"  │  lookup 0-hit requests={r.lookup_0hit_reqs}  "
              f"avg_hit_rate(>0)={r.avg_lookup_hit_rate:.4f}")
        # cache lifespan
        if r.avg_lifespan_min:
            print(f"  │  avg cache lifespan={r.avg_lifespan_min:.2f} min")
        print(f"  └──  wall={r.wall_ms:.0f}ms")


def print_comparison(scenarios: List[str], rows: List[Snapshot]) -> None:
    """Cross-scenario comparison for the offload_hit phase."""
    # collect offload_hit rows keyed by (scenario, num_requests)
    data: Dict[str, Dict[int, Snapshot]] = {}
    for sc in scenarios:
        data[sc] = {r.num_requests: r for r in rows
                    if r.scenario == sc and r.phase == "offload_hit"}

    all_n = sorted(set(n for d in data.values() for n in d))
    if not all_n:
        return

    W = 100
    print(f"\n{'═' * W}")
    print(f"  OFFLOAD HIT COMPARISON  ·  {' vs '.join(s.upper() for s in scenarios)}")
    print(f"{'═' * W}")

    col = 13
    hdr = f"  {'Reqs':<6}"
    for sc in scenarios:
        hdr += f"  {(sc+' wall'):>{col}}  {(sc+' speed'):>{col}}  {(sc+' hit%'):>8}"
    print(hdr)
    print(f"  {'─'*6}" + (f"  {'─'*col}  {'─'*col}  {'─'*8}") * len(scenarios))

    for n in all_n:
        row = f"  {n:<6}"
        for sc in scenarios:
            s = data[sc].get(n)
            if s:
                row += (f"  {s.wall_ms:>{col}.0f}ms"
                        f"  {s.avg_retrieve_tps:>{col}.0f} t/s"
                        f"  {s.retrieve_hit_rate:>7.1%}")
            else:
                row += f"  {'—':>{col}}  {'—':>{col}}  {'—':>8}"
        print(row)


def print_report(scenarios: List[str], collector: Collector) -> None:
    rows = collector.all
    for sc in scenarios:
        print_phase_table(sc, rows)
        print_metrics_detail(sc, rows)
    if len(scenarios) > 1:
        print_comparison(scenarios, rows)


# ── file output ───────────────────────────────────────────────────────────────

def save(collector: Collector, config: dict, out_dir: str) -> None:
    """
    Write JSON + CSV to out_dir.
    Both files are stamped with the current time so repeated runs don't
    overwrite each other.
    """
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    rows = collector.all

    # ── JSON ──────────────────────────────────────────────────────────────
    json_path = os.path.join(out_dir, f"icms_results_{stamp}.json")
    with open(json_path, "w") as fh:
        json.dump(
            {"timestamp": stamp, "config": config, "results": [r.as_dict() for r in rows]},
            fh,
            indent=2,
            default=str,
        )
    print(f"\n  [saved] JSON → {json_path}")

    # ── CSV ───────────────────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, f"icms_results_{stamp}.csv")
    if rows:
        fields = list(rows[0].as_dict().keys())
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(r.as_dict() for r in rows)
        print(f"  [saved] CSV  → {csv_path}")
