#!/usr/bin/env bash
# run_bench.sh — wrapper for bench.py (server-client mode)
#
# Override any default by setting the corresponding env var before running,
# or pass --flag value directly after the script name.
#
# Examples:
#   # Full run, all scenarios
#   DATA=/data/ShareGPT_V3.json bash run_bench.sh
#
#   # Single scenario, custom params
#   DATA=/data/ShareGPT.json SCENARIO=gpu_cpu NUM_REQUESTS=16 bash run_bench.sh
#
#   # Connect to external server (skip lifecycle management)
#   SERVER_URL=http://localhost:8000 bash run_bench.sh --scenario gpu_cpu
#
#   # WSL2 / local dev (small model, low VRAM)
#   DATA=./ShareGPT.json MODEL=facebook/opt-125m bash run_bench.sh --local-dev

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Required ──────────────────────────────────────────────────────────────────
DATA="${DATA:-}"

if [[ -z "$DATA" ]]; then
    echo "ERROR: DATA path is required. Set DATA=/path/to/ShareGPT.json" >&2
    exit 1
fi
if [[ ! -f "$DATA" ]]; then
    echo "ERROR: DATA file not found: $DATA" >&2
    exit 1
fi

# ── Scenario ──────────────────────────────────────────────────────────────────
SCENARIO="${SCENARIO:-all}"               # gpu_only | gpu_cpu | gpu_ssd | gpu_cpu_ssd | all

# ── vLLM server ───────────────────────────────────────────────────────────────
MODEL="${MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
PORT="${PORT:-8000}"
GPU_MEM="${GPU_MEM:-0.5}"                 # gpu_memory_utilization
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
NUM_GPU_BLOCKS="${NUM_GPU_BLOCKS:-}"      # leave empty to let vLLM auto-size
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-300}"
SERVER_URL="${SERVER_URL:-}"              # set to connect to existing server

# ── Benchmark ─────────────────────────────────────────────────────────────────
NUM_REQUESTS="${NUM_REQUESTS:-8}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-64}"
HIT_RATES="${HIT_RATES:-0.0 0.25 0.50 0.75 1.0}"
MAX_CHARS="${MAX_CHARS:-4000}"
DATA_SEED="${DATA_SEED:-42}"
OFFLOAD_WAIT="${OFFLOAD_WAIT:-5.0}"
STATS_FLUSH_WAIT="${STATS_FLUSH_WAIT:-12.0}"

# ── LMCache ───────────────────────────────────────────────────────────────────
CHUNK_SIZE="${CHUNK_SIZE:-256}"
MAX_CPU_GB="${MAX_CPU_GB:-10.0}"
MAX_DISK_GB="${MAX_DISK_GB:-10.0}"
DISK_PATH="${DISK_PATH:-/tmp/kvcache_bench}"

# ── Prometheus ────────────────────────────────────────────────────────────────
PROMETHEUS_URL="${PROMETHEUS_URL:-http://localhost:9090/metrics}"

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_DIR="${OUTPUT_DIR:-./bench_results}"

# ── Build args array ──────────────────────────────────────────────────────────
ARGS=(
    --data              "$DATA"
    --scenario          "$SCENARIO"
    --model             "$MODEL"
    --port              "$PORT"
    --gpu-mem           "$GPU_MEM"
    --max-model-len     "$MAX_MODEL_LEN"
    --startup-timeout   "$STARTUP_TIMEOUT"
    --num-requests      "$NUM_REQUESTS"
    --max-output-tokens "$MAX_OUTPUT_TOKENS"
    --hit-rates         $HIT_RATES
    --max-chars         "$MAX_CHARS"
    --data-seed         "$DATA_SEED"
    --offload-wait      "$OFFLOAD_WAIT"
    --stats-flush-wait  "$STATS_FLUSH_WAIT"
    --chunk-size        "$CHUNK_SIZE"
    --max-cpu-gb        "$MAX_CPU_GB"
    --max-disk-gb       "$MAX_DISK_GB"
    --disk-path         "$DISK_PATH"
    --prometheus-url    "$PROMETHEUS_URL"
    --output-dir        "$OUTPUT_DIR"
)

[[ -n "$NUM_GPU_BLOCKS" ]] && ARGS+=(--num-gpu-blocks "$NUM_GPU_BLOCKS")
[[ -n "$SERVER_URL"     ]] && ARGS+=(--server-url     "$SERVER_URL")

# Pass any extra CLI args directly (e.g. --local-dev)
ARGS+=("$@")

echo "Running: python ${SCRIPT_DIR}/bench.py ${ARGS[*]}"
exec python "${SCRIPT_DIR}/bench.py" "${ARGS[@]}"
