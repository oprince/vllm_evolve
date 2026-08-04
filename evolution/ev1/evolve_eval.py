"""
Evaluation harness for OpenEvolve.

Writes the evolved handler.py to disk, starts a vLLM server, runs the
benchmark, and returns a composite score.

OpenEvolve contract:
    evaluate(program_path: str) -> dict
    Must return a dict with at least {"combined_score": float}.
    Extra keys are stored as metadata.
"""

import ast
import json
import logging
import os
import pathlib
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────

# Root of the vLLM repository (two levels up from evolution/ev1/)
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

load_dotenv(PROJECT_ROOT / ".env")


BENCH_BACKEND = os.environ.get("EVOLVE_BENCH_BACKEND", "vllm_bench")

MODEL = os.environ.get(
    "EVOLVE_MODEL", "NousResearch/Hermes-3-Llama-3.1-8B"
)
NUM_PROMPTS = int(os.environ.get("EVOLVE_NUM_PROMPTS", "30"))
SERVER_PORT = int(os.environ.get("EVOLVE_SERVER_PORT", "8000"))
KV_OFFLOAD_SIZE = os.environ.get("EVOLVE_KV_OFFLOAD_SIZE", "16")
KV_OFFLOADING_BACKEND = os.environ.get("EVOLVE_KV_OFFLOAD_BACKEND", "cpu")
EVOLVE_FILENAME = os.environ.get("EVOLVE_FILENAME", "evolved.py")
SERVER_URL = f"http://localhost:{SERVER_PORT}"
HEALTH_URL = f"{SERVER_URL}/health"

# Evolve the in-tree vLLM CPU offloading policy directly
EVOLVE_SRC = PROJECT_ROOT / "vllm" / "v1" / "kv_offload" / "cpu" / "policies" / EVOLVE_FILENAME
EVOLVE_BACKUP = EVOLVE_SRC.with_suffix(".py.bak")

# ── Baseline metrics (fill in from Phase 1) ──────────────────────────────────

BASELINE_THROUGHPUT = float(os.environ.get("EVOLVE_BASELINE_THROUGHPUT", "1.0"))
# Mean TTFT (ms) of the unmodified LRU manager on the same benchmark workload.
# Candidates score higher when their TTFT beats this number (ratio > 1).
BASELINE_TTFT_MS = float(os.environ.get("EVOLVE_BASELINE_TTFT_MS", "81.0"))

# ── Seed validation state ────────────────────────────────────────────────────

_SEED_VALIDATED = False

# ── Helpers ──────────────────────────────────────────────────────────────────

SERVER_LOG_DIR = PROJECT_ROOT / "vllm_server_logs"


def _wait_for_gpu_free(timeout: int = 120) -> None:
    """Poll nvidia-smi until no compute processes are using the GPU."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            logger.warning("nvidia-smi failed (rc=%d), skipping GPU check",
                           result.returncode)
            time.sleep(5)
            return
        if not result.stdout.strip():
            logger.info("GPU is free")
            return
        logger.info("GPU still in use, waiting... (%s)",
                    result.stdout.strip().replace("\n", ", "))
        time.sleep(3)
    logger.warning("GPU not free after %ds — proceeding anyway", timeout)


def _kill_existing_server(port: int, timeout: int = 30) -> None:
    """Kill any process listening on the given port and wait for GPU release."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}"],
            capture_output=True, text=True,
        )
        pids = result.stdout.strip().split()
        if not pids:
            return

        logger.info("Found existing process(es) on port %d: %s", port, pids)
        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGTERM)
            except (ProcessLookupError, ValueError):
                pass

        deadline = time.time() + timeout
        while time.time() < deadline:
            check = subprocess.run(
                ["lsof", "-ti", f"tcp:{port}"],
                capture_output=True, text=True,
            )
            if not check.stdout.strip():
                break
            time.sleep(1)
        else:
            # Force kill anything still alive
            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except (ProcessLookupError, ValueError):
                    pass
            time.sleep(2)

        # Wait until GPU memory is actually freed
        _wait_for_gpu_free(timeout=timeout)
    except FileNotFoundError:
        # lsof not available — fall back to killing by process name
        subprocess.run(
            ["pkill", "-f", f"serve.py.*--port.*{port}"],
            capture_output=True,
        )
        _wait_for_gpu_free(timeout=timeout)


def _drain_pipe(pipe, captured_lines=None, all_lines=None):
    """Read a pipe in a background thread to prevent deadlock.

    If *captured_lines* (a list) is provided, matching log lines are appended.
    If *all_lines* (a list) is provided, every line is appended (for log dump).
    """
    def _reader():
        try:
            for line in pipe:
                stripped = line.rstrip()
                if all_lines is not None:
                    all_lines.append(stripped)
                if captured_lines is not None and (
                    "loaded OffloadingManager" in line
                    or "External prefix cache hit rate" in line
                ):
                    captured_lines.append(stripped)
        except (ValueError, OSError):
            pass
    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    return t


def _dump_server_log(all_lines):
    """Write captured server output to a timestamped file for debugging."""
    try:
        SERVER_LOG_DIR.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = SERVER_LOG_DIR / f"vllm_server_{timestamp}.log"
        with open(log_path, "w") as f:
            f.write("\n".join(all_lines))
        logger.info("Server log written to %s (%d lines)", log_path, len(all_lines))
    except Exception as e:
        logger.warning("Failed to write server log: %s", e)


def _wait_for_ready(timeout: int = 120) -> bool:
    """Poll the health endpoint until the server is ready."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(HEALTH_URL, timeout=2)
            return True
        except Exception:
            time.sleep(2)
    return False


def _correctness_check() -> bool:
    """Send one prompt and verify we get a non-empty response."""
    payload = json.dumps({
        "model": MODEL,
        "prompt": "The capital of France is",
        "max_tokens": 10,
    }).encode()
    req = urllib.request.Request(
        f"{SERVER_URL}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        body = json.loads(resp.read())
        text = body["choices"][0]["text"].strip()
        return len(text) > 0
    except Exception as e:
        logger.warning("correctness check failed: %s", e)
        return False


def _parse_bench_output(stdout: str) -> dict:
    """Extract metrics from vllm bench serve output.

    Tries JSON first (--save-result format), then falls back to regex on
    the human-readable output.
    """
    metrics = {}

    # Try to find a JSON blob in the output
    try:
        # vllm bench may print a JSON object with results
        json_match = re.search(r"\{[^{}]*\"request_throughput\"[^{}]*\}", stdout)
        if json_match:
            metrics = json.loads(json_match.group())
            return metrics
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: regex extraction from human-readable table output
    # vllm bench prints lines like "Request throughput (req/s):       0.25"
    patterns = {
        "request_throughput": r"Request throughput[^:]*:\s*([\d.]+)",
        "output_token_throughput": r"Output token throughput[^:]*:\s*([\d.]+)",
        "total_token_throughput": r"Total [Tt]oken throughput[^:]*:\s*([\d.]+)",
        "mean_ttft_ms": r"Mean TTFT[^:]*:\s*([\d.]+)",
        "p99_ttft_ms": r"P99 TTFT[^:]*:\s*([\d.]+)",
        "mean_tpot_ms": r"Mean TPOT[^:]*:\s*([\d.]+)",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, stdout, re.IGNORECASE)
        if m:
            metrics[key] = float(m.group(1))

    return metrics


def _kill_server(proc: subprocess.Popen):
    """Terminate the server process, escalating to SIGKILL if needed."""
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def _extract_python_from_markdown(code: str) -> str:
    """Extract Python code from markdown code fences if present."""
    lines = code.split('\n')

    # Check if first line is markdown fence
    if lines and lines[0].strip().startswith('```'):
        # Remove opening fence
        lines = lines[1:]
        # Remove closing fence if present
        if lines and lines[-1].strip().startswith('```'):
            lines = lines[:-1]
        return '\n'.join(lines).strip()

    # If no markdown fences, return as-is
    return code.strip()


def _validate_evolved_code(code: str) -> tuple[bool, str]:
    """
    Pre-flight validation of evolved code.

    Returns:
        (is_valid, error_message)
    """
    # Extract Python from markdown if wrapped
    code = _extract_python_from_markdown(code)

    # 1. Check syntax
    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, f"syntax error at line {e.lineno}: {e.msg}"

    # 2. Try to compile
    try:
        compile(code, "<evolved>", "exec")
    except Exception as e:
        return False, f"compilation error: {str(e)}"

    # 3. Try to execute in a sandbox and find the expected class
    try:
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        namespace = {}
        exec(code, namespace)

        # Look for expected classes (manager.py or handler.py or policy)
        expected_names = [
            "LRUOffloadingManager", "CpuGpuOffloadingHandlers",
            "ARCOffloadingManager", "handler", "Manager",
            "EVCachePolicy",
        ]
        # Also accept any class ending in "Manager" or "Policy"
        found = any(name in namespace for name in expected_names) or any(
            (name.endswith("Manager") or name.endswith("Policy"))
            and not name.startswith("__")
            for name in namespace
        )
        if not found:
            # List what was defined
            defined = [k for k in namespace.keys() if not k.startswith("__")]
            return False, f"no expected class found. Defined: {defined[:5]}"

        return True, ""
    except Exception as e:
        return False, f"runtime error during execution: {str(e)}"


# ── Benchmark backends ──────────────────────────────────────────────────────

HERE = pathlib.Path(__file__).resolve().parent


def _run_benchmark_vllm_bench() -> dict:
    """Run vllm bench serve and return parsed metrics."""
    dataset_name = os.environ.get("DATASET_NAME", "sharegpt")
    dataset_path = os.environ.get("DATASET_PATH")
    if not dataset_path:
        return {"error": "DATASET_PATH env var not set"}

    logger.info("Starting vllm bench (%d prompts)...", NUM_PROMPTS)
    bench_cmd = [
        "vllm", "bench", "serve",
        "--backend", "vllm",
        "--model", MODEL,
        "--endpoint", "/v1/completions",
        "--dataset-name", dataset_name,
        "--dataset-path", dataset_path,
        "--num-prompts", str(NUM_PROMPTS),
        "--port", str(SERVER_PORT),
        "--disable-shuffle",
        "--max-concurrency", os.environ.get("EVOLVE_MAX_CONCURRENCY", "4"),
    ]
    result = subprocess.run(
        bench_cmd,
        capture_output=True,
        text=True,
        timeout=600,
    )

    logger.info("Benchmark finished")
    metrics = _parse_bench_output(result.stdout + "\n" + result.stderr)
    if not metrics.get("request_throughput"):
        return {
            "error": "benchmark produced no metrics",
            "stdout": result.stdout[-500:],
            "stderr": result.stderr[-500:],
        }
    return metrics


def _run_benchmark_inference_perf() -> dict:
    """Run inference-perf with SWE-Smith replay plan and return parsed metrics."""
    config_file = os.environ.get(
        "INFERENCE_PERF_CONFIG",
        str(HERE / "config_inference_perf_swe.yml"),
    )

    logger.info("Starting inference-perf benchmark...")
    result = subprocess.run(
        ["inference-perf", "--config_file", config_file],
        capture_output=True,
        text=True,
        timeout=900,
    )

    if result.returncode != 0:
        return {
            "error": f"inference-perf failed (rc={result.returncode})",
            "stdout": result.stdout[-500:],
            "stderr": result.stderr[-500:],
        }

    # Config sets storage.local_storage.path = "reports" (relative to cwd)
    reports_dir = PROJECT_ROOT / 'inference_perf' / "reports"
    summary_path = reports_dir / "summary_lifecycle_metrics.json"
    if not summary_path.exists():
        return {"error": f"summary_lifecycle_metrics.json not found in {reports_dir}"}

    logger.info("Parsing inference-perf results from %s", summary_path)
    summary = json.loads(summary_path.read_text())

    successes = summary.get("successes", {})
    throughput = successes.get("throughput", {})
    latency = successes.get("latency", {})

    metrics = {
        "request_throughput": throughput.get("requests_per_sec", 0.0),
        "output_token_throughput": throughput.get("output_tokens_per_sec", 0.0),
        "mean_ttft_ms": latency.get("time_to_first_token", {}).get("mean", 0.0),
        "p99_ttft_ms": latency.get("time_to_first_token", {}).get("p99", 0.0),
        "mean_request_latency_ms": latency.get("request_latency", {}).get("mean", 0.0),
        "total_requests": successes.get("count", 0),
        "failures": summary.get("failures", {}).get("count", 0),
    }

    if not metrics.get("request_throughput"):
        return {
            "error": "inference-perf produced no throughput metric",
            "raw_summary": str(summary)[:500],
        }

    return metrics


# ── Main evaluator ───────────────────────────────────────────────────────────

def _save_evolved_program(code: str, original_path: str):
    """Save the evolved program for debugging."""
    iterations_dir = PROJECT_ROOT / "evolution_iterations"
    iterations_dir.mkdir(exist_ok=True)

    # Extract iteration number from OpenEvolve's path if possible
    # (typically something like /tmp/.../iter_N or similar)
    file_stem = pathlib.Path(original_path).stem

    # Use timestamp + stem as filename
    timestamp = int(time.time() * 1000)  # milliseconds for uniqueness
    output_file = iterations_dir / f"{timestamp}_{file_stem}.py"

    output_file.write_text(code)
    logger.info("Saved evolved program to %s", output_file)


def _is_seed_iteration() -> bool:
    """True if this is the first evaluation (seed program)."""
    return not _SEED_VALIDATED


def evaluate(program_path):
    """
    OpenEvolve evaluator entry point.

    Args:
        program_path: path to the evolved handler.py candidate.

    Returns:
        dict with "combined_score" (float) and optional "metrics" / "error".
    """
    global _SEED_VALIDATED

    logger.info("program_path: %s", program_path)
    raw_code = pathlib.Path(program_path).read_text()

    # ── Extract Python code if wrapped in markdown ───────────────────────────
    evolved_code = _extract_python_from_markdown(raw_code)

    # ── Save evolved program for debugging ────────────────────────────────────
    _save_evolved_program(evolved_code, program_path)

    # ── Validate evolved code before deployment ──────────────────────────────
    is_valid, error_msg = _validate_evolved_code(evolved_code)
    if not is_valid:
        return {"combined_score": 0.0, "error": f"invalid evolved code: {error_msg}"}

    # ── Kill any leftover server from a prior evaluation ───────────────────
    _kill_existing_server(SERVER_PORT)

    # Backup original handler on first run
    if not EVOLVE_BACKUP.exists():
        EVOLVE_BACKUP.write_text(EVOLVE_SRC.read_text())

    server = None
    try:
        # ── Deploy evolved handler ───────────────────────────────────────
        EVOLVE_SRC.write_text(evolved_code)

        # ── Start vLLM server ────────────────────────────────────────────
        # Use the vLLM venv Python to ensure all dependencies are available
        vllm_python = os.environ.get(
            "VLLM_PYTHON",
            str(PROJECT_ROOT / ".venv" / "bin" / "python"),
        )

        server_env = {
            **os.environ,
        }
        logger.info("Starting vLLM server...")
        # "--no-enable-prefix-caching",
        server = subprocess.Popen(
            [
                vllm_python, str(PROJECT_ROOT / "serve.py"),
                "--model", MODEL,
                "--kv-offloading-backend", KV_OFFLOADING_BACKEND,
                "--kv-offloading-eviction-policy", os.environ.get("EVOLVE_EVICTION_POLICY", "evolved"),
                "--kv-offloading-size", KV_OFFLOAD_SIZE,
                "--gpu-memory-utilization", os.environ.get("EVOLVE_GPU_MEM_UTIL", "0.9"),
                "--port", str(SERVER_PORT),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=server_env,
        )
        server_log_lines = []
        server_all_lines = []
        _drain_pipe(server.stdout, server_log_lines, server_all_lines)
        _drain_pipe(server.stderr, server_log_lines, server_all_lines)

        server_timeout = int(os.environ.get("EVOLVE_SERVER_TIMEOUT", "180"))
        if not _wait_for_ready(timeout=server_timeout):
            logger.error("Server failed to start after %ds", server_timeout)
            _dump_server_log(server_all_lines)
            if _is_seed_iteration():
                msg = (f"FATAL: Server failed to start for seed iteration "
                       f"after {server_timeout}s. Aborting evolution — fix "
                       f"server config before retrying.")
                logger.critical(msg)
                raise SystemExit(msg)
            return {"combined_score": 0.0, "error": "server failed to start"}
        logger.info("Server is ready")

        for line in server_log_lines:
            logger.info("%s", line)

        # ── Correctness gate ─────────────────────────────────────────────
        logger.info("Running correctness check...")
        if not _correctness_check():
            logger.error("Correctness check failed")
            return {"combined_score": 0.0, "error": "correctness check failed"}
        logger.info("Correctness check passed")

        # ── Run benchmark ────────────────────────────────────────────────
        logger.info("Benchmark backend: %s", BENCH_BACKEND)
        if BENCH_BACKEND == "inference_perf":
            metrics = _run_benchmark_inference_perf()
        else:
            metrics = _run_benchmark_vllm_bench()

        if "error" in metrics:
            return {"combined_score": 0.0, **metrics}

        # ── Cross-check: parse vLLM engine metric lines ──────────────────
        # Engine logs lines like:
        #   Engine 000: ... Prefix cache hit rate: 83.6%,
        #   External prefix cache hit rate: 0.6%
        # The last such line captured reflects end-of-benchmark state.
        engine_gpu_prefix_hit_rate = None
        engine_external_prefix_hit_rate = None
        for line in reversed(server_log_lines):
            if "External prefix cache hit rate" not in line:
                continue
            m_ext = re.search(
                r"External prefix cache hit rate:\s*([\d.]+)%", line
            )
            m_gpu = re.search(
                r"(?<!External )Prefix cache hit rate:\s*([\d.]+)%", line
            )
            if m_ext:
                engine_external_prefix_hit_rate = float(m_ext.group(1)) / 100.0
            if m_gpu:
                engine_gpu_prefix_hit_rate = float(m_gpu.group(1)) / 100.0
            break

        # ── Compute composite score ──────────────────────────────────────

        # 1. CPU cache hit rate — from vLLM's built-in external prefix cache
        #    metrics (logged by the engine as "External prefix cache hit rate").
        #    This measures: of tokens not found in GPU cache, what fraction was
        #    served from the CPU offload tier via the KVConnector.
        cpu_hit_rate = engine_external_prefix_hit_rate or 0.0

        # 2. TTFT ratio — behavioral consequence of offload quality.
        #    A better eviction policy keeps hot prefix blocks in CPU, so
        #    cache hits translate into less prefill work and lower TTFT.
        #    Ratio > 1 means the candidate beat the baseline's mean TTFT.
        mean_ttft_ms = metrics.get("mean_ttft_ms", 0.0)
        ttft_ratio = (
            BASELINE_TTFT_MS / mean_ttft_ms if mean_ttft_ms > 0 else 0.0
        )

        # 3. Throughput ratio (req/s vs baseline)
        throughput_ratio = metrics["request_throughput"] / BASELINE_THROUGHPUT

        # Gate: zero CPU hits means the cache is non-functional — the program
        # may be gaming TTFT by bypassing lookups rather than improving eviction.
        if cpu_hit_rate == 0.0:
            if _is_seed_iteration():
                msg = ("FATAL: No CPU evictions detected in seed iteration "
                       "(cpu_hit_rate=0). The workload does not generate "
                       "enough cache pressure to exercise the eviction "
                       "policy. Aborting evolution — adjust "
                       "gpu_memory_utilization, max_model_len, or workload "
                       "concurrency to force offloading.")
                logger.critical(msg)
                raise SystemExit(msg)
            return {
                "combined_score": 0.0,
                "cpu_hit_rate": cpu_hit_rate,
                "ttft_ratio": ttft_ratio,
                "throughput_ratio": throughput_ratio,
                "engine_external_prefix_hit_rate": engine_external_prefix_hit_rate,
                "engine_gpu_prefix_hit_rate": engine_gpu_prefix_hit_rate,
                "metrics": metrics,
                "error": "cpu_hit_rate=0: cache non-functional",
            }

        # Failure penalty: requests that error out may inflate TTFT/throughput
        # metrics by skipping real work — penalize proportionally.
        total_requests = metrics.get("total_requests", 1)
        failure_rate = metrics.get("failures", 0) / max(total_requests, 1)
        failure_multiplier = max(0.0, 1.0 - 2 * failure_rate)

        # TTFT is the primary goal (exp1 showed eviction quality shows up in
        # TTFT more than raw hit count). cpu_hit_rate validates the mechanism.
        combined_score = (
            0.50 * ttft_ratio
            + 0.30 * cpu_hit_rate
            + 0.20 * throughput_ratio
        ) * failure_multiplier

        if _is_seed_iteration():
            _SEED_VALIDATED = True
            logger.info("Seed iteration validated: cpu_hit_rate=%.4f, "
                        "combined_score=%.4f", cpu_hit_rate, combined_score)

        return {
            "combined_score": combined_score,
            "cpu_hit_rate": cpu_hit_rate,
            "ttft_ratio": ttft_ratio,
            "throughput_ratio": throughput_ratio,
            "engine_external_prefix_hit_rate": engine_external_prefix_hit_rate,
            "engine_gpu_prefix_hit_rate": engine_gpu_prefix_hit_rate,
            "metrics": metrics,
        }

    except subprocess.TimeoutExpired:
        return {"combined_score": 0.0, "error": "benchmark timed out"}
    except Exception as e:
        return {"combined_score": 0.0, "error": str(e)}
    finally:
        if server:
            _kill_server(server)
        _dump_server_log(server_all_lines)
        # Restore original handler so the repo stays clean between trials
        if EVOLVE_BACKUP.exists():
            EVOLVE_SRC.write_text(EVOLVE_BACKUP.read_text())
        time.sleep(3)  # let GPU memory release


# ── CLI entry point (for manual testing) ─────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <path-to-evolved-handler.py>")
        sys.exit(1)
    result = evaluate(sys.argv[1])
    print(json.dumps(result, indent=2))
