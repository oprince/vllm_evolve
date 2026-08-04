#!/usr/bin/env python3
"""
Entry point for running SkyDiscover (AdaEvolve) on the labcpu manager.

Usage:
    python evolution/run_skydiscover.py
    python evolution/run_skydiscover.py --search adaevolve
    python evolution/run_skydiscover.py --search evox --output my_output/
    python evolution/run_skydiscover.py --checkpoint skydiscover_output/checkpoints/checkpoint_50

Environment variables (see evolve_eval.py for full list):
    EVOLVE_MODEL               - model name (default: NousResearch/Hermes-3-Llama-3.1-8B)
    EVOLVE_BASELINE_THROUGHPUT - baseline req/s from Phase 1
    EVOLVE_BASELINE_TTFT_MS    - baseline mean TTFT (ms) from Phase 1
    DATASET_NAME               - benchmark dataset name (default: sharegpt)
    DATASET_PATH               - path to dataset file (falls back to SHAREGPT_PATH)
    EVOLVE_NUM_PROMPTS         - prompts per benchmark (default: 30)
    EVOLVE_SERVER_PORT         - vLLM server port (default: 8000)
    EVOLVE_KV_OFFLOAD_SIZE     - KV offload size in GB
"""

import argparse
import os
import pathlib
import shutil
import sys
import time

import yaml
from dotenv import load_dotenv

HERE = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent

load_dotenv(PROJECT_ROOT / ".env")

KV_OFFLOADING_BACKEND = os.environ.get("EVOLVE_KV_OFFLOAD_BACKEND", "cpu")
EVOLVE_FILENAME = os.environ.get("EVOLVE_FILENAME", "evolved.py")
# Evolve the in-tree vLLM CPU offloading policy directly
EVOLVE_SRC = PROJECT_ROOT / "vllm" / "v1" / "kv_offload" / "cpu" / "policies" / EVOLVE_FILENAME


def check_evolve_model_available(config_path: str) -> None:
    """Verify the evolve LLM is reachable before starting evolution."""
    import requests

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    llm_cfg = cfg.get("llm", {})
    models = llm_cfg.get("models", [])
    api_base = llm_cfg.get("api_base", "")
    api_key = llm_cfg.get("api_key", os.environ.get("OPENAI_API_KEY", ""))

    # Resolve ${OPENAI_API_KEY} placeholder
    if api_key and api_key.startswith("${") and api_key.endswith("}"):
        env_var = api_key[2:-1]
        api_key = os.environ.get(env_var, "")

    if not models:
        print("ERROR: No models configured in llm.models in the config.")
        sys.exit(1)

    model_name = models[0].get("name", "")
    model_api_base = models[0].get("api_base", api_base)

    if not model_name:
        print("ERROR: No model name found in config.")
        sys.exit(1)

    if not model_api_base:
        print("ERROR: No api_base found in config.")
        sys.exit(1)

    print(f"Checking evolve model availability: {model_name} @ {model_api_base}")

    # Try a minimal completion request using the same sampling parameters
    # that skydiscover will use, to catch parameter conflicts early
    url = f"{model_api_base.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
    }
    # Include sampling params from config to detect conflicts (e.g. Bedrock
    # rejects requests with both temperature and top_p)
    if "temperature" in llm_cfg:
        payload["temperature"] = llm_cfg["temperature"]
    if "top_p" in llm_cfg:
        payload["top_p"] = llm_cfg["top_p"]

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        if resp.status_code == 404:
            print(f"ERROR: Model '{model_name}' not found at {model_api_base}")
            print(f"  HTTP {resp.status_code}: {resp.text[:200]}")
            sys.exit(1)
        elif resp.status_code == 401 or resp.status_code == 403:
            print(f"ERROR: Authentication failed for {model_api_base}")
            print(f"  HTTP {resp.status_code}: {resp.text[:200]}")
            print("  Check OPENAI_API_KEY environment variable.")
            sys.exit(1)
        elif resp.status_code >= 500:
            print(f"ERROR: Evolve model endpoint unavailable (server error)")
            print(f"  HTTP {resp.status_code}: {resp.text[:200]}")
            sys.exit(1)
        elif resp.status_code == 400:
            print(f"ERROR: Evolve model rejected the request (bad parameters)")
            print(f"  HTTP {resp.status_code}: {resp.text[:500]}")
            print("  Check config sampling params (temperature/top_p) — "
                  "some providers reject both simultaneously.")
            sys.exit(1)
        elif resp.status_code >= 400:
            print(f"ERROR: Evolve model request failed")
            print(f"  HTTP {resp.status_code}: {resp.text[:200]}")
            sys.exit(1)
        print(f"  Model '{model_name}' is available.")
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Cannot connect to evolve model endpoint: {model_api_base}")
        print("  The LLM service may be down or unreachable.")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print(f"ERROR: Timeout connecting to evolve model endpoint: {model_api_base}")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to check evolve model availability: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run SkyDiscover (AdaEvolve) to evolve the KV-offloading manager"
    )
    parser.add_argument(
        "--output", type=str, default=str(PROJECT_ROOT / "skydiscover_output"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--config", type=str, default=str(HERE / "config_skydiscover.yaml"),
        help="Path to SkyDiscover config YAML",
    )
    parser.add_argument(
        "--search", type=str, default=None,
        help="Search algorithm (adaevolve, evox, topk, beam_search, best_of_n)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint directory to resume from",
    )
    parser.add_argument(
        "--target-score", type=float, default=None,
        help="Stop early if this score is reached",
    )
    parser.add_argument(
        "--iterations", type=int, default=None,
        help="Number of iterations (overrides config)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="LLM model override (e.g. gcp/gemini-3-flash-preview)",
    )
    parser.add_argument(
        "--no-monitor", dest="monitor", action="store_false", default=True,
        help="Disable the monitoring dashboard",
    )
    args = parser.parse_args()

    dataset_path = os.environ.get("DATASET_PATH")
    if not dataset_path:
        print("ERROR: DATASET_PATH environment variable must be set.")
        print("  export DATASET_PATH=/path/to/your_dataset.json")
        sys.exit(1)

    baseline_throughput = float(os.environ.get("EVOLVE_BASELINE_THROUGHPUT", "0"))
    if baseline_throughput == 0:
        print("WARNING: Baseline metrics not set. Run Phase 1 first:")
        print("  bash evolution/baseline/run_baseline.sh")
        print("  Then set EVOLVE_BASELINE_THROUGHPUT and EVOLVE_BASELINE_TTFT_MS")
        print()
        resp = input("Continue with baseline=1.0 (scores will be raw ratios)? [y/N] ")
        if resp.lower() != "y":
            sys.exit(1)

    try:
        from skydiscover import run_discovery
    except ImportError:
        print("ERROR: skydiscover not installed. Run:")
        print('  pip install -e ".[skydiscover]"')
        sys.exit(1)

    initial_program = str(EVOLVE_SRC)
    evaluator = str(HERE / "evolve_eval.py")
    search = args.search or "adaevolve"

    print("=" * 60)
    print(f"SkyDiscover ({search}) run")
    print("=" * 60)
    print(f"  Initial program    : {initial_program}")
    print(f"  Evaluator          : {evaluator}")
    print(f"  Config             : {args.config}")
    print(f"  Output             : {args.output}")
    print(f"  Search algorithm   : {search}")
    print(f"  Baseline throughput: {os.environ.get('EVOLVE_BASELINE_THROUGHPUT', '1.0')}")
    print(f"  Baseline TTFT      : {os.environ.get('EVOLVE_BASELINE_TTFT_MS', '1.0')}")
    print(f"  Dataset name       : {os.environ.get('DATASET_NAME', 'sharegpt')}")
    print(f"  Dataset path       : {dataset_path}")
    print(f"  Model              : {os.environ.get('EVOLVE_MODEL', 'NousResearch/Hermes-3-Llama-3.1-8B')}")
    print(f"  Num prompts        : {os.environ.get('EVOLVE_NUM_PROMPTS', '30')}")
    print(f"  kv offload size    : {os.environ.get('EVOLVE_KV_OFFLOAD_SIZE', '8')}")
    print(f"  kv offload backend : {os.environ.get('EVOLVE_KV_OFFLOAD_BACKEND', 'labcpu')}")
    print(f"  kv offload policy  : {os.environ.get('EVOLVE_EVICTION_POLICY', 'evolved')}")
    print(f"  Monitor            : {args.monitor}")
    print("=" * 60)

    # ── Save previous logs before starting ────────────────────────────────
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = PROJECT_ROOT / "save_logs" / f"run_{timestamp}"
    dirs_to_save = [
        PROJECT_ROOT / "vllm_server_logs",
        pathlib.Path(args.output) / "logs",
    ]
    # Also save the main evolution log (.err file from bsub)
    main_logs = list(PROJECT_ROOT.glob("evolution_run_*.err")) + \
                list(PROJECT_ROOT.glob("evolution_run_*.log"))

    has_logs = any(d.exists() and any(d.iterdir()) for d in dirs_to_save) \
               or main_logs
    if has_logs:
        save_dir.mkdir(parents=True, exist_ok=True)
        for src_dir in dirs_to_save:
            if src_dir.exists() and any(src_dir.iterdir()):
                dst = save_dir / src_dir.name
                shutil.copytree(src_dir, dst)
                print(f"  Saved {src_dir} -> {dst}")
        for log_file in main_logs:
            shutil.copy2(log_file, save_dir / log_file.name)
            print(f"  Saved {log_file.name} -> {save_dir}")
        print(f"  Previous logs saved to: {save_dir}")
    print()

    # ── Pre-flight: verify evolve model is reachable ─────────────────────
    check_evolve_model_available(args.config)
    print()

    result = run_discovery(
        initial_program=initial_program,
        evaluator=evaluator,
        config=args.config,
        iterations=args.iterations,
        output_dir=args.output,
        search=args.search,
        model=args.model,
        cleanup=False,
    )

    print()
    print("=" * 60)
    print("Evolution complete!")
    print(f"  Best score      : {result.best_score:.4f}")
    if result.initial_score is not None:
        print(f"  Initial score   : {result.initial_score:.4f}")
        improvement = result.best_score - result.initial_score
        print(f"  Improvement     : {improvement:+.4f}")
    if result.output_dir:
        print(f"  Output          : {result.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
