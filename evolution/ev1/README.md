# Running the Evolve Process

## 1. Install dependencies

```bash
uv venv --python 3.12
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e ".[skydiscover]" --torch-backend=auto
```

## 2. Install inference-perf (optional)

Only needed if using `EVOLVE_BENCH_BACKEND=inference_perf`:

```bash
uv pip install -e "git+https://github.ibm.com/AI4SYS/inference-perf.git@4343d0ee273fbbcd1724715541112228d86c2118#egg=inference_perf"
```

## 3. Create `.env` at the project root

The `.env` file is not committed. Create it manually:

```bash
cat > .env << 'EOF'
# === REQUIRED ===

# Dataset for benchmarking
DATASET_PATH=/path/to/ShareGPT_V3_unfiltered_cleaned_split.json
DATASET_NAME=sharegpt

# Baseline metrics from Phase 1 (LRU seed run)
EVOLVE_BASELINE_THROUGHPUT=1.05
EVOLVE_BASELINE_TTFT_MS=51.84

# LLM API key (used by skydiscover to call the mutation LLM)
OPENAI_API_KEY=<your-litellm-or-openai-key>

# === SERVER / MODEL ===

EVOLVE_MODEL=NousResearch/Hermes-3-Llama-3.1-8B
EVOLVE_SERVER_PORT=8000
EVOLVE_KV_OFFLOAD_SIZE=16
EVOLVE_KV_OFFLOAD_BACKEND=cpu
EVOLVE_EVICTION_POLICY=evolved
EVOLVE_NUM_PROMPTS=30
EVOLVE_FILENAME=evolved.py

# === OPTIONAL ===

VLLM_PYTHON=.venv/bin/python
EVOLVE_GPU_MEM_UTIL=0.9
EVOLVE_MAX_CONCURRENCY=4
EVOLVE_SERVER_TIMEOUT=180

# Benchmark backend: "vllm_bench" (default) or "inference_perf"
EVOLVE_BENCH_BACKEND=vllm_bench

# Path to inference-perf config file (only used when EVOLVE_BENCH_BACKEND=inference_perf)
# Defaults to evolution/ev1/config_inference_perf_swe.yml
INFERENCE_PERF_CONFIG=/path/to/config_inference_perf.yml
EOF
```

Adjust `DATASET_PATH`, `OPENAI_API_KEY`, and `EVOLVE_MODEL` for your environment.

## 4. Run the evolution

### Option A: Python wrapper (recommended)

```bash
source .venv/bin/activate

python evolution/ev1/run_skydiscover.py \
  --config evolution/ev1/active_config_skydiscover.yaml \
  --output skydiscover_output/ \
  --search adaevolve \
  --iterations 50
```

### Option B: skydiscover CLI directly

```bash
skydiscover-run \
  vllm/v1/kv_offload/cpu/policies/evolved.py \
  evolution/ev1/evolve_eval.py \
  -c evolution/ev1/active_config_skydiscover.yaml \
  -o skydiscover_output/ \
  -s adaevolve \
  --codebase . \
  -l DEBUG
```

### Option C: Quick validation (3 iterations)

```bash
python evolution/ev1/run_skydiscover.py \
  --config evolution/ev1/active_config_skydiscover.yaml \
  --output test_output/ \
  --search best_of_n \
  --iterations 3
```

## 5. Resume from checkpoint

```bash
python evolution/ev1/run_skydiscover.py \
  --config evolution/ev1/active_config_skydiscover.yaml \
  --output skydiscover_output/ \
  --checkpoint skydiscover_output/checkpoints/checkpoint_50
```

## Requirements

- GPU node with enough VRAM to run the model
- The model must be downloadable (or already cached in `~/.cache/huggingface`)
- Port 8000 (or `EVOLVE_SERVER_PORT`) must be free — the evaluator starts/stops a vLLM server each iteration

## How it works

1. `run_skydiscover.py` calls `skydiscover.run_discovery()` with the seed policy and evaluator
2. Each iteration, skydiscover asks the LLM to mutate `vllm/v1/kv_offload/cpu/policies/evolved.py`
3. `evolve_eval.py` deploys the candidate, starts vLLM via `serve.py`, runs a benchmark, and returns a composite score
4. Score = `0.50 * ttft_ratio + 0.30 * cpu_hit_rate + 0.20 * throughput_ratio`
5. Best candidates survive; evolution repeats

## Config notes

The LLM used for mutations is configured in `active_config_skydiscover.yaml` under `llm.models`. The current config uses `claude-sonnet-4-6` via a LiteLLM proxy. Change the model name and `api_base` if using a different endpoint.
