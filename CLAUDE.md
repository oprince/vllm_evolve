# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

## Architecture Overview

vLLM is a high-throughput LLM inference and serving engine. The codebase has two engine generations:

- **V1 engine** (`vllm/v1/`) — the current default, using a multiprocess architecture with ZMQ-based IPC between the engine frontend and the EngineCore (scheduler + workers). This is where active development happens.
- **Legacy engine** (`vllm/engine/`) — older synchronous/async engine, still present but deprecated.

### Key subsystems (V1 path)

```
Request flow:
  API server (vllm/entrypoints/) → AsyncLLM (v1/engine/async_llm.py)
    → EngineCoreClient (ZMQ) → EngineCore (v1/engine/core.py)
      → Scheduler (v1/core/sched/) → Executor → Worker → ModelRunner
```

- **Entrypoints** (`vllm/entrypoints/`): OpenAI-compatible API server, Anthropic Messages API, gRPC, CLI (`vllm serve`, `vllm bench`), offline `LLM` class.
- **Config** (`vllm/config/`): Dataclass-based configs (one file per concern: `model.py`, `parallel.py`, `cache.py`, `scheduler.py`, etc.). `VllmConfig` is the top-level container.
- **Scheduler** (`vllm/v1/core/sched/`): Decides which requests to process each iteration, manages KV cache block allocation via `KVCacheManager`.
- **Executor** (`vllm/v1/executor/`): Spawns and manages worker processes. `MultiProcExecutor` for local multi-GPU, `RayExecutor` for distributed.
- **Worker / ModelRunner** (`vllm/v1/worker/`): Each GPU gets a Worker. `GPUModelRunner` handles input preparation, model forward pass, and sampling.
- **Model implementations** (`vllm/model_executor/models/`): ~250+ model files. Each model registers itself via the registry (`registry.py`).
- **Layers** (`vllm/model_executor/layers/`): Reusable building blocks — linear layers, attention, RMSNorm, rotary embeddings, quantization wrappers, MoE (`fused_moe/`).
- **Attention backends** (`vllm/v1/attention/backends/`): FlashAttention, FlashInfer, Triton, MLA, etc. Selected at runtime based on hardware and model type.
- **Compilation** (`vllm/compilation/`): `torch.compile` integration with piecewise CUDA graph capture.
- **Distributed** (`vllm/distributed/`): Tensor/pipeline/expert parallelism primitives, custom all-reduce, KV transfer for disaggregated serving.
- **Platforms** (`vllm/platforms/`): Hardware abstraction — CUDA, ROCm, TPU, XPU, CPU.
- **Multimodal** (`vllm/multimodal/`): Image/audio/video input processing and encoder management.
- **Speculative decoding** (`vllm/v1/spec_decode/`): Draft model, EAGLE, n-gram, DFlash proposers.
- **LoRA** (`vllm/lora/`): Multi-LoRA adapter support.
- **Rust components** (`rust/`): Tool/reasoning parsers, tokenizer utilities, chat rendering — exposed as PyO3 extensions.
- **C++/CUDA kernels** (`csrc/`): Custom attention, quantization, MoE, and all-reduce kernels.

### Adding a new model

New model files go in `vllm/model_executor/models/`. Register in `registry.py`. Models use layers from `vllm/model_executor/layers/` rather than raw PyTorch modules to get quantization, TP, and compilation support for free.

## Build & Development Commands

```bash
# Environment setup
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements/lint.txt
pre-commit install

# Install (Python-only changes — fast, uses precompiled binaries)
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto

# Install (if editing C++/CUDA code — full build)
uv pip install -e . --torch-backend=auto

# Run tests
uv pip install -r requirements/test/cuda.in
.venv/bin/python -m pytest tests/path/to/test_file.py -v

# Linting (pre-commit handles ruff, clang-format, typos, mypy, SPDX headers)
pre-commit run --all-files
pre-commit run ruff-check --all-files
pre-commit run mypy-3.12 --all-files --hook-stage manual
```

## Important Conventions

- **SPDX headers required** on all Python files:
  ```python
  # SPDX-License-Identifier: Apache-2.0
  # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
  ```
- **Signed-off-by** is auto-appended to commits by the commit-msg hook.
- **Line length**: 88 characters (enforced by ruff).
- **Docstrings**: Google-style (`Args:`/`Returns:`/`Raises:`), not Sphinx `:param:` style.
- **Never use system python** — always `uv` and `.venv/bin/python`.
- **Environment variables**: Defined in `vllm/envs.py` with type annotations.
