# MiMo V2.5 Dual-DGX-Spark Crash Analysis

## Problem

A single OpenAI-compatible multimodal (image + text) request with an extreme
`max_tokens` value (999078) caused the entire two-node vLLM service to crash,
requiring a full ~12-minute cold restart.

## Root cause chain

```
Client sends max_tokens: 999078  (essentially the entire 1M context window)
  → No server-side cap → request passes validation
    → Worker GPU runs _merge_multimodal_embeddings
      → index_put_ overruns tensor bounds → cudaErrorNotPermitted
        → torch.AcceleratorError caught by except RuntimeError
          → re-raised as ValueError("Error during index put operation")
            → RayTaskError propagates to EngineCore.step()
              → _process_engine_step() has no try/except
                → run_busy_loop() has no try/except
                  → run_engine_core() except Exception → _send_engine_dead()
                    → API server shuts down
```

## Three sub-issues

| # | Issue | Layer |
|---|-------|-------|
| 1 | No server-side `max_tokens` cap — client can request nearly the entire 1M context as output | `vllm/v1/engine/input_processor.py` |
| 2 | Per-request error kills the entire EngineCore — no error isolation | `vllm/v1/engine/core.py` |
| 3 | CUDA error message swallowed by generic fallback in `_merge_multimodal_embeddings` | `vllm/model_executor/models/utils.py` |

## Possible fixes

### Fix 1 — Clamp `max_tokens` in the input processor

This version of vLLM (0.21.1rc1.dev85) does **not** have a `--max-tokens` CLI
argument. The clamp must be done in code. After the existing default-value
logic in `InputProcessor._process_request` (`input_processor.py`), add:

```python
_max_tokens_limit = int(os.environ.get("MAX_TOKENS_LIMIT", "32768"))
if (sampling_params.max_tokens is not None
        and sampling_params.max_tokens > _max_tokens_limit):
    sampling_params.max_tokens = _max_tokens_limit
```

This caps per-request `max_tokens` to `MAX_TOKENS_LIMIT` (default 32768)
before any GPU tensors are allocated.

### Fix 2 — Error isolation in `EngineCore.step()`

Wrap `future.result()` inside the `with log_error_detail / log_iteration_details`
block with a `try/except` that catches known per-request error patterns
(e.g. `ValueError` from multimodal merge) and aborts the affected batch via the
existing `self.abort_requests()` path instead of letting the exception crash
the engine.

### Fix 3 — Better CUDA diagnostic in `_merge_multimodal_embeddings`

Replace the generic `raise ValueError("Error during index put operation")` with
a CUDA-aware branch that preserves the GPU error text, making diagnosis easier
and giving Fix 2 a detectable pattern to match.

## Original crash excerpt

```
EngineCore encountered a fatal error.

Traceback (most recent call last):
  File "vllm/v1/engine/core.py", line 1152, in run_engine_core
  File "vllm/v1/engine/core.py", line 1193, in run_busy_loop
  File "vllm/v1/engine/core.py", line 1232, in _process_engine_step
  File "vllm/v1/engine/core.py", line 445, in step
    model_output = self.model_executor.sample_tokens(grammar_output)
  File "vllm/v1/executor/ray_executor.py", line 449, in sample_tokens
  File "vllm/v1/executor/ray_executor.py", line 467, in _execute_dag
    output = refs[0].get()

ray.exceptions.RayTaskError(ValueError): ray::RayWorkerWrapper()
torch.AcceleratorError: CUDA error: operation not permitted

  File "vllm/v1/worker/gpu_model_runner.py", line 3386, in _preprocess
    inputs_embeds_scheduled = self.model.embed_input_ids(...)
  File "vllm/model_executor/models/interfaces.py", line 404, in embed_input_ids
    return _merge_multimodal_embeddings(...)
  File "vllm/model_executor/models/utils.py", line 492, in _merge_multimodal_embeddings
    raise ValueError("Error during index put operation") from e

ValueError: Error during index put operation
EngineCore: Shutting down Ray distributed executor.
APIServer: Application shutdown complete.
```

## Notes

- The `--max-tokens` CLI flag does **not** exist in vLLM 0.21.1rc1.dev85.
  Any fix must patch the Python source directly.
- The recipe documentation recommends a client cap of 32,768 output tokens.
- Client has since been corrected to cap at 32,768.
- Runtime-tested patch scripts are in `patches/` (see git history).
